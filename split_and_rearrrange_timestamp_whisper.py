import os
import json
import csv
import shutil
import threading
import numpy as np
import torch
import torchaudio
from scipy.io.wavfile import write
from transformers import WhisperProcessor, WhisperModel
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ProcessPoolExecutor, as_completed
from torch.amp import autocast

# ---------------------------
# Global placeholders for Whisper processor and model
# These will be initialized in each worker process
# ---------------------------
processor = None
model = None


def init_worker():
    """
    Initialize Whisper Transformer (medium model) in each worker process.
    This ensures each parallel worker has its own model copy loaded into GPU.
    """
    global processor, model
    if processor is None or model is None:
        processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
        model = WhisperModel.from_pretrained("openai/whisper-medium").to("cuda").eval()
        print(f"Worker {os.getpid()} initialized Whisper Transformer (medium).\n")


def extract_segment_features(waveform, sr, segments):
    """
    Extract features for each audio segment:
    - Slice the waveform based on start/end timestamps
    - Encode with Whisper encoder using mixed precision (autocast)
    - Aggregate segment representation using mean + max pooling
    Returns: stacked feature vectors for all valid segments
    """
    feats = []
    for seg in segments:
        s, e = int(seg['start'] * sr), int(seg['end'] * sr)

        # Skip invalid or empty segments
        if e <= s:
            print(f"⚠️  Skipping invalid segment: start={s/sr}, end={e/sr}")
            continue
        slice_np = waveform[:, s:e].squeeze().cpu().numpy()
        if slice_np.size == 0:
            print(f"⚠️  Skipping empty segment: start={s/sr}, end={e/sr}")
            continue

        try:
            # Prepare features for Whisper
            inputs = processor(slice_np, sampling_rate=sr, return_tensors="pt")
            input_feats = inputs.input_features.to('cuda')

            # Encoder forward pass
            with torch.no_grad(), autocast('cuda'):
                enc_out = model.encoder(input_feats).last_hidden_state.cpu()[0]

            # Mean + Max pooling for robustness
            mean_f = enc_out.mean(dim=0, keepdim=True)
            max_f = enc_out.max(dim=0, keepdim=True)[0]
            combined = torch.cat([mean_f, max_f], dim=1).numpy()[0]
            feats.append(combined)

        except Exception as e:
            print(f"❌ Failed to extract features for segment: {e}")
            continue

    if not feats:
        raise ValueError("No valid segments for feature extraction.")
    return np.vstack(feats)


def compute_similarity_chain(features):
    """
    Compute a greedy similarity chain:
    - Start with the segment most similar to all others (highest mean similarity)
    - Iteratively pick the next segment most similar to the last one
    - Produces an ordering of segments maximizing local similarity
    """
    sim = cosine_similarity(features)
    order = [int(sim.mean(axis=1).argmax())]  # start with most "central" segment
    rem = set(range(sim.shape[0])) - set(order)

    while rem:
        last = order[-1]
        nxt = max(rem, key=lambda i: sim[last, i])  # pick next most similar
        order.append(nxt)
        rem.remove(nxt)
    return order


def reorder_single(item):
    """
    Reorder a single audio file based on computed similarity chain:
    - Load waveform and resample to 16kHz
    - Extract features per segment
    - Reorder segments according to similarity chain
    - Concatenate reordered audio segments
    - Normalize and save as WAV
    Fallback: If feature extraction fails, copy the original audio
    """
    audio_path = item['audio']
    out_path = item['output']
    segments = item['segments']

    # Skip if already processed
    if os.path.exists(out_path):
        return f"Skipped (already exists): {os.path.basename(audio_path)}"

    # Load waveform
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000

    # If too few segments, just copy the file
    if len(segments) <= 2:
        shutil.copy(audio_path, out_path)
        return f"Copied (1 segment): {os.path.basename(audio_path)}"

    # Ensure last segment ends at total duration
    total_dur = waveform.shape[1] / sr
    if segments and abs(segments[-1]['end'] - total_dur) > 1e-3:
        segments[-1]['end'] = total_dur

    # Extract features and compute order
    try:
        feats = extract_segment_features(waveform, sr, segments)
    except ValueError as e:
        print(f"❌ Skipping {os.path.basename(audio_path)} due to empty features: {e}")
        shutil.copy(audio_path, out_path)
        return f"Copied fallback: {os.path.basename(audio_path)}"
    order = compute_similarity_chain(feats)

    # Rebuild waveform in new order
    audio_np = waveform.numpy().squeeze()
    orig_len = audio_np.shape[0]
    parts = [audio_np[int(segments[i]['start']*sr):int(segments[i]['end']*sr)] for i in order]
    new_audio = np.concatenate(parts)

    # Match original length
    if new_audio.shape[0] > orig_len:
        new_audio = new_audio[:orig_len]
    elif new_audio.shape[0] < orig_len:
        new_audio = np.pad(new_audio, (0, orig_len-new_audio.shape[0]), mode='constant')

    # Normalize to [-1, 1]
    new_audio = new_audio / np.max(np.abs(new_audio))
    write(out_path, sr, (new_audio * 32767).astype('int16'))
    return f"Processed: {os.path.basename(audio_path)}"


def process_all_from_csv(csv_path, input_folder, output_folder, max_workers=2):
    """
    Process and reorder all files listed in a CSV:
    - Read utterance IDs and word-level timestamps from CSV
    - Build processing items with paths + segments
    - Run parallel processing with ProcessPoolExecutor
    - Start a monitor thread that prints progress every minute
    - After processing, verify missing files and copy them
    """
    os.makedirs(output_folder, exist_ok=True)

    # Load CSV (utterance_id + segments as JSON dict)
    items = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for utt, ts in reader:
            ts_map = json.loads(ts)
            segs = [{'start': v[0], 'end': v[1]} for k, v in sorted(ts_map.items(), key=lambda x: int(x[0]))]
            items.append({
                'audio': os.path.join(input_folder, f"{utt}.wav"),
                'output': os.path.join(output_folder, f"{utt}.wav"),
                'segments': segs
            })

    total = len(items)
    stop_event = threading.Event()

    # Progress monitor (runs in background thread)
    def monitor():
        while not stop_event.is_set():
            count = len([f for f in os.listdir(output_folder) if f.lower().endswith('.wav')])
            print(f"Processed {count}/{total} files so far...")
            stop_event.wait(60)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()

    # Run parallel reordering
    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker) as exe:
        futures = {exe.submit(reorder_single, item): item for item in items}
        for fut in as_completed(futures):
            _ = fut.result()  # each returns a status string

    # Stop monitor after work is done
    stop_event.set()
    monitor_thread.join()

    # Verify missing files and copy them if needed
    print("\nVerifying all files are present in output folder...")
    missing = []
    input_files = {f for f in os.listdir(input_folder) if f.lower().endswith('.wav')}
    output_files = {f for f in os.listdir(output_folder) if f.lower().endswith('.wav')}
    for f in input_files:
        if f not in output_files:
            print(f"Missing: {f} — copying from input to output.")
            shutil.copy(os.path.join(input_folder, f), os.path.join(output_folder, f))
            missing.append(f)
    print(f"\nVerification complete. {len(missing)} files were missing and copied over.")


# ---------------------------
# Main entry point
# ---------------------------
if __name__ == "__main__":
    input_folder = "/app/datasets/vpc/T25-1/data/train-clean-360/wav_ori"
    output_folder = "/app/datasets/vpc/T25-1/data/train-clean-360/wav_word_rearrrange_whisper"
    csv_file = "/app/datasets/vpc/T8-5/data/train-clean-360/word_timestamps.csv"

    process_all_from_csv(csv_file, input_folder, output_folder)

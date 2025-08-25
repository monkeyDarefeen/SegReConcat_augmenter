import whisper
import os
import glob
import json
import csv
import shutil
import librosa
import numpy as np
import networkx as nx
from pydub import AudioSegment
from librosa.sequence import dtw
from concurrent.futures import ProcessPoolExecutor, as_completed

# Global variable for each worker process (initialized once per process)
_whisper_model = None


def init_worker():
    """
    Initializer for each worker process.
    Loads the Whisper model once (large model, GPU accelerated).
    """
    global _whisper_model
    _whisper_model = whisper.load_model("large").to("cuda")
    print("Loaded Whisper model in process:", os.getpid())


def transcribe_audio(input_audio, model):
    """
    Transcribes an audio file using Whisper and extracts word-level timestamps.

    Args:
        input_audio (str): Path to input audio file.
        model: Whisper model object.

    Returns:
        list of dicts: Each entry contains {"word", "start", "end"}.
    """
    result = model.transcribe(input_audio, word_timestamps=True)
    words = []
    for segment in result["segments"]:
        for word_info in segment["words"]:
            words.append({
                "word": word_info["word"].strip(),
                "start": word_info["start"],
                "end": word_info["end"]
            })
    return words


def split_audio_in_memory(input_audio, words):
    """
    Splits an audio file into word-level segments (AudioSegments) using timestamps.

    Args:
        input_audio (str): Path to input audio file.
        words (list): Whisper transcription output with timestamps.

    Returns:
        tuple of dicts:
            - segments: {key -> AudioSegment}
            - text_map: {key -> word string}
            - timestamps: {key -> (start, end)}
    """
    full_audio = AudioSegment.from_file(input_audio)
    segments, text_map, timestamps = {}, {}, {}
    total_duration_ms = len(full_audio)

    # Handle initial silence before the first word (if any).
    if words and words[0]["start"] > 0:
        key = "silence_0"
        start = 0
        end = int(words[0]["start"] * 1000)
        segments[key] = full_audio[start:end]
        text_map[key] = ""  # Silence segment contains no word.
        timestamps[key] = (0, words[0]["start"])

    # Slice audio into word-based segments.
    for i, w in enumerate(words):
        start = int(w["start"] * 1000)
        end = int(words[i+1]["start"] * 1000) if i < len(words) - 1 else total_duration_ms

        if end <= start:  # Skip invalid durations.
            continue

        key = f"{w['word']}_{i}"
        segments[key] = full_audio[start:end]
        text_map[key] = w["word"]
        timestamps[key] = (w["start"], end / 1000.0)

    return segments, text_map, timestamps


def get_mfcc_from_segment(segment, n_mfcc=13, max_length=100):
    """
    Converts an AudioSegment into MFCC features (fixed length).

    Args:
        segment (AudioSegment): Input audio segment.
        n_mfcc (int): Number of MFCC coefficients.
        max_length (int): Max number of frames (pad/truncate).

    Returns:
        np.ndarray: MFCC feature matrix (n_mfcc x max_length).
    """
    samples = np.array(segment.get_array_of_samples()).astype(np.float32)
    sr = segment.frame_rate
    mfcc = librosa.feature.mfcc(y=samples, sr=sr, n_mfcc=n_mfcc)

    # Pad or truncate to fixed length
    if mfcc.shape[1] < max_length:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_length - mfcc.shape[1])), mode="constant")
    else:
        mfcc = mfcc[:, :max_length]
    return mfcc


def compute_similarity_from_segments(segments):
    """
    Computes similarity scores between all pairs of word segments using DTW on MFCC features.

    Args:
        segments (dict): {key -> AudioSegment}

    Returns:
        dict: {(key1, key2) -> similarity score}
    """
    mfccs = {k: get_mfcc_from_segment(seg) for k, seg in segments.items()}
    sim = {}
    keys = list(segments.keys())

    for i, k1 in enumerate(keys):
        for j, k2 in enumerate(keys):
            if i != j:
                D, _ = dtw(mfccs[k1], mfccs[k2], metric="euclidean")
                sim[(k1, k2)] = 1 / (1 + D[-1, -1])  # Similarity is inverse of DTW distance
    return sim


def tsp_reorder(keys, similarity):
    """
    Reorders segments using a greedy Traveling Salesman heuristic
    based on similarity between segments.

    Args:
        keys (list): List of segment keys.
        similarity (dict): Pairwise similarity scores.

    Returns:
        list: Reordered list of keys.
    """
    initial_silence = None
    keys_for_reorder = keys.copy()

    # Keep silence at the beginning if present
    for k in keys:
        if k.startswith("silence_"):
            initial_silence = k
            keys_for_reorder.remove(k)
            break

    if not keys_for_reorder:
        return [initial_silence] if initial_silence else []

    # Greedy nearest-neighbor reordering
    visited = [0]
    remaining = set(range(1, len(keys_for_reorder)))
    while remaining:
        last = keys_for_reorder[visited[-1]]
        next_idx = min(remaining, key=lambda i: 1 - similarity.get((last, keys_for_reorder[i]), 0))
        visited.append(next_idx)
        remaining.remove(next_idx)
    new_order = [keys_for_reorder[i] for i in visited]

    if initial_silence:
        new_order = [initial_silence] + new_order
    return new_order


def merge_audio_in_memory(order, segments):
    """
    Concatenates a list of AudioSegments into one continuous audio.

    Args:
        order (list): List of keys specifying concatenation order.
        segments (dict): {key -> AudioSegment}

    Returns:
        AudioSegment: Combined audio.
    """
    combined = AudioSegment.silent(duration=0)
    for k in order:
        combined += segments[k]
    return combined


def main_process(input_audio, output_folder, speaker_id, chapter_id, utterance_filename):
    """
    Full pipeline for one audio file:
      - Transcribe
      - Split into segments
      - Compute similarities
      - Reorder
      - Merge and save
      - Collect transcripts and word timestamps

    Returns:
        tuple: (utterance_id, speaker_id, chapter_id, transcript, word_timestamps)
    """
    try:
        global _whisper_model

        # 1. Transcribe
        words = transcribe_audio(input_audio, _whisper_model)
        original_transcript = " ".join([w["word"] for w in words])

        # Early exit if too few words
        if not words or len(words) <= 1:
            out_dir = output_folder
            os.makedirs(out_dir, exist_ok=True)
            output_audio_path = os.path.join(out_dir, utterance_filename)
            full_audio = AudioSegment.from_file(input_audio)
            full_audio.export(output_audio_path, format="wav")

            rearranged_word_timestamps = {
                0: (words[0]["start"], words[0]["end"])
            } if words else {}

            return (os.path.splitext(utterance_filename)[0].replace("-", "-"),
                    speaker_id, chapter_id, original_transcript, rearranged_word_timestamps)

        # 2. Split audio into segments
        segments, text_map, timestamps = split_audio_in_memory(input_audio, words)

        # 3. Compute similarity + reorder
        keys = list(segments.keys())
        similarity = compute_similarity_from_segments(segments)
        new_order = tsp_reorder(keys, similarity)

        # 4. Merge reordered audio
        final_audio = merge_audio_in_memory(new_order, segments)
        rearranged_transcript = " ".join([text_map[k] for k in new_order if text_map[k]])
        rearranged_word_timestamps = {
            idx: timestamps[k] for idx, k in enumerate(new_order)
        }

        # 5. Save output audio
        out_dir = output_folder
        os.makedirs(out_dir, exist_ok=True)
        output_audio_path = os.path.join(out_dir, utterance_filename)
        final_audio.export(output_audio_path, format="wav")

        return (os.path.splitext(utterance_filename)[0].replace("-", "-"),
                speaker_id, chapter_id, rearranged_transcript, rearranged_word_timestamps)

    except Exception as e:
        with open("processing_errors.txt", "a") as err_log:
            err_log.write(f"Error in main_process for {input_audio}: {e}\n")
        return None


def process_single_file(audio_file, input_folder, output_folder):
    """
    Worker wrapper for one audio file.
    Extracts speaker/chapter IDs and calls main_process().
    """
    try:
        base = os.path.basename(audio_file)
        parts = base.split("-")
        if len(parts) < 3:
            print(f"Skipping {audio_file}, filename does not match the expected format.")
            return None

        speaker_id = parts[0]
        chapter_id = parts[1]
        utterance_filename = base
        result = main_process(audio_file, output_folder, speaker_id, chapter_id, utterance_filename)
        return result
    except Exception as e:
        with open("processing_errors.txt", "a") as err_log:
            err_log.write(f"Error processing file {audio_file}: {e}\n")
        return None


def process_all_files(input_folder, output_folder, timestamp_csv):
    """
    Processes all .wav files in a dataset folder:
      - Parallel transcription & reordering
      - Saves word timestamps (CSV)
      - Saves global and per-chapter transcripts
    """
    csv_file_path = timestamp_csv
    transcript_global_path = os.path.join(output_folder, "speaker_transcripts.txt")

    # Initialize outputs
    if not os.path.exists(csv_file_path):
        os.makedirs(output_folder, exist_ok=True)
        with open(csv_file_path, "w", newline="") as f:
            csv.writer(f).writerow(["utterance_id", "word_timestamps"])

    if os.path.exists(transcript_global_path):
        os.remove(transcript_global_path)

    audio_files = glob.glob(os.path.join(input_folder, "*.wav"))
    print(f"Found {len(audio_files)} audio files to process in {input_folder}.")

    # Parallel execution
    results = []
    with ProcessPoolExecutor(max_workers=4, initializer=init_worker) as executor:
        futures = [executor.submit(process_single_file, file, input_folder, output_folder)
                   for file in audio_files]
        for future in as_completed(futures):
            try:
                res = future.result()
                if res is not None:
                    results.append(res)
            except Exception as e:
                with open("processing_errors.txt", "a") as err_log:
                    err_log.write(f"Future exception: {e}\n")

    # Write CSV and global transcript
    with open(csv_file_path, "a", newline="", encoding="utf-8") as csvfile, \
         open(transcript_global_path, "a", encoding="utf-8") as transcript_file:
        writer = csv.writer(csvfile)
        for utterance_id, speaker_id, chapter_id, transcript, timestamps in results:
            writer.writerow([utterance_id, json.dumps(timestamps)])
            transcript_file.write(f"{utterance_id} {transcript}\n")

    # Write per-chapter transcripts
    chapter_data = {}
    for utterance_id, speaker_id, chapter_id, transcript, _ in results:
        key = (speaker_id, chapter_id)
        chapter_data.setdefault(key, []).append((utterance_id, transcript))

    for (speaker_id, chapter_id), entries in chapter_data.items():
        chapter_transcript_path = os.path.join(
            output_folder, speaker_id, chapter_id, f"{speaker_id}-{chapter_id}.trans.txt"
        )
        os.makedirs(os.path.dirname(chapter_transcript_path), exist_ok=True)
        with open(chapter_transcript_path, "a", encoding="utf-8") as f:
            for utterance_id, transcript in entries:
                f.write(f"{utterance_id} {transcript}\n")

    print("Processing complete.")


if __name__ == "__main__":
    #Example of testing on B3
    list_ori = [
        "/app/datasets/vpc/B3/data/train-clean-360/wav"
    ]
    list_re = [
        "/app/datasets/vpc/B3/data/train-clean-360/wav_re"
    ]
    csv_file = ["/app/datasets/vpc/T8-5/data/train-clean-360/word_timestamps.csv"]
    for i in range(len(list_ori)):
        process_all_files(list_ori[i], list_re[i],csv_file[i])

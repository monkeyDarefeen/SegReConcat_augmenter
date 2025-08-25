import os
import csv
import json
import whisper
from concurrent.futures import ProcessPoolExecutor, as_completed

# Global Whisper model for workers
def init_worker():
    global _model
    _model = whisper.load_model("large")


def transcribe_to_timestamps(audio_path):
    """
    Transcribe audio with Whisper and return mapping of word indices to [start, end] timestamps.
    """
    result = _model.transcribe(audio_path, word_timestamps=True)
    timestamps = {}
    idx = 0
    for segment in result.get("segments", []):
        for w in segment.get("words", []):
            timestamps[str(idx)] = [w["start"], w["end"]]
            idx += 1
    return os.path.splitext(os.path.basename(audio_path))[0], timestamps


def generate_csv(input_folders, output_csv, max_workers=4):
    """
    Walk through each input folder, transcribe .wav files, and write word timestamp CSV.

    Args:
        input_folders (List[str]): List of directories containing .wav files.
        output_csv (str): Path to the output CSV file.
        max_workers (int, optional): Number of parallel workers.
    """
    # Prepare CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["utterance_id", "word_timestamps"]);

    # Collect all audio paths
    audio_paths = []
    for folder in input_folders:
        for root, _, files in os.walk(folder):
            for fname in files:
                if fname.lower().endswith('.wav'):
                    audio_paths.append(os.path.join(root, fname))

    # Set up parallel transcription
    if max_workers is None:
        max_workers = os.cpu_count() - 1 or 1

    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker) as exe:
        futures = {exe.submit(transcribe_to_timestamps, p): p for p in audio_paths}
        with open(output_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for fut in as_completed(futures):
                try:
                    utt_id, ts_map = fut.result()
                    writer.writerow([utt_id, json.dumps(ts_map)])
                    print(f"Wrote timestamps for {utt_id}")
                except Exception as e:
                    print(f"Error processing {futures[fut]}: {e}")


if __name__ == '__main__':
    # Example usage:
    folders = ['/app/datasets/vpc/T8-5/data/train-clean-360/wav_ori'
    ]
    output_csv = '/app/datasets/vpc/T8-5/data/train-clean-360/word_timestamps.csv'
    generate_csv(folders, output_csv)

import os
import csv
import ast
import random
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
from tqdm import tqdm

# ---------------------------
# Max number of threads for parallel processing
# ---------------------------
MAX_THREADS = 32


def read_csv(csv_file):
    """
    Read the CSV file containing utterance IDs and word-level timestamps.
    - Each row: { 'utterance_id': ..., 'word_timestamps': ... }
    - word_timestamps is stored as a string representation of a Python dict, 
      so we use ast.literal_eval to safely parse it.
    Returns: dict {utterance_id: {word_index: (start, end), ...}}
    """
    timestamps = {}
    with open(csv_file, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            utterance_id = row['utterance_id']
            word_timestamps = ast.literal_eval(row['word_timestamps'])
            timestamps[utterance_id] = word_timestamps
    return timestamps


def process_audio_file(input_folder, output_folder, utterance_id, word_timestamps):
    """
    Process a single audio file:
    - Load the audio file with pydub
    - Slice it into word-level segments using timestamps
    - Shuffle the segments randomly
    - Concatenate them and save as a new WAV
    Returns: True if successful, False otherwise
    """
    input_file_path = os.path.join(input_folder, f"{utterance_id}.wav")
    output_file_path = os.path.join(output_folder, f"{utterance_id}.wav")

    # Ensure audio file exists
    if not os.path.exists(input_file_path):
        print(f"[ERROR] Audio file not found: {input_file_path}")
        return False

    # Load audio with pydub
    try:
        audio = AudioSegment.from_wav(input_file_path)
    except Exception as e:
        print(f"[ERROR] Could not load {input_file_path}: {e}")
        return False

    reshuffled_segments = []
    audio_length_ms = len(audio)  # total duration in ms

    # Loop through word timestamps and extract segments
    word_items = list(word_timestamps.items())
    for idx, (key, (start, end)) in enumerate(word_items):
        start_ms = int(start * 1000)
        end_ms = int(end * 1000)

        # If this is the last word and end exceeds audio, take until end of file
        if idx == len(word_items) - 1 and end_ms > audio_length_ms:
            segment = audio[start_ms:]
        else:
            # Skip invalid segments (out of bounds or reversed)
            if start_ms >= audio_length_ms or end_ms > audio_length_ms or start_ms >= end_ms:
                continue
            segment = audio[start_ms:end_ms]

        reshuffled_segments.append(segment)

    if not reshuffled_segments:
        print(f"[WARNING] No valid segments for: {utterance_id}")
        return False

    # Randomize order of segments
    random.shuffle(reshuffled_segments)

    # Concatenate segments into one audio
    reshuffled_audio = AudioSegment.empty()
    for segment in reshuffled_segments:
        reshuffled_audio += segment

    # Export reshuffled audio
    reshuffled_audio.export(output_file_path, format="wav")
    return True


def reshuffle_audio(input_folder, output_folder, csv_file):
    """
    Main function to reshuffle audio dataset:
    - Read timestamps from CSV
    - Process all files in parallel using threads
    - Show progress with tqdm
    - Verify output folder has all expected files
    """
    timestamps = read_csv(csv_file)
    os.makedirs(output_folder, exist_ok=True)

    failures = []

    # ThreadPoolExecutor for parallel audio processing
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = {
            executor.submit(process_audio_file, input_folder, output_folder, utterance_id, word_timestamps): utterance_id
            for utterance_id, word_timestamps in timestamps.items()
        }

        # Collect results with tqdm progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Audio"):
            result = future.result()
            if not result:
                failures.append(futures[future])

    # After processing, verify all files are present
    verify_and_copy_missing(input_folder, output_folder, timestamps.keys())

    # Print summary of failures
    if failures:
        print(f"\n[SUMMARY] Failed to process {len(failures)} files:")
        for fid in failures:
            print(f" - {fid}")


def verify_and_copy_missing(input_folder, output_folder, utterance_ids):
    """
    Ensure all expected files exist in the output folder.
    - If any are missing, copy the original audio file as a fallback.
    """
    for utterance_id in utterance_ids:
        output_file_path = os.path.join(output_folder, f"{utterance_id}.wav")
        input_file_path = os.path.join(input_folder, f"{utterance_id}.wav")
        if not os.path.exists(output_file_path):
            print(f"[MISSING] {utterance_id}.wav")
            if os.path.exists(input_file_path):
                shutil.copy(input_file_path, output_file_path)
                print(f"[COPIED] {utterance_id}.wav from input to output")


# ---------------------------
# Main entry point
# ---------------------------
if __name__ == "__main__":
    input_folder = "/app/datasets/vpc/T8-5/data/train-clean-360/wav"
    output_folder = "/app/datasets/vpc/T8-5/data/train-clean-360/wav_word_shuffled"
    csv_file = "/app/datasets/vpc/T8-5/data/train-clean-360/word_timestamps.csv"

    reshuffle_audio(input_folder, output_folder, csv_file)

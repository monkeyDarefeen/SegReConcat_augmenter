import os
import random
from multiprocessing import Pool, cpu_count
from pydub import AudioSegment

# Directories
DIRS = ['/app/datasets/vpc/T25-1/data/train-clean-360/wav_ori', '/app/datasets/vpc/T25-1/data/train-clean-360/wav_word_shuffled']
OUTPUT_DIR = '/app/datasets/vpc/T25-1/data/train-clean-360/wav_ori_word_shuffled_concan'
NUM_WORKERS = cpu_count()  # Use all available CPUs

def process_file(filename):
    try:
        # Full paths to the three input versions of the file
        file_paths = [os.path.join(d, filename) for d in DIRS]

        # Check if all files exist
        if not all(os.path.exists(p) for p in file_paths):
            print(f" Skipping {filename} - not all files exist.")
            return

        # Shuffle and load audio
        random.shuffle(file_paths)
        combined = AudioSegment.empty()
        for path in file_paths:
            combined += AudioSegment.from_wav(path)

        # Export to output directory
        output_path = os.path.join(OUTPUT_DIR, filename)
        combined.export(output_path, format="wav")
        print(f" Processed: {filename}")
    except Exception as e:
        print(f" Error processing {filename}: {e}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # List of filenames in the first directory
    all_files = [f for f in os.listdir(DIRS[0]) if f.lower().endswith(".wav")]

    print(f" Found {len(all_files)} .wav files.")
    print(f" Starting processing with {NUM_WORKERS} workers...")

    # Create pool and map work
    with Pool(NUM_WORKERS) as pool:
        pool.map(process_file, all_files)

    print(" Done processing all audio files!")

if __name__ == "__main__":
    main()

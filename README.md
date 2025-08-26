# SegReConcat: A Data Augmentation Method for Voice Anonymization Attack
## Paper link: {will be updated soon}
This Repo Represents SegReConcat Data Augmentation code from the paper
SegReConcat: A Data Augmentation Method for Voice Anonymization Attack
by Ridwan Arefeen, Xiaoxiao Miao, Rong Tong, Aik Beng, Simon See

## The Repo Contains the following files
- word_timestamp.csv -> Sample how the timstamp file should be.
- create_timestamp.py -> It creates word_timestamp.csv file by taking input of a wav file folder and output folder path and file name for csv
- SegReConcat_timestamp.py -> it follows the papers implementation of SegReConcat, it takes wav folder input path. output folder path and timestamp csv for all wav files.
- split_and_rearrange_timestamp_MFCC.py -> One variation of SegRe part where rearrangement is done using MFCC  similarity. Note that it doesn't concanate, you need to use concat_audio.py to concanate separately.
- split_and_rearrange_timestamp_whisper.py -> Another variation of SegRe part where rearrangement is done using Whisper encoder similarity. Note that it doesn't concanate, you need to use concat_audio.py to concanate separately.
- concat_audio.py -> It takes two or more wav folder paths of where wav needs to be concanated and another folder path to output the audio. It assumes that both wav folder path container wav file using same name.

## Pre-requests
- Please install following packages :
`pip install pydub librosa whisper tqdm torch torchaudio sklearn transformers scipy`
- Make sure audio files are in `wav` format
- The scrips except a folder path where wav files will be available.
- The SegReConcat uses timestamp for each word in csv format, a sample has been provided as word_timestamp.csv

## How to run
### Step 1
- Create Timestamps for your audio files using `create_timestamp.py`, edit folders and output_csv and give your audio folder path / output csv path

### Step 2
- To run SegReConcat pipeline, run `SegReConcat_timestamp.py` by editing input_folder, output_folder, csv_file path.

### Step 3 [to test different rearrangement system]
- For MFCC rearrangement step, run `split_and_rearrange_timestamp_MFCC.py` by editing input_folder, output_folder, csv_file path.
- Output should be Audio which is rearranged but not concanated.
- To Run concatenation part of the SegReConcat, run `concat_audio.py` by editing DIRS [folders that contain audio to concatenate (one for original audio, another for rearranged audio folder)] , OUTPUT_DIR [where the audio will be saved]

### Step 4
- For Whisper encoding similarity rearrangement step, run `split_and_rearrrange_timestamp_whisper.py` by editing input_folder, output_folder, csv_file path.
- Output should be Audio which is rearranged but not concanated.
- To Run concatenation part of the SegReConcat, run `concat_audio.py` by editing DIRS [folders that contain audio to concatenate (one for original audio, another for rearranged audio folder)] , OUTPUT_DIR [where the audio will be saved]

### By following Step 1-2, you will get audio following SegReConcat paper, where we use random shuffle for words rearrangement. 
### To test with different rearrangement procedure, `Step 1 -> Step 3/4` can be used.
#### Please note that python file comments are written using Local AI.


## License
MIT License

## Acknowledgements
This work is supported by Ministry of Education, Singapore, under its Academic Research Tier 1 (R-R13-A405-0005). Xiaoxiao Miao is the corresponding author and this work was conducted while she was at SIT. 

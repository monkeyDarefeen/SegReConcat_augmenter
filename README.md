# SegReConcat_augmenter
This Repo Represents SegReConcat Data Augmentation

## The Repo Contains the following files
- word_timestamp.csv -> Sample how the timstamp file should be.
- create_timestamp.py -> It creates word_timestamp.csv file by taking input of a wav file folder and output folder path and file name for csv
- SegReConcat_timestamp.py -> it follows the papers implementation of SegReConcat, it takes wav folder input path. output folder path and timestamp csv for all wav files.
- split_and_rearrange_timestamp_MFCC.py -> One variation of SegRe part where rearrangement is done using MFCC  similarity. Note that it doesn't concanate, you need to use concat_audio.py to concanate separately.
- split_and_rearrange_timestamp_whisper.py -> Another variation of SegRe part where rearrangement is done using Whisper encoder similarity. Note that it doesn't concanate, you need to use concat_audio.py to concanate separately.
- concat_audio.py -> It takes two or more wav folder paths of where wav needs to be concanated and another folder path to output the audio. It assumes that both wav folder path container wav file using same name.


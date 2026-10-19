import os
import sys
import warnings
from pathlib import Path
import boto3
import botocore
from botocore.config import Config
import librosa

warnings.filterwarnings("ignore")

script_dir = Path(__file__).resolve().parent
PROJECT_ROOT = script_dir.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from core.spectrogram import generate_mel_spectrogram

PT_TIMESTAMPS = ['2025-11-09_23-58']
NODE = 'rpi_port_townsend'
BUCKET = 'audio-orcasound-net'
OUTPUT_BASE = PROJECT_ROOT / 'dataset' / 'port_townsend'

s3 = boto3.client('s3', config=Config(signature_version=botocore.UNSIGNED))

def process_pt_bout(timestamp):
    print(f"\nProcessing {timestamp}")
    local_folder = OUTPUT_BASE / timestamp
    local_folder.mkdir(parents=True, exist_ok=True)

    prefix = f"{NODE}/flac/{timestamp}"

    try:
        paginator = s3.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=BUCKET, Prefix=prefix)

        print(f"Scanning {prefix}")

        for page in page_iterator:
            if 'Contents' not in page:
                print(f"No data found for timestamp {timestamp} in {NODE}")
                return

            for obj in page['Contents']:
                file_key = obj['Key']
                file_name = file_key.split('/')[-1]
                local_path = f"temp_{file_name}"

                print(f"Downloading {file_name}...")
                s3.download_file(BUCKET, file_key, local_path)

                stream_duration = 300
                total_duration = librosa.get_duration(path=local_path)

                print(f"Total file duration: {total_duration/60:.2f} minutes")

                for start_time in range(0, int(total_duration), stream_duration):
                    y, sr = librosa.load(local_path, sr=44100,
                                         offset=start_time,
                                         duration=stream_duration)

                    duration = librosa.get_duration(y=y, sr=sr)
                    if duration == 0:
                        break

                    slice_len = 2.0  # 2.0s slice aligned with standard model input
                    for start in range(0, int(duration), int(slice_len)):
                        if start + slice_len > duration:
                            break

                        y_slice = y[int(start * sr): int((start + slice_len) * sr)]
                        img, _ = generate_mel_spectrogram(y_slice, sr=sr)

                        img_name = local_folder / f"slice_{file_name}_{start}.png"
                        img.save(img_name)

                if os.path.exists(local_path):
                    os.remove(local_path)
                print(f"Generated slices for {file_name}")

    except Exception as e:
        print(f"Error processing {timestamp}: {e}")

if __name__ == "__main__":
    for ts in PT_TIMESTAMPS:
        process_pt_bout(ts)
    print("\nDone. Slices are saved in dataset/port_townsend")

import boto3
import os
from botocore import UNSIGNED
from botocore.config import Config
from collections import defaultdict
from pathlib import Path

BUCKET_NAME = 'acoustic-sandbox'
script_dir = Path(__file__).resolve().parent
BASE_OUTPUT_DIR = script_dir.parent / "Data" / "Acoustic-Sound-Data" / "mirror"

MAX_FILES_PER_FOLDER = 4
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

folder_counts = defaultdict(int)

def download_files():
    print(f"Searching for folder containing 'audio_5' in s3://{BUCKET_NAME}")
  
    paginator = s3.get_paginator('list_objects_v2')
  
    for page in paginator.paginate(Bucket=BUCKET_NAME):
        if 'Contents' not in page:
            continue
          
        for obj in page['Contents']:
            key = obj['Key']
          
            if key.lower().endswith('.wav'):
                # Folder names in the key path
                relative_path = os.path.dirname(key)
                path_parts = relative_path.split('/')

                # The folder has to be in the path exactly
                if "audio_5" not in path_parts:
                    continue

                if folder_counts[relative_path] >= MAX_FILES_PER_FOLDER:
                    continue 

                filename = os.path.basename(key)
                target_dir = os.path.join(BASE_OUTPUT_DIR, relative_path)
                local_wav_path = os.path.join(target_dir, filename)

                if not os.path.exists(target_dir):
                    os.makedirs(target_dir, exist_ok=True)
              
                if os.path.exists(local_wav_path):
                    folder_counts[relative_path] += 1
                    continue

                print(f"Downloading: {key} -> {local_wav_path}")

                try:
                    s3.download_file(BUCKET_NAME, key, local_wav_path)
                    folder_counts[relative_path] += 1
                except Exception as e:
                    print(f"  Error: {e}")

    print("\nDone")

if __name__ == "__main__":
    download_files()
import boto3
import os
import sys
from botocore import UNSIGNED
from botocore.config import Config
from collections import defaultdict
from pathlib import Path

script_dir = Path(__file__).resolve().parent
PROJECT_ROOT = script_dir.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from core.spectrogram import save_spectrogram

BUCKET_NAME = 'acoustic-sandbox'
BASE_OUTPUT_DIR = PROJECT_ROOT / "Data" / "Acoustic-Sound-Data" / "mirror"

MAX_FILES_PER_FOLDER = 4
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

# Files processed per folder
folder_counts = defaultdict(int)

def mirror_bucket():
    print(f"Mirroring s3://{BUCKET_NAME} (max {MAX_FILES_PER_FOLDER} files per folder)")
  
    paginator = s3.get_paginator('list_objects_v2')
  
    for page in paginator.paginate(Bucket=BUCKET_NAME):
        if 'Contents' not in page:
            continue
          
        for obj in page['Contents']:
            key = obj['Key']
          
            if key.lower().endswith('.wav'):
                relative_path = os.path.dirname(key)
              
                # Skip folders that hit the limit
                if folder_counts[relative_path] >= MAX_FILES_PER_FOLDER:
                    continue

                filename = os.path.basename(key)
                image_filename = filename.replace('.wav', '.png').replace('.WAV', '.png')
              
                target_dir = os.path.join(BASE_OUTPUT_DIR, relative_path)
            
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir, exist_ok=True)
                    print(f"Created {target_dir}")
              
                local_wav = os.path.join(target_dir, filename)
                local_img = os.path.join(target_dir, image_filename)
              
                if os.path.exists(local_img):
                    folder_counts[relative_path] += 1
                    continue

                print(f"Processing: {filename} ({folder_counts[relative_path] + 1}/{MAX_FILES_PER_FOLDER})")
              
                try:
                    s3.download_file(BUCKET_NAME, key, local_wav)
                    save_spectrogram(local_wav, local_img)
                    os.remove(local_wav)
                    folder_counts[relative_path] += 1
                except Exception as e:
                    print(f"Error processing {key}: {e}")

    print("\nDone")

if __name__ == "__main__":
    mirror_bucket()

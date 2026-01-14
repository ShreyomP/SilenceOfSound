import boto3
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from botocore import UNSIGNED
from botocore.config import Config
from collections import defaultdict
from pathlib import Path

# --- CONFIGURATION ---
BUCKET_NAME = 'acoustic-sandbox'
# This gets the folder where your script is currently located
script_dir = Path(__file__).resolve().parent
# This goes "up" one level and then into Code/Orca
BASE_OUTPUT_DIR = script_dir.parent / "Data" / "Acoustic-Sound-Data" / "mirror"

MAX_FILES_PER_FOLDER = 50  # Your new limit
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))


# To keep track of how many files we've processed in each directory
folder_counts = defaultdict(int)


def create_spectrogram(audio_path, image_path):
   """Converts audio to a standardized 224x224 grayscale spectrogram."""
   try:
       y, sr = librosa.load(audio_path, duration=2.0)
       S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
       S_dB = librosa.power_to_db(S, ref=np.max)
      
       plt.figure(figsize=(12, 6))
       librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=20000)
       plt.colorbar(format='%+2.0f dB')
       plt.title('Silence in the Sound: Raw Hydrophone Spectrogram (Port Townsend)')
       plt.xlabel('Time (seconds)')
       plt.ylabel('Frequency (Hz)')
       plt.tight_layout()
       plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
       plt.close()

# 4. Plotting
                       



   except Exception as e:
       print(f"  ⚠️ Skipping {audio_path}: {e}")


def run_capped_mirror():
   print(f"🚀 Mirroring s3://{BUCKET_NAME} (Max {MAX_FILES_PER_FOLDER} per folder)...")
  
   paginator = s3.get_paginator('list_objects_v2')
  
   for page in paginator.paginate(Bucket=BUCKET_NAME):
       if 'Contents' not in page:
           continue
          
       for obj in page['Contents']:
           key = obj['Key']
          
           if key.lower().endswith('.wav'):
               relative_path = os.path.dirname(key)
              
               # Check if this specific folder has hit the limit
               if folder_counts[relative_path] >= MAX_FILES_PER_FOLDER:
                   continue # Skip to the next file in the bucket


               filename = os.path.basename(key)
               image_filename = filename.replace('.wav', '.png').replace('.WAV', '.png')
              
               target_dir = os.path.join(BASE_OUTPUT_DIR, relative_path)

            
               if not os.path.exists(target_dir):
                    os.makedirs(target_dir, exist_ok=True)
                    print(f"created {target_dir}")
              
               local_wav = os.path.join(target_dir, filename)
               local_img = os.path.join(target_dir, image_filename)
              
               # Skip if already processed
               if os.path.exists(local_img):
                   # Increment count even if it exists so we don't exceed limit
                   folder_counts[relative_path] += 1
                   continue


               print(f"📦 [{folder_counts[relative_path] + 1}/{MAX_FILES_PER_FOLDER}] in {relative_path}: {filename}")
              
               try:
                   s3.download_file(BUCKET_NAME, key, local_wav)
                   create_spectrogram(local_wav, local_img)
                   os.remove(local_wav)
                  
                   # Successfully processed - increment the counter
                   folder_counts[relative_path] += 1
                  
               except Exception as e:
                   print(f"  ❌ Error processing {key}: {e}")


   print("\n✨ Capped Mirroring Complete!")


if __name__ == "__main__":
   run_capped_mirror()

import boto3
import botocore
from botocore.config import Config
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings


warnings.filterwarnings("ignore")


# 1. Configuration
# These are the Port Townsend "Golden Timestamps" we identified
PT_TIMESTAMPS = ['2025-11-09_23-58'  ]
NODE = 'rpi_port_townsend'
BUCKET = 'audio-orcasound-net'
OUTPUT_BASE = 'dataset/port_townsend'
maxkey = 1000


# Setup AWS (Public)
s3 = boto3.client('s3', config=Config(signature_version=botocore.UNSIGNED))


def process_pt_bout(timestamp):
   print(f"\n--- Processing Bout: {timestamp} ---")
   local_folder = f"{OUTPUT_BASE}/{timestamp}"
   if not os.path.exists(local_folder): os.makedirs(local_folder)


   # We'll look for the first 5 segments in the HLS folder for this timestamp
   prefix = f"{NODE}/flac/{timestamp}"
  
   try:
       #response = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix, MaxKeys=maxkey)
       paginator = s3.get_paginator('list_objects_v2')
  
       # Create a "Page Iterator"
       page_iterator = paginator.paginate(Bucket=BUCKET, Prefix=prefix)


       file_count = 0
       print(f"Starting full bout scan for: {prefix}")


       for page in page_iterator:


           if 'Contents' not in page:
               print(f"No data found for timestamp {timestamp} in {NODE}")
               return


           for obj in page['Contents']:
               file_key = obj['Key']
               file_name = file_key.split('/')[-1]
               local_path = f"temp_{file_name}"
          


               # Download
               print(f"Downloading {file_name}...")
               s3.download_file(BUCKET, file_key, local_path)


               # Configuration for chunks
               stream_duration = 300 # Process 5 minutes at a time
               total_duration = librosa.get_duration(path=local_path)


               print(f"Total file duration: {total_duration/60:.2f} minutes")


               # Loop through the file in segments
               for  start_time in range(0, int(total_duration), stream_duration):
                   print(f"Processing segment starting at {start_time} seconds...")
          
                   # Load just this chunk
                   y, sr = librosa.load(local_path, sr=None,
                                       offset=start_time,
                                       duration=stream_duration)




                   # Process into 5-second slices
                   #y, sr = librosa.load(local_path, sr=None)
                   duration = librosa.get_duration(y=y, sr=sr)
                   if duration == 0:
                       break
              
                   for start in range(0, int(duration), 5):
                       # Ensure we have a full 5 seconds
                       if start + 5 > duration: break
                      
                       y_slice = y[int(start*sr) : int((start+5)*sr)]
                       S = librosa.feature.melspectrogram(y=y_slice, sr=sr, n_mels=128)
                       S_dB = librosa.power_to_db(S, ref=np.max)
                      
                       # Save the AI "Flashcard"
                       #plt.figure(figsize=(2, 2))
                       #librosa.display.specshow(S_dB, sr=sr)
                       #plt.axis('off')
                      


                       # 4. Plotting
                       plt.figure(figsize=(12, 6))
                       librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=20000)
                       plt.colorbar(format='%+2.0f dB')
                       plt.title('Silence in the Sound: Raw Hydrophone Spectrogram (Port Townsend)')
                       plt.xlabel('Time (seconds)')
                       plt.ylabel('Frequency (Hz)')
                       plt.tight_layout()


                       img_name = f"{local_folder}/slice_{file_name}_{start}.png"
                       plt.savefig(img_name, bbox_inches='tight', pad_inches=0)
                       plt.close()


              
               os.remove(local_path) # Clean up temp audio file
               print(f"Generated slices for {file_name}")


   except Exception as e:
       print(f"Error processing {timestamp}: {e}")


# Run the automation
for ts in PT_TIMESTAMPS:
   process_pt_bout(ts)


print("\nMISSION COMPLETE: Your Port Townsend Training Library is ready in /dataset/")


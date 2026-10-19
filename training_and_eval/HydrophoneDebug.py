import os
import sys
import time
import subprocess
from pathlib import Path
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import librosa
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from core.config import FFMPEG_BINARY
from core.spectrogram import generate_mel_spectrogram

class HydrophoneDebug:
    def __init__(self, node="rpi_port_townsend"):
        self.bucket_name = "audio-orcasound-net"
        self.prefix = f"{node}/hls/"
        self.ffmpeg_path = FFMPEG_BINARY
        self.s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    def get_latest_capture(self, output_path="debug_spec.png"):
        try:
            lookback_start = int(time.time()) - (24 * 60 * 60)
            start_after_key = f"{self.prefix}{lookback_start}.ts"

            print("Searching for latest audio segment")
            paginator = self.s3.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(
                Bucket=self.bucket_name,
                Prefix=self.prefix,
                StartAfter=start_after_key
            )

            latest_obj = None
            for page in page_iterator:
                if "Contents" in page:
                    latest_obj = page['Contents'][-1]

            if not latest_obj:
                print("No files found in the lookback period")
                return False

            print(f"Downloading {latest_obj['Key']}")
            self.s3.download_file(self.bucket_name, latest_obj['Key'], 'current_capture.ts')

            print("Converting to wav")
            subprocess.run([
                self.ffmpeg_path, '-i', 'current_capture.ts',
                '-ar', '44100', '-ac', '1',
                'current_capture.wav', '-y'
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            print("Generating spectrogram")
            y, sr = librosa.load('current_capture.wav', sr=44100, duration=2.0)
            img, _ = generate_mel_spectrogram(y, sr=sr)
            img.save(output_path)

            if os.path.exists('current_capture.ts'):
                os.remove('current_capture.ts')
            if os.path.exists('current_capture.wav'):
                os.remove('current_capture.wav')

            print(f"Saved spectrogram to {output_path}")
            return True

        except Exception as e:
            print(f"Error: {e}")
            return False

if __name__ == "__main__":
    debugger = HydrophoneDebug()
    if debugger.get_latest_capture():
        try:
            img = Image.open("debug_spec.png")
            img.show()
        except Exception:
            print("Spectrogram saved as debug_spec.png")
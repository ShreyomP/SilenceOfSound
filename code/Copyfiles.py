import os
import shutil
from pathlib import Path

# This gets the folder where your script is currently located
script_dir = Path(__file__).resolve().parent
# This goes "up" one level and then into Code/Orca
SOURCE_DIR = script_dir.parent / "Data" / "InferenceData" 
print(f"Absolute path to data: {SOURCE_DIR}")

#SOURCE_DIR = "data/mirror/acoustic-separation/dataset/train"
FILES = ["mixed.png", "noise.png", "orca.png"]

for folder_name in os.listdir(SOURCE_DIR):
    folder_path = os.path.join(SOURCE_DIR, folder_name)

    if os.path.isdir(folder_path):
        for file in FILES:
            src_file = os.path.join(folder_path, file)
            dst_file = os.path.join(os.path.splitext(file)[0], folder_name + ".png")

            if os.path.isfile(src_file):
                print(f'{src_file} and {dst_file}')
                shutil.copy2(src_file, dst_file)

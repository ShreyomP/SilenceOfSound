import os
import shutil
from pathlib import Path

def organize_files(source_root):
    # Source filename -> destination folder
    target_map = {
        'orca.png': 'Orca',
        'mixed.png': 'Mixed',
        'noise.png': 'Noise'
    }

    for folder in target_map.values():
        dest_dir = os.path.join(source_root, folder)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir, exist_ok=True)
            print(f"Created folder: {dest_dir}")

    for root, dirs, files in os.walk(source_root):
        for filename in files:
            if filename in target_map:
                # New name comes from the parent folder
                parent_folder_name = os.path.basename(root)

                dest_folder = target_map[filename]
                extension = os.path.splitext(filename)[1]
                new_filename = f"{parent_folder_name}{extension}"

                src_path = os.path.join(root, filename)
                dest_path = os.path.join(source_root,dest_folder, new_filename)

                try:
                    shutil.copy2(src_path, dest_path)
                    print(f"Copied: {src_path} -> {dest_path}")
                except Exception as e:
                    print(f"Error copying {src_path}: {e}")

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    SOURCE_DIR = script_dir.parent / "Data" / "Inference"
    organize_files(SOURCE_DIR)
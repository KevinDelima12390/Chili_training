import os
import sys
import urllib.request
import zipfile
from pathlib import Path

# Mendeley Data Public Zip Download URL for w9mr3vf56s version 1
DATASET_URL = "https://data.mendeley.com/public-api/zip/w9mr3vf56s/download/1"

def progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 / total_size)
        sys.stdout.write(f"\r[Download] Progress: {percent:.1f}% ({downloaded / 1_000_000:.1f} MB / {total_size / 1_000_000:.1f} MB)")
    else:
        sys.stdout.write(f"\r[Download] Downloaded: {downloaded / 1_000_000:.1f} MB (calculating size...)")
    sys.stdout.flush()

def main():
    script_dir = Path(__file__).parent
    workspace_dir = script_dir.parent
    
    zip_path = workspace_dir / "dataset.zip"
    extract_dir = workspace_dir / "dataset"
    
    print("=================================================================")
    print("      Chili Plant Leaf Disease & Stage Dataset Downloader        ")
    print("=================================================================")
    print(f"Dataset URL: {DATASET_URL}")
    print(f"Target Zip File: {zip_path}")
    print(f"Extraction Folder: {extract_dir}")
    print("=================================================================\n")

    # 1. Download the ZIP file
    print(f"[Download] Starting download from Mendeley Data...")
    try:
        urllib.request.urlretrieve(DATASET_URL, zip_path, reporthook=progress_hook)
        print("\n[Download] ✓ Download completed successfully!")
    except Exception as e:
        print(f"\n[Download] ✗ Error downloading dataset: {e}")
        print("Please check your internet connection or try copy-pasting the URL directly into your browser.")
        sys.exit(1)

    # 2. Extract the ZIP file
    print(f"[Extract] Extracting dataset to {extract_dir} ...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of files to show extraction progress
            files = zip_ref.namelist()
            total_files = len(files)
            print(f"[Extract] Found {total_files} files in the zip archive.")
            
            for idx, file in enumerate(files):
                zip_ref.extract(file, extract_dir)
                if idx % max(1, total_files // 20) == 0 or idx == total_files - 1:
                    percent = (idx + 1) * 100 / total_files
                    sys.stdout.write(f"\r[Extract] Progress: {percent:.1f}% ({idx + 1}/{total_files} files)")
                    sys.stdout.flush()
            print("\n[Extract] ✓ Extraction completed successfully!")
    except Exception as e:
        print(f"\n[Extract] ✗ Error extracting zip file: {e}")
        sys.exit(1)

    # 3. Clean up the downloaded ZIP file to save disk space
    if zip_path.exists():
        print(f"[Cleanup] Removing zip archive to free up space...")
        zip_path.unlink()
        print("[Cleanup] ✓ Removed dataset.zip")

    # 4. Scan the folder and display the structure
    print("\n[Structure] Scanning extracted dataset directory...")
    categories = [d for d in extract_dir.iterdir() if d.is_dir()]
    if categories:
        print("\nExtracted Categories:")
        for cat in categories:
            print(f" └──  {cat.name}/")
            subdirs = [sd for sd in cat.iterdir() if sd.is_dir()]
            for sd in subdirs:
                file_count = len(list(sd.glob("*")))
                print(f"      └── {sd.name}/ ({file_count} items)")
    else:
        file_count = len(list(extract_dir.glob("**/*")))
        print(f"No top-level folders found. Total files extracted: {file_count}")

    print("\n=================================================================")
    print(" ✓ Dataset is ready! You can now proceed to training.")
    print("=================================================================")

if __name__ == "__main__":
    main()

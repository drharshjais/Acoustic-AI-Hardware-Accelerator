import os
import zipfile
import requests
import pandas as pd
import librosa
import soundfile as sf

# Configuration
ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
ZIP_PATH = "ESC-50-master.zip"
EXTRACT_DIR = "ESC-50-master"
OUTPUT_DIR = "processed_dataset"
TARGET_SR = 16000 # 16 kHz sample rate for low-power edge processing

# Define our binary classes based on ESC-50 categories
THREAT_CLASSES = ['chainsaw']
SAFE_CLASSES = ['rain', 'crickets', 'chirping_birds', 'sea_waves', 'wind']

def download_and_extract():
    if not os.path.exists(EXTRACT_DIR):
        print("Downloading ESC-50 dataset (this is about 600MB and may take a few minutes)...")
        response = requests.get(ESC50_URL, stream=True)
        with open(ZIP_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print("Extracting files...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(".")
        os.remove(ZIP_PATH)
    else:
        print("Dataset already downloaded and extracted.")

def process_audio():
    # Setup output directories
    threat_dir = os.path.join(OUTPUT_DIR, "threat")
    safe_dir = os.path.join(OUTPUT_DIR, "safe")
    os.makedirs(threat_dir, exist_ok=True)
    os.makedirs(safe_dir, exist_ok=True)

    # Read the metadata
    meta_path = os.path.join(EXTRACT_DIR, "meta", "esc50.csv")
    audio_dir = os.path.join(EXTRACT_DIR, "audio")
    df = pd.read_csv(meta_path)

    print(f"Processing audio to {TARGET_SR}Hz Mono...")
    
    count = 0
    for index, row in df.iterrows():
        filename = row['filename']
        category = row['category']
        
        # Sort into threat or safe folders
        if category in THREAT_CLASSES:
            out_folder = threat_dir
        elif category in SAFE_CLASSES:
            out_folder = safe_dir
        else:
            continue # Skip irrelevant sounds 
            
        # Load and compress the audio
        file_path = os.path.join(audio_dir, filename)
        y, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
        
        # Save the new file
        out_path = os.path.join(out_folder, filename)
        sf.write(out_path, y, TARGET_SR)
        count += 1
        
        if count % 20 == 0:
            print(f"Processed {count} files...")

    print(f"\nSuccess! Processed {count} audio files into the '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    download_and_extract()
    process_audio()
import os
import librosa
import numpy as np

# Configuration
DATA_DIR = "processed_dataset"
TARGET_SR = 16000
N_MELS = 64
# 157 is the standard number of frames for a 5-second clip at 16kHz
MAX_LENGTH = 157 

def get_spectrogram(file_path):
    # Load audio
    y, sr = librosa.load(file_path, sr=TARGET_SR)
    
    # Extract Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Pad or truncate to ensure uniform size across all clips
    if mel_spec_db.shape[1] < MAX_LENGTH:
        pad_width = MAX_LENGTH - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :MAX_LENGTH]
        
    return mel_spec_db

def build_dataset():
    features = []
    labels = []
    
    threat_dir = os.path.join(DATA_DIR, "threat")
    safe_dir = os.path.join(DATA_DIR, "safe")
    
    print("Extracting features from THREAT files (Label: 1)...")
    for filename in os.listdir(threat_dir):
        if filename.endswith(".wav"):
            filepath = os.path.join(threat_dir, filename)
            spec = get_spectrogram(filepath)
            features.append(spec)
            labels.append(1) # 1 represents Chainsaw/Threat
            
    print("Extracting features from SAFE files (Label: 0)...")
    for filename in os.listdir(safe_dir):
        if filename.endswith(".wav"):
            filepath = os.path.join(safe_dir, filename)
            spec = get_spectrogram(filepath)
            features.append(spec)
            labels.append(0) # 0 represents Background Nature/Safe
            
    # Convert lists to NumPy arrays
    X = np.array(features)
    y = np.array(labels)
    
    # Reshape X to add a "channel" dimension (required for Convolutional Neural Networks)
    # Shape changes from (240, 64, 157) to (240, 64, 157, 1)
    X = X[..., np.newaxis]
    
    print(f"\nFeature matrix 'X' shape: {X.shape}")
    print(f"Labels vector 'y' shape: {y.shape}")
    
    # Save to disk
    np.save("X_features.npy", X)
    np.save("y_labels.npy", y)
    print("Successfully saved 'X_features.npy' and 'y_labels.npy' to the project folder.")

if __name__ == "__main__":
    build_dataset()
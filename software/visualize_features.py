import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Configuration
DATA_DIR = "processed_dataset"
TARGET_SR = 16000

def plot_spectrogram(file_path, title):
    # Load the processed audio
    y, sr = librosa.load(file_path, sr=TARGET_SR)
    
    # Generate the Mel Spectrogram
    # n_mels=64 means we are compressing the frequency range into 64 distinct bands
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=8000)
    
    # Convert power (amplitude squared) to Decibels (log scale)
    # Humans (and AI) perceive sound logarithmically, not linearly
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Plot it
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Pick one chainsaw file and one safe file to compare
    # Note: os.listdir gets the first file it finds in the directory
    threat_dir = os.path.join(DATA_DIR, "threat")
    safe_dir = os.path.join(DATA_DIR, "safe")
    
    threat_file = os.path.join(threat_dir, os.listdir(threat_dir)[0])
    safe_file = os.path.join(safe_dir, os.listdir(safe_dir)[0])
    
    print("Generating Spectrogram for Threat (Chainsaw)...")
    plot_spectrogram(threat_file, "Mel Spectrogram - Threat (Chainsaw)")
    
    print("Generating Spectrogram for Safe (Background Nature)...")
    plot_spectrogram(safe_file, "Mel Spectrogram - Safe (Background)")
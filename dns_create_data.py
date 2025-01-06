import os
import librosa
import soundfile as sf
import numpy as np

# Define directories
noisy_dir = "/data/ephraim/datasets/dns_test/no_reverb/noisy/"
clean_dir = "/data/ephraim/datasets/dns_test/no_reverb/clean/"
noises_dir = "/data/ephraim/datasets/dns_test/no_reverb/noises"

# # Create the output directory if it doesn't exist
# os.makedirs(noises_dir, exist_ok=True)

# Build a dictionary for clean files based on their suffix
clean_files = {file.split("_")[-1]: file for file in os.listdir(clean_dir) if file.endswith(".wav")}

# Iterate through noisy WAV files
for noisy_file in os.listdir(noisy_dir):
    if noisy_file.endswith(".wav"):
        # Extract the suffix (e.g., "fileid_X.wav")
        suffix = noisy_file.split("_")[-1]
        
        # Find the corresponding clean file using the suffix
        clean_file = clean_files.get(suffix)
        if clean_file:
            noisy_path = os.path.join(noisy_dir, noisy_file)
            clean_path = os.path.join(clean_dir, clean_file)
            
            # Load both noisy and clean WAV files
            noisy_audio, sr_noisy = librosa.load(noisy_path, sr=None)
            clean_audio, sr_clean = librosa.load(clean_path, sr=None)
            
            # Ensure sampling rates match
            if sr_noisy != sr_clean:
                raise ValueError(f"Sampling rate mismatch between {noisy_file} and {clean_file}")
            
            # Subtract clean from noisy to get the noise
            noise_audio = noisy_audio - clean_audio
            
            # Construct output file path
            noise_output_path = os.path.join(noises_dir, noisy_file)
            
            # Save the resulting noise audio
            sf.write(noise_output_path, noise_audio, sr_noisy)
            print(f"Processed {noisy_file} -> {noise_output_path}")
        else:
            print(f"No matching clean file found for {noisy_file}, skipping.")
import os
import librosa
import numpy as np
import json

def is_stationary(signal, sr, window_size=1.0, threshold=0.1):
    """
    Check if the noise signal is close to stationary.

    Args:
        signal (numpy.ndarray): 1D array representing the noise signal.
        sr (int): Sampling rate of the signal.
        window_size (float): Size of the window in seconds (default: 1 second).
        threshold (float): Threshold for relative standard deviation (RSD) to determine stationarity.

    Returns:
        bool: True if the noise is close to stationary, False otherwise.
        dict: Detailed statistics (mean RSD, variance RSD).
    """
    # Convert window size from seconds to samples
    window_samples = int(window_size * sr)

    # Divide the signal into overlapping windows
    num_windows = len(signal) // window_samples
    windows = [signal[i * window_samples:(i + 1) * window_samples] for i in range(num_windows)]

    # Calculate mean and variance for each window
    means = [np.mean(w) for w in windows]
    variances = [np.var(w) for w in windows]

    # Calculate relative standard deviation (RSD) for mean and variance
    rsd_mean = np.std(means) / np.mean(means) if np.mean(means) != 0 else 0
    rsd_variance = np.std(variances) / np.mean(variances) if np.mean(variances) != 0 else 0

    # Check if RSDs are below the threshold
    stationary = rsd_mean < threshold and rsd_variance < threshold

    # Return result and detailed statistics
    return bool(stationary), {"mean_rsd": float(rsd_mean), "variance_rsd": float(rsd_variance)}

def analyze_directory(input_dir, output_file, window_size=1.0, threshold=0.1):
    """
    Analyze all WAV files in a directory for stationarity and save results to a JSON file.

    Args:
        input_dir (str): Path to the directory containing WAV files.
        output_file (str): Path to the output JSON file.
        window_size (float): Size of the window in seconds (default: 1 second).
        threshold (float): Threshold for relative standard deviation (RSD) to determine stationarity.
    """
    results = {}

    # Process each WAV file in the directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".wav"):
            file_path = os.path.join(input_dir, file_name)
            
            try:
                # Load the WAV file
                signal, sr = librosa.load(file_path, sr=None)

                # Check stationarity
                stationary, stats = is_stationary(signal, sr, window_size, threshold)

                # Store the result
                results[file_name] = {
                    "stationary": stationary,  # Ensure this is a Python bool
                    "mean_rsd": stats["mean_rsd"],  # Ensure these are Python floats
                    "variance_rsd": stats["variance_rsd"]
                }
            except Exception as e:
                # Handle errors for specific files
                results[file_name] = {"error": str(e)}

    # Write results to JSON file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")

# Example usage
if __name__ == "__main__":
    input_directory = "/data/ephraim/datasets/DNS-Challenge_old/synth_exp_n/noise"  # Directory containing noise WAV files
    output_json = "/data/ephraim/datasets/DNS-Challenge_old/synth_exp_n/stationarity_results.json"  # Output JSON file
    analyze_directory(input_directory, output_json, window_size=1.0, threshold=0.4)
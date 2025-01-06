import os
import json
import shutil

def copy_stationary_noises_and_clean(json_file, noisy_dir, clean_dir, destination_noisy_dir, destination_clean_dir):
    """
    Copy stationary noise files and their corresponding clean files to new directories.

    Args:
        json_file (str): Path to the JSON file containing stationarity results.
        noisy_dir (str): Directory containing the noise files.
        clean_dir (str): Directory containing the clean files.
        destination_noisy_dir (str): Directory to store stationary noise files.
        destination_clean_dir (str): Directory to store corresponding clean files.

    Returns:
        None
    """
    try:
        # Load the JSON file
        with open(json_file, "r") as f:
            results = json.load(f)

        # Ensure the destination directories exist
        # os.makedirs(destination_noisy_dir, exist_ok=True)
        # os.makedirs(destination_clean_dir, exist_ok=True)

        # Copy stationary noise files and corresponding clean files
        for noisy_file, data in results.items():
            if "stationary" in data and data["stationary"]:
                # Paths for the noisy file
                noisy_source_path = os.path.join(noisy_dir, noisy_file)
                noisy_destination_path = os.path.join(destination_noisy_dir, noisy_file)

                # Find the corresponding clean file by its suffix
                suffix = noisy_file.split("_")[-1]  # Extract "fileid_X.wav"
                clean_file = next((f for f in os.listdir(clean_dir) if f.endswith(suffix)), None)

                # Paths for the clean file
                if clean_file:
                    clean_source_path = os.path.join(clean_dir, clean_file)
                    clean_destination_path = os.path.join(destination_clean_dir, clean_file)

                # Copy the noisy file
                if os.path.exists(noisy_source_path):
                    shutil.copy(noisy_source_path, noisy_destination_path)
                    print(f"Copied: {noisy_file} -> {destination_noisy_dir}")
                else:
                    print(f"Noise file not found: {noisy_source_path}")

                # Copy the clean file
                if clean_file and os.path.exists(clean_source_path):
                    shutil.copy(clean_source_path, clean_destination_path)
                    print(f"Copied: {clean_file} -> {destination_clean_dir}")
                else:
                    print(f"Corresponding clean file not found for {noisy_file}")

        print("Stationary noise and corresponding clean files have been copied.")

    except Exception as e:
        print(f"Error: {e}")

# Example usage
if __name__ == "__main__":
    json_file_path = "/data/ephraim/datasets/dns_test/no_reverb/stationarity_results.json"  # Path to the JSON file
    noisy_directory = "/data/ephraim/datasets/dns_test/no_reverb/noises"  # Directory with noise files
    clean_directory = "/data/ephraim/datasets/dns_test/no_reverb/clean"  # Directory with clean files
    destination_noisy_directory = "/data/ephraim/datasets/known_noise/undiff_exps/exp_n_real/noises/"  # Destination for stationary noises
    destination_clean_directory = "/data/ephraim/datasets/known_noise/undiff_exps/exp_n_real/cleans"  # Destination for corresponding clean files

    copy_stationary_noises_and_clean(
        json_file_path,
        noisy_directory,
        clean_directory,
        destination_noisy_directory,
        destination_clean_directory
    )
import os
import torchaudio

def process_signals(speech_dir, noise_dir, factors):
    """
    Loads signals from speech and noise directories, multiplies them by the given factors,
    and saves the results back to the respective directories with filenames prefixed by 'X_'.

    Args:
        speech_dir (str): Path to the directory containing speech files.
        noise_dir (str): Path to the directory containing noise files.
        factors (list): List of factors to multiply the signals by.
    """
    # Ensure the directories exist
    if not os.path.exists(speech_dir):
        raise FileNotFoundError(f"Speech directory does not exist: {speech_dir}")
    if not os.path.exists(noise_dir):
        raise FileNotFoundError(f"Noise directory does not exist: {noise_dir}")

    # Get list of files in each directory
    speech_files = [f for f in os.listdir(speech_dir) if f.endswith('.wav')]
    noise_files = [f for f in os.listdir(noise_dir) if f.endswith('.wav')]

    for speech_file, noise_file in zip(speech_files, noise_files):
        # Load the speech and noise signals
        speech_path = os.path.join(speech_dir, speech_file)
        noise_path = os.path.join(noise_dir, noise_file)

        speech_signal, speech_rate = torchaudio.load(speech_path)
        noise_signal, noise_rate = torchaudio.load(noise_path)

        # Ensure the sample rates match
        if speech_rate != noise_rate:
            raise ValueError(f"Sample rates do not match: {speech_file} ({speech_rate}), {noise_file} ({noise_rate})")

        # Ensure the signals have the same length
        min_length = min(speech_signal.size(1), noise_signal.size(1))
        speech_signal = speech_signal[:, :min_length]
        noise_signal = noise_signal[:, :min_length]

        # Multiply and save the signals
        for factor in factors:
            speech_combined_signal = speech_signal * factor
            noise_combined_signal = noise_signal * factor

            # Save the speech signal
            speech_output_filename = f"X{factor}_{speech_file}"
            speech_output_path = os.path.join(speech_dir, speech_output_filename)
            torchaudio.save(speech_output_path, speech_combined_signal, sample_rate=speech_rate)
            print(f"Saved speech: {speech_output_path}")

            # Save the noise signal
            noise_output_filename = f"X{factor}_{noise_file}"
            noise_output_path = os.path.join(noise_dir, noise_output_filename)
            torchaudio.save(noise_output_path, noise_combined_signal, sample_rate=noise_rate)
            print(f"Saved noise: {noise_output_path}")


# Example usage
speech_directory = "/data/ephraim/datasets/known_noise/undiff_exps/exp_n_much/cleans"
noise_directory = "/data/ephraim/datasets/known_noise/undiff_exps/exp_n_much/noises"
multiplication_factors = [0.25, 0.5, 1, 2, 4, 8]

process_signals(speech_directory, noise_directory, multiplication_factors)


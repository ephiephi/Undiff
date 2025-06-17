import torchaudio
import numpy as np

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch
import numpy as np
from scipy import signal
import pickle
from glob import glob
import os
import shutil
import torchaudio.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np
import librosa
import torchaudio
import pandas as pd
import soundfile as sf
import torchaudio.functional as F
import torch
import os
import torchaudio
import os
from audio_tools2  import *
from run_exp_m_much_data import run_exp
import yaml
import subprocess
import argparse
import pathlib
import pandas as pd
import os
import random
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt

from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
import torch.utils.data
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchaudio

from torch.autograd import Variable
import torch
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy import signal
import pickle
from torch import nn
import torch.optim as optim

import pickle

import os
import pandas as pd
from torch.utils.data import Dataset

from analyze.analyze_exp import analyze_exp, choose_closest_to_median
from run_storm import run_storm
from create_exp_m import NetworkNoise2, NetworkNoise3,NetworkNoise4, train_nets_parralel, get_named_beta_schedule


        
import logging

def setup_logging(exp_root):
    """Set up logging to save logs in the same directory as the model."""
    # model_dir = os.path.dirname(model_path)
    analysis_root = os.path.join(exp_root, "analysis")
    
    os.makedirs(analysis_root, exist_ok=True)  # Ensure the directory exists

    log_file = os.path.join(analysis_root, "training_log.txt")

    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),  # Save logs to file
            logging.StreamHandler()  # Print logs to console
        ]
    )

    return log_file



class NoiseDataset(Dataset):
    def __init__(self, data_tensor, gt_tensor):
        self.data_tensor =data_tensor
        self.gt_tensor = gt_tensor

    def __len__(self):
        return self.data_tensor.shape[0]

    def __getitem__(self, idx):
        item = self.data_tensor[idx,:,:]
        cur_gt = self.gt_tensor[idx]
        return item, cur_gt


import os
import shutil

import os
from pathlib import Path
import shutil



import os
from pathlib import Path
import re
import pandas as pd
import torchaudio



def create_directories(output_dir, file_id):
    """Create necessary subdirectories for a given file ID."""
    file_id_dir = output_dir / file_id
    noisy_wav_dir = file_id_dir / "noisy_wav"
    noises_dir_out = file_id_dir / "noises"
    clean_wav_dir = file_id_dir / "clean_wav"
    train_noise_dir = file_id_dir / "train_noise_wav"

    noisy_wav_dir.mkdir(parents=True, exist_ok=True)
    noises_dir_out.mkdir(parents=True, exist_ok=True)
    clean_wav_dir.mkdir(parents=True, exist_ok=True)
    train_noise_dir.mkdir(parents=True, exist_ok=True)

    return noisy_wav_dir, noises_dir_out, clean_wav_dir, train_noise_dir


def extract_metadata(noisy_file):
    """Extract file ID and SNR value from a noisy file name."""
    suffix = noisy_file.split("fileid_")[-1]
    file_id = suffix.split(".wav")[0]

    snr_match = re.search(r"snr(-?\d+)", noisy_file)
    snr_value = int(snr_match.group(1)) if snr_match else None

    return file_id, snr_value


def trim_and_save(audio, sr, start_sample, end_sample, target_path, save_audio):
    """Trim audio and save it to the target path."""
    trimmed_audio = audio[:, start_sample:end_sample]
    if save_audio:
        torchaudio.save(target_path, trimmed_audio, sr, encoding="PCM_F")


def process_wav_file(wav_path, start_seconds, end_seconds, sample_rate=16000):
    """Load, trim, and calculate sample indices for a WAV file."""
    audio, sr = torchaudio.load(wav_path)
    if sr != sample_rate:
        audio = F.resample(audio, sr, sample_rate)
    start_sample = int(start_seconds * sample_rate)
    if end_seconds == None:
        end_sample = audio.shape[1]
    else:
        end_sample = int(end_seconds * sample_rate)
    return audio, sample_rate, start_sample, end_sample


def add_metadata(df, idx, snr, noise_scaling, file_id, noisy_path, noise_path, clean_path, train_noise_path):
    """Add metadata to the DataFrame."""
    df.at[idx, "snr"] = snr
    df.at[idx, "noise_scaling"] = noise_scaling
    df.at[idx, "dir"] = file_id
    df.at[idx, "noise_idx"] = file_id
    df.at[idx, "noisy_wav"] = str(noisy_path)
    df.at[idx, "noise_path"] = str(noise_path) if noise_path else None
    df.at[idx, "clean_wav"] = str(clean_path) if clean_path else None
    df.at[idx, "train_noise_path"] = str(train_noise_path) if train_noise_path else None


def organize_wav_files(exp_root, output_pickle, num_train_seconds=0, num_test_seconds=10,save_audio=True):
    """
    Organize files into directories based on file IDs (X), copy and trim WAV files,
    and save metadata to a DataFrame.
    """
    # Directories
    noisy_dir = Path(exp_root) / "noisy_wav"
    noises_dir = Path(exp_root) / "noises"
    clean_dir = Path(exp_root) / "clean_wav" #todo: replace with "cleans"
    output_dir = Path(exp_root)

    # Lists to store metadata
    created_directories = []
    noise_ids = []
    snr_values = []
    df_snr = pd.DataFrame()

    # Process each noisy file
    for idx, noisy_file in enumerate(os.listdir(noisy_dir)):
        if not noisy_file.endswith(".wav"):
            continue

        # Extract metadata
        file_id, snr_value = extract_metadata(noisy_file)
        noisy_name_with_prefix = f"noise{file_id}_{noisy_file}"
        if file_id not in noise_ids:
            noise_ids.append(file_id)
        if snr_value not in snr_values:
            snr_values.append(snr_value)

        # Create directories
        noisy_wav_dir, noises_dir_out, clean_wav_dir, train_noise_dir = create_directories(output_dir, file_id)
        if file_id not in created_directories:
            created_directories.append(file_id)

        # File paths
        noisy_path = noisy_dir / noisy_file
        noise_file = next((f for f in os.listdir(noises_dir) if f.endswith(f"fileid_{file_id}.wav")), None)
        clean_file = next((f for f in os.listdir(clean_dir) if f.endswith(f"fileid_{file_id}.wav")), None)

        # Process noisy WAV
        noisy_audio, sr, start_sample, end_sample = process_wav_file(noisy_path, num_train_seconds, num_test_seconds)
        target_noisy_wav = noisy_wav_dir / noisy_name_with_prefix
        trim_and_save(noisy_audio, sr, start_sample, end_sample, target_noisy_wav,save_audio)

        # Process noise WAV
        if noise_file:
            noise_audio, _, train_start, train_end = process_wav_file(noises_dir / noise_file, 0, num_train_seconds)
            _, _, noise_start, noise_end = process_wav_file(noises_dir / noise_file, num_train_seconds, num_test_seconds)

            target_noise_path = noises_dir_out / noisy_name_with_prefix
            trim_and_save(noise_audio, sr, noise_start, noise_end, target_noise_path,save_audio)

            # Save the train noise WAV file
            target_train_noise_path = train_noise_dir / noisy_name_with_prefix
            trim_and_save(noise_audio, sr, train_start, train_end, target_train_noise_path,save_audio)
        else:
            target_noise_path, target_train_noise_path = None, None

        # Process clean WAV
        if clean_file:
            clean_audio, _, _, _ = process_wav_file(clean_dir / clean_file, num_train_seconds, num_test_seconds)
            target_clean_wav = clean_wav_dir / noisy_name_with_prefix
            trim_and_save(clean_audio, sr, start_sample, end_sample, target_clean_wav,save_audio)
        else:
            target_clean_wav = None

        # Add metadata
        add_metadata(
            df_snr, idx, snr_value, float(1), file_id, target_noisy_wav,
            target_noise_path, target_clean_wav, target_train_noise_path
        )

        logging.info(f"Processed file_id_{file_id}: Noisy, Noise, Clean, and Train Noise WAV files organized.")

    # Save DataFrame to pickle
    df_snr.to_pickle(output_pickle)
    logging.info(f"Metadata saved to {output_pickle}")

    return created_directories, noise_ids, snr_values








def reshape_signal(signal, batch_size):
    """
    Reshapes the original signal tensor into batches of the given batch size,
    removing the tail if necessary to make the length divisible by the batch size.
    
    Args:
        signal (torch.Tensor): The original signal tensor, assumed to be 1-dimensional.
        batch_size (int): The desired batch size.
    
    Returns:
        torch.Tensor: Reshaped tensor with shape (batch_size, -1).
    """
    # Flatten the signal in case it's multi-dimensional
    signal = signal.flatten()
    
    # Calculate the maximum length that fits evenly into the batch size
    total_length = signal.size(0)
    length_to_use = (total_length // batch_size) * batch_size
    
    # Truncate the signal to the required length
    truncated_signal = signal[:length_to_use]
    
    # Reshape the signal into (batch_size, -1)
    reshaped_signal = truncated_signal.view(batch_size, -1)
    return reshaped_signal



def train_noisemodel(root, network="NetworkNoise2",epochs=1800, dataset_size=1, n_samples=4,batch_size=2,g_t=None, num_nets=200,min_epochs=400,slope_epochs=2,test_len=1,mog=0,lr=None,one_network=False,scheduler=None):
    logging.info("starting training")
    logging.info(f"{root}")
    
    with open(Path(root)/'5f_snrs.pickle', 'rb') as handle:
        snr_df = pickle.load(handle)
    logging.info(snr_df)
    failed = []
    for i in tqdm(snr_df.index):
        for trial in [0]:
            train_idx = i
            cur_snr = snr_df["snr"][train_idx]
            noise_idx = snr_df["noise_idx"][train_idx]
            logging.info(f"noise_idx {noise_idx}")

            cur_dir = Path(root)/ snr_df["dir"][train_idx]
            logging.info(f"______{cur_dir}_______")

            cur_snr = snr_df["snr"][train_idx]
            cur_noise_scaling = snr_df["noise_scaling"][train_idx]
            noise_index = snr_df["noise_idx"][train_idx]
            if "X" in noise_idx:
                noise_index = noise_index.split("X")[0]
            logging.info(f"{cur_snr} {cur_dir} {noise_index} {cur_noise_scaling}")

            noise_path = snr_df["noise_path"][train_idx]

            train_noise_path = snr_df["train_noise_path"][train_idx]

            noise, sr = torchaudio.load(train_noise_path)    
            noise_train = noise[:, int(test_len*sr):]
            noise_test = noise[:,  :int(test_len*sr)]
            logging.info(f"noise_test.shape: {noise_test.shape}")
            
            train_ar = reshape_signal(noise_train, batch_size=n_samples)
            test_ar = reshape_signal(noise_test, batch_size=1)
            
            # sr = 16000
            train_tensor = torch.tensor(train_ar, dtype=torch.float32)#.view(1,1,-1)
            test_tensor = torch.tensor(test_ar, dtype=torch.float32)#.view(1,1,-1)
            
            
            logging.info("creating tensors")

            train_full_tensors = train_tensor.squeeze().view(train_tensor.shape[0],1,-1)
            test_full_tensors = test_tensor.squeeze().view(test_tensor.shape[0],1,-1)


            
            logging.info("starting training")
            nets,loss_array,loss_test_array = train_nets_parralel(network,train_full_tensors, test_full_tensors,trial,epochs=epochs,num_nets=num_nets,batch_size=batch_size,g_t=g_t,exp_root=exp_root,min_epochs=min_epochs,slope_epochs=slope_epochs,mog=mog,lr=lr,one_network=one_network,scheduler=scheduler)
            logging.info("end 1 training")
            
            # params_dict = {"nets": nets, "train_dataset": None, "test_dataset": None,"ar_coefs":None, "loss_array":loss_array, "loss_test_array": loss_test_array, "ar_noise": None, "noise_scaling": cur_noise_scaling, "snr": str(int(cur_snr)), "noise_name": noise_idx, "noise_path": noise_path}
            params_dict = {"nets": [net.state_dict() for net in nets],"network":network, "train_dataset": None, "test_dataset": None,"ar_coefs":None, "loss_array":loss_array, "loss_test_array": loss_test_array, "ar_noise": None, "noise_scaling": cur_noise_scaling, "snr": str(int(cur_snr)), "noise_name": noise_idx, "noise_path": noise_path}
            
            # params_dict = {"result": result}
            params_dict_debug = {"nets": [net.state_dict() for net in nets],"network":network, "train_dataset": train_full_tensors, "test_dataset": test_full_tensors,"ar_coefs":None, "loss_array":loss_array, "loss_test_array": loss_test_array, "ar_noise": None, "noise_scaling": cur_noise_scaling, "snr": str(int(cur_snr)), "noise_name": noise_idx, "noise_path": noise_path}
            
            #save in name of 0
            pickle_path = cur_dir/(str(0)+"_"+"snr"+str(int(cur_snr))+"_"+str(noise_index)+"_models.pickle")

            tmp_pickle_path = cur_dir/(str(0)+"_"+"tmp_"+"snr"+str(int(cur_snr))+"_"+str(noise_index)+"_models.pickle")
            
    
            try:
                del train_full_tensors
            except:
                logging.info(" del train_full_tensors failed")
            torch.cuda.empty_cache()
            logging.info("torch.cuda.empty_cache()")

            try:
                with open(pickle_path, 'wb') as handle:
                    pickle.dump(params_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            except:
                logging.info(f"failed in {i}")
                failed.append(i)
            try:
                with open(tmp_pickle_path, 'wb') as handle:
                    pickle.dump(params_dict_debug, handle, protocol=pickle.HIGHEST_PROTOCOL)
            except:
                logging.info(f"failed in {i}")
                failed.append(i)
                continue
    logging.info(f"failed: {failed}")
    logging.info(pickle_path)




def calculate_scaling_factor(clean_audio, noise_audio, target_snr):
    """Calculate the scaling factor to adjust noise to the target SNR level."""
    target_snr = float(target_snr)
    clean_power = torch.mean(clean_audio**2)
    noise_power = torch.mean(noise_audio**2)
    desired_noise_power = clean_power / (10 ** (target_snr / 10))
    scaling_factor = torch.sqrt(desired_noise_power / noise_power)
    return scaling_factor






def organize_wav_files_snr_levels_old(exp_root, output_pickle, snr_levels, num_train_seconds=0, num_test_seconds=10):
    """
    Organize files into directories based on file IDs (X), calculate and scale noise for multiple SNR levels,
    save the results, and save metadata to a DataFrame.
    """
    # Directories
    noisy_dir = Path(exp_root) / "noisy_wav"
    noises_dir = Path(exp_root) / "noises"
    clean_dir = Path(exp_root) / "cleans"
    output_dir = Path(exp_root)

    # Check if necessary directories exist
    if not noisy_dir.exists():
        raise FileNotFoundError(f"'noisy_wav' directory not found in {exp_root}")
    if not noises_dir.exists():
        raise FileNotFoundError(f"'noises' directory not found in {exp_root}")
    if not clean_dir.exists():
        clean_dir = Path(exp_root) / "clean_wav"
        if not clean_dir.exists():
            logging.info(f" 'clean_wav' directory not found in {exp_root}. Skipping processing.")
            raise FileNotFoundError(f"'noises' directory not found in {exp_root}")

    # Metadata storage
    df_snr = pd.DataFrame()
    created_directories = []
    noise_ids = []
    snr_values = []
    df_idx=-1

    # Process each noisy file
    for idx, noise_file in enumerate(os.listdir(noises_dir)):
        if not noise_file.endswith(".wav") or "original" in noise_file:
            continue

        # Extract metadata
        suffix = noise_file.split("fileid_")[-1]
        if "X" in suffix:
            file_idx = suffix.split("X")[0]
        else:
            file_idx = suffix.split(".wav")[0]
        file_id = suffix.split(".wav")[0]
        noisy_file = next((f for f in os.listdir(noisy_dir) if f.endswith(f"fileid_{file_idx}.wav")), None)
        
        # Create directories
        noisy_wav_dir, noises_dir_out, clean_wav_dir, train_noise_dir = create_directories(output_dir, file_id)
        if file_id not in created_directories:
            created_directories.append(file_id)

        # File paths
        noisy_path = noisy_dir / noisy_file
        noise_file = next((f for f in os.listdir(noises_dir) if f.endswith(f"fileid_{file_id}.wav")), None)
        clean_file = next((f for f in os.listdir(clean_dir) if f.endswith(f"fileid_{file_id}.wav")), None)

        if not noise_file or not clean_file:
            logging.info(f"Warning: Missing 'noise' or 'clean' file for file_id {file_id}. Skipping.")
            continue

        # Load clean and noise WAVs
        clean_audio, sr, clean_start, clean_end = process_wav_file(clean_dir / clean_file, num_train_seconds, num_test_seconds)
        noise_audio, _, noise_start, noise_end = process_wav_file(noises_dir / noise_file, num_train_seconds, num_test_seconds)

        # Process training noise
        train_noise_audio, _, train_start, train_end = process_wav_file(noises_dir / noise_file, 0, num_train_seconds)

        for snr in snr_levels:
            # Scale noise to target SNR
            scaling_factor = calculate_scaling_factor(clean_audio[:, clean_start:clean_end], noise_audio[:, noise_start:noise_end], snr)
            scaled_noise_audio = noise_audio[:, noise_start:noise_end] * scaling_factor
            
            original_snr=noisy_file.split("snr")[-1].split("_")[0]
            new_noisy_name = noisy_file.replace("snr"+original_snr,"snr"+snr)

            # Save scaled noise
            scaled_noise_name = f"noise{file_idx}_{new_noisy_name}"
            target_noise_path = noises_dir_out / scaled_noise_name
            torchaudio.save(target_noise_path, scaled_noise_audio, sr, encoding="PCM_F")

            # Add scaled noise to clean audio
            noisy_audio = clean_audio[:, clean_start:clean_end] + scaled_noise_audio
            target_noisy_path = noisy_wav_dir / scaled_noise_name
            torchaudio.save(target_noisy_path, noisy_audio, sr, encoding="PCM_F")

            # Scale and save training noise
            scaled_train_noise_audio = train_noise_audio[:,train_start:train_end] * scaling_factor
            train_noise_name = f"noise{file_idx}_{new_noisy_name}"
            target_train_noise_path = train_noise_dir / train_noise_name
            torchaudio.save(target_train_noise_path, scaled_train_noise_audio, sr, encoding="PCM_F")
            
            # Save the clean WAV
            clean_save_path = clean_wav_dir / f"noise{file_idx}_{new_noisy_name}"
            torchaudio.save(clean_save_path, clean_audio[:, clean_start:clean_end], sr, encoding="PCM_F")

            # Add metadata
            df_idx += 1 
            add_metadata(
                df_snr, df_idx, snr, float(scaling_factor.item()), file_id, target_noisy_path,
                target_noise_path, clean_save_path, target_train_noise_path
            )

            logging.info(f"Processed file_id_{file_id} for SNR={snr}: Scaled Noise, Noisy WAV, Clean WAV, and Train Noise saved.")

            if file_idx not in noise_ids:
                noise_ids.append(file_idx)
            if snr not in snr_values    :
                snr_values.append(snr)

    # Save DataFrame to pickle
    df_snr.to_pickle(output_pickle)
    logging.info(f"Metadata saved to {output_pickle}")

    return created_directories, noise_ids, snr_values


def organize_wav_files_snr_levels(exp_root, output_pickle, snr_levels, num_train_seconds=0, num_test_seconds=10, num_train_seconds_noise=0, num_test_seconds_noise=10,train_on_test=False):
    """
    Organize files into directories based on file IDs (X), calculate and scale noise for multiple SNR levels,
    save the results, and save metadata to a DataFrame.
    """
    # Directories
    noisy_dir = Path(exp_root) / "noisy_wav"
    noises_dir = Path(exp_root) / "noises"
    clean_dir = Path(exp_root) / "cleans"
    output_dir = Path(exp_root)

    # Check if necessary directories exist
    if not noisy_dir.exists():
        raise FileNotFoundError(f"'noisy_wav' directory not found in {exp_root}")
    if not noises_dir.exists():
        raise FileNotFoundError(f"'noises' directory not found in {exp_root}")
    if not clean_dir.exists():
        clean_dir = Path(exp_root) / "clean_wav"
        if not clean_dir.exists():
            logging.info(f" 'clean_wav' directory not found in {exp_root}. Skipping processing.")
            raise FileNotFoundError(f"'noises' directory not found in {exp_root}")

    # Metadata storage
    df_snr = pd.DataFrame()
    created_directories = []
    noise_ids = []
    snr_values = []
    df_idx = -1

    # Process each noisy file
    for idx, noise_file in enumerate(os.listdir(noises_dir)):
        if not noise_file.endswith(".wav") or "original" in noise_file:
            continue

        # Extract metadata
        suffix = noise_file.split("fileid_")[-1]
        file_idx = suffix.split("X")[0] if "X" in suffix else suffix.split(".wav")[0]
        file_id = suffix.split(".wav")[0]
        noisy_file = next((f for f in os.listdir(noisy_dir) if f.endswith(f"fileid_{file_idx}.wav")), None)
        
        # Create directories
        noisy_wav_dir, noises_dir_out, clean_wav_dir, train_noise_dir = create_directories(output_dir, file_id)
        if file_id not in created_directories:
            created_directories.append(file_id)

        # File paths
        noisy_path = noisy_dir / noisy_file
        noise_file = next((f for f in os.listdir(noises_dir) if f.endswith(f"fileid_{file_id}.wav")), None)
        clean_file = next((f for f in os.listdir(clean_dir) if f.endswith(f"fileid_{file_id}.wav")), None)

        if not noise_file or not clean_file:
            logging.info(f"Warning: Missing 'noise' or 'clean' file for file_id {file_id}. Skipping.")
            continue

        # Load clean and noise WAVs
        clean_audio, sr, clean_start, clean_end = process_wav_file(clean_dir / clean_file, num_train_seconds, num_test_seconds)
        noise_audio, _, noise_start, noise_end = process_wav_file(noises_dir / noise_file, num_train_seconds_noise, num_test_seconds_noise)

        # Process training noise
        train_noise_audio, _, train_start, train_end = process_wav_file(noises_dir / noise_file, 0, num_train_seconds_noise)

        for snr in snr_levels:
            # Scale noise to target SNR
            scaling_factor = calculate_scaling_factor(clean_audio[:, clean_start:clean_end], noise_audio[:, noise_start:noise_end], snr)
            scaled_noise_audio = noise_audio[:, noise_start:noise_end] * scaling_factor
            
            original_snr = noisy_file.split("snr")[-1].split("_")[0]
            new_noisy_name = noisy_file.replace("snr" + original_snr, "snr" + snr)

            # Save scaled noise
            scaled_noise_name = f"noise{file_idx}_{new_noisy_name}"
            target_noise_path = noises_dir_out / scaled_noise_name
            torchaudio.save(target_noise_path, scaled_noise_audio, sr, encoding="PCM_F")

            # Add scaled noise to clean audio
            noisy_audio = clean_audio[:, clean_start:clean_end] + scaled_noise_audio
            target_noisy_path = noisy_wav_dir / scaled_noise_name
            torchaudio.save(target_noisy_path, noisy_audio, sr, encoding="PCM_F")

            # Scale and save training noise
            scaled_train_noise_audio = train_noise_audio[:, train_start:train_end] * scaling_factor
            if train_on_test:
                scaled_train_noise_audio = scaled_noise_audio
            train_noise_name = f"noise{file_idx}_{new_noisy_name}"
            target_train_noise_path = train_noise_dir / train_noise_name
            torchaudio.save(target_train_noise_path, scaled_train_noise_audio, sr, encoding="PCM_F")
            
            # Save the clean WAV
            clean_save_path = clean_wav_dir / f"noise{file_idx}_{new_noisy_name}"
            torchaudio.save(clean_save_path, clean_audio[:, clean_start:clean_end], sr, encoding="PCM_F")

            # Add metadata
            df_idx += 1 
            add_metadata(
                df_snr, df_idx, snr, float(scaling_factor.item()), file_id, target_noisy_path,
                target_noise_path, clean_save_path, target_train_noise_path
            )

            logging.info(f"Processed file_id_{file_id} for SNR={snr}: Scaled Noise, Noisy WAV, Clean WAV, and Train Noise saved.")

            if file_idx not in noise_ids:
                noise_ids.append(file_idx)
            if snr not in snr_values:
                snr_values.append(snr)

    # Save DataFrame to pickle
    df_snr.to_pickle(output_pickle)
    logging.info(f"Metadata saved to {output_pickle}")

    return created_directories, noise_ids, snr_values



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="measure guided")
    parser.add_argument(
        "-config",
        default="exps_configs/librAR_net3_6_snrs.yaml",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        trials = yaml.safe_load(f)
    s_array = trials.get("s_array", [])
    snr_array = trials.get("snr_array", [])
    exp_root = trials.get("exp_root", "")
    network = trials["network"] 
    epochs = trials.get("epochs", 1500) 
    min_epochs=trials.get("min_epochs", 1500) 
    slope_epochs=trials.get("slope_epochs", 2) 
    batch_size = trials.get("batch_size", 0) 
    dataset_size = eval(trials.get("dataset_size", "0"))
    n_samples = trials["n_samples"] 
    scheduler_type = trials.get("scheduler_type","linear")
    num_steps = trials.get("num_steps",200)
    scheduler = trials.get("scheduler","60")
    test_start_sec = trials.get("test_start_sec",6)
    test_end_sec = trials.get("test_end_sec",10)
    test_start_sec_noise = trials.get("test_start_sec_noise",test_start_sec)
    test_end_sec_noise = trials.get("test_end_sec_noise",test_end_sec)
    trained_model_path = trials.get("trained_model_path","0")
    test_len = trials.get("val_len","1")
    mog = trials.get("mog", 0) 
    lr = trials.get("lr", 0.001 )
    one_network = trials.get("one_network", False) 
    training_scheduler = trials.get("training_scheduler", None) 
    train_on_test = trials.get("train_on_test", False)
    loss_model = trials.get("loss_model", "loss_model")
    s_scheduler = trials.get("s_inference_scheduler", "enhanced_60")
    
    print("loss_model: ",loss_model )
    
    log_file = setup_logging(exp_root)
    logging.info(f"\nDir: {args.config}\n")
    logging.info(f"Logging to: {log_file}")

    
    betas=get_named_beta_schedule(scheduler_type, num_steps)
    alphas = 1.0 - betas
    alphas_cumprod =  torch.from_numpy(np.cumprod(alphas, axis=0))
    g_t = torch.sqrt((1-alphas_cumprod)/(alphas_cumprod))
    

    output_pickle_path = Path(exp_root)/"5f_snrs.pickle"

    
    save_audio = True
    if len(snr_array)==0:
        names, noises_names,snr_array = organize_wav_files(exp_root, output_pickle_path, num_train_seconds=test_start_sec, num_test_seconds=test_end_sec, save_audio=save_audio)
    else:
        names, noises_names,snr_array = organize_wav_files_snr_levels(exp_root, output_pickle_path, snr_array, num_train_seconds=test_start_sec, num_test_seconds=test_end_sec,  num_train_seconds_noise=test_start_sec_noise, num_test_seconds_noise=test_end_sec_noise,train_on_test=train_on_test)
    
    run_network=None 
    if trained_model_path == "0":
        logging.info("---startin training---")
        train_noisemodel(exp_root, network=network,epochs=epochs, dataset_size=dataset_size, n_samples=n_samples,batch_size=batch_size,g_t=g_t,num_nets=num_steps,min_epochs=min_epochs,slope_epochs=slope_epochs,test_len=test_len,mog=mog,lr=lr,one_network=one_network,scheduler=training_scheduler)
    else:
        run_network=  network
    # names=["6","18"]
    # noises_names = ["6","18"]
    
    logging.info("---run_exp---")
    run_exp(exp_root, dirnames=names,s_array=s_array, reset=False, s_schedule=scheduler, scheduler_type=scheduler_type,noise_mosel_path=trained_model_path, network=run_network,mog=mog,loss_model=loss_model,outdirname=s_scheduler)
    
    storm_root = str(Path(exp_root)/"storm")
    run_storm(exp_root,storm_root) #200 steps
    
    logging.info("---analyzing---")
    # analyze_exp(exp_root,noises_names,snr_array,names,specific_s=None)
    
    analyze_exp(exp_root,noises_names,snr_array,names,specific_s=None)
    
    ours_results_path = Path(exp_root)/"analysis"/"ours_all.xlsx"
    winner_s = choose_closest_to_median(ours_results_path)
    analyze_exp(exp_root,noises_names,snr_array,names,specific_s=winner_s,output_namedir="analysis_specific_s",)



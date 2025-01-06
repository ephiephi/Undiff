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
import seaborn as sns
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

from analyze.analyze_exp import analyze_exp
from run_storm import run_storm
from create_exp_m import NetworkNoise2, NetworkNoise3,NetworkNoise4, train_nets_parralel, get_named_beta_schedule


        
# def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
#     """
#     Get a pre-defined beta schedule for the given name.
#     The beta schedule library consists of beta schedules which remain similar
#     in the limit of num_diffusion_timesteps.
#     Beta schedules may be added, but should not be removed or changed once
#     they are committed to maintain backwards compatibility.
#     """
#     if schedule_name == "linear": ###chosen in default
#         # Linear schedule from Ho et al, extended to work for any number of
#         # diffusion steps.
#         # scale = 1000 / num_diffusion_timesteps
#         beta_start = 0.0001  # scale * 0.0001
#         beta_end = 0.02  # scale * 0.02
#         return np.linspace(
#             beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
#         )
#     elif schedule_name == "cosine":
#         return betas_for_alpha_bar(
#             num_diffusion_timesteps,
#             lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
#         )
#     else:
#         raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


# def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
#     """
#     Create a beta schedule that discretizes the given alpha_t_bar function,
#     which defines the cumulative product of (1-beta) over time from t = [0,1].
#     :param num_diffusion_timesteps: the number of betas to produce.
#     :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
#                       produces the cumulative product of (1-beta) up to that
#                       part of the diffusion process.
#     :param max_beta: the maximum beta to use; use values lower than 1 to
#                      prevent singularities.
#     """
#     betas = []
#     for i in range(num_diffusion_timesteps):
#         t1 = i / num_diffusion_timesteps
#         t2 = (i + 1) / num_diffusion_timesteps
#         betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
#     return np.array(betas)
        
# betas=get_named_beta_schedule("linear", 200)

# alphas = 1.0 - betas
# alphas_cumprod =  torch.from_numpy(np.cumprod(alphas, axis=0))
# g_t = torch.sqrt((1-alphas_cumprod)/(alphas_cumprod))



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


def process_wav_file(wav_path, start_seconds, end_seconds):
    """Load, trim, and calculate sample indices for a WAV file."""
    audio, sr = torchaudio.load(wav_path)
    start_sample = int(start_seconds * sr)
    end_sample = int(end_seconds * sr)
    return audio, sr, start_sample, end_sample


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

        print(f"Processed file_id_{file_id}: Noisy, Noise, Clean, and Train Noise WAV files organized.")

    # Save DataFrame to pickle
    df_snr.to_pickle(output_pickle)
    print(f"Metadata saved to {output_pickle}")

    return created_directories, noise_ids, snr_values




    


# import torch.multiprocessing as mp

# def train_nets_process(network, train_full_tensors, test_full_tensors, device, idxes,trial, epochs=6000,batch_size=16):
#     if network=="NetworkNoise2":
#         nets = [NetworkNoise2() for i in range(len(idxes))]
#     elif network=="NetworkNoise3":
#         nets = [NetworkNoise3() for i in range(len(idxes))]
#     else:
#         print("network unknown")
#         raise Exception
#     # elif trial==1:
#     #     nets = [Network2() for i in range(len(idxes))]
#     # elif trial==2:
#     #     nets = [Network3() for i in range(len(idxes))]
#     # elif trial==3:
#     #     nets = [Network4() for i in range(len(idxes))]
    
#     print("idxes[0]:",idxes[0])
#     if idxes[0] == 0:
#         quarter_idx=0
#     elif idxes[0] == 50:
#         quarter_idx=1
#     elif idxes[0] == 100:
#         quarter_idx=2
#     elif idxes[0] == 150:
#         quarter_idx=3
#     else:
#         print ("no identifiesd quarter")
#         raise Exception
    
#     cur_epochs = epochs
    

#     loss_array = {}
#     loss_test_array = {}

#     net_counter=-1
#     for i in idxes:
        
#         cur_white_noise_diffusion = torch.normal(0,1,train_full_tensors[:,0,:].shape)
#         cur_train_full_tensors = train_full_tensors[:,0,:]+cur_white_noise_diffusion*g_t[i]
        
#         cur_white_noise_diffusion = torch.normal(0,1,test_full_tensors[:,0,:].shape)
#         cur_test_full_tensors = test_full_tensors[:,0,:]+cur_white_noise_diffusion*g_t[i]
        
#         #Create TensorDatasets
#         dataset_size_ = train_full_tensors.shape[0]
#         train_dataset = BatchNoiseDataset(cur_train_full_tensors.reshape(dataset_size_,1,-1),g_t[i])
#         test_dataset = BatchNoiseDataset(cur_test_full_tensors.reshape(dataset_size_,1,-1),g_t[i])

#         #Create DataLoaders
#         train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) #todo: numbers
#         test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        
#         net_counter+=1
#         model = nets[net_counter]
#         model.to(device)
#         model.train()

#         optimizer = optim.Adam(model.parameters())
        
#         cur_epochs = epochs

#         for epoch in range(cur_epochs):
#             running_loss = 0.0
#             for batch_idx, (batch_tensor, gt_tensor) in enumerate(train_loader):
#                 optimizer.zero_grad()
#                 batch_tensor = batch_tensor.to(device, dtype=torch.float)
#                 gt_tensor = gt_tensor.to(device, dtype=torch.float)
                
#                 means, stds = model(batch_tensor, gt_tensor)
#                 loss = model.casual_loss( means, stds, wav_tensor=batch_tensor).mean()
#                 loss.backward()
#                 optimizer.step()
#                 running_loss += loss.item()
                
#             if epoch%1==0:
#                 with torch.no_grad():
#                     for batch_idx, (test_inputs, gt_test) in enumerate(test_loader):
#                         test_inputs = test_inputs.to(device, dtype=torch.float)
#                         gt_test = gt_test.to(device, dtype=torch.float)
#                         meanst, stdst = model(test_inputs, gt_test)
#                 loss_t = model.casual_loss( meanst, stdst, wav_tensor=test_inputs).mean()
#                 if i in loss_test_array:
#                     loss_test_array[i].append(float(loss_t))
#                 else:
#                     loss_test_array[i] = [float(loss_t)]
            
#             if i in loss_array:
#                 loss_array[i].append(float(loss))
#             else:
#                 loss_array[i] = [float(loss)]
#         nets[net_counter].parameters = model.parameters
#         print(f"Model {i} Epoch {epoch+1}/{epochs}, Loss: {running_loss}")
    
#     return nets, loss_array, loss_test_array, quarter_idx




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



def train_noisemodel(root, network="NetworkNoise2",epochs=1800, dataset_size=1, n_samples=4,batch_size=2,g_t=None, num_nets=200):
    print("starting training")
    print(root)
    
    with open(Path(root)/'5f_snrs.pickle', 'rb') as handle:
        snr_df = pickle.load(handle)
    print(snr_df)
    failed = []
    for i in tqdm(snr_df.index):
        for trial in [0]:
            train_idx = i
            cur_snr = snr_df["snr"][train_idx]
            noise_idx = snr_df["noise_idx"][train_idx]
            print("noise_idx", noise_idx)

            cur_dir = Path(root)/ snr_df["dir"][train_idx]
            print(f"______{cur_dir}_______")

            cur_snr = snr_df["snr"][train_idx]
            cur_noise_scaling = snr_df["noise_scaling"][train_idx]
            noise_index = snr_df["noise_idx"][train_idx]
            if "X" in noise_idx:
                noise_index = noise_index.split("X")[0]
            print(cur_snr,cur_dir, noise_index, cur_noise_scaling)

            noise_path = snr_df["noise_path"][train_idx]

            train_noise_path = snr_df["train_noise_path"][train_idx]

            noise, sr = torchaudio.load(train_noise_path)    
            noise_train = noise[:, int(1*sr):]
            noise_test = noise[:,  :int(1*sr)]
            print("noise_test.shape:", noise_test.shape)
            
            train_ar = reshape_signal(noise_train, batch_size=n_samples)
            test_ar = reshape_signal(noise_test, batch_size=1)
            
            # sr = 16000
            train_tensor = torch.tensor(train_ar, dtype=torch.float32)#.view(1,1,-1)
            test_tensor = torch.tensor(test_ar, dtype=torch.float32)#.view(1,1,-1)
            
            
            print("creating tensors")

            train_full_tensors = train_tensor.squeeze().view(dataset_size,1,-1)
            test_full_tensors = test_tensor.reshape(dataset_size,200,-1)


            
            print("starting training")
            nets,loss_array,loss_test_array = train_nets_parralel(network,train_full_tensors, test_full_tensors,trial,epochs=epochs,num_nets=num_nets,batch_size=batch_size,g_t=g_t,exp_root=exp_root)
            print("end 1 training")
            
            params_dict = {"nets": nets, "train_dataset": None, "test_dataset": None,"ar_coefs":None, "loss_array":loss_array, "loss_test_array": loss_test_array, "ar_noise": None, "noise_scaling": cur_noise_scaling, "snr": str(int(cur_snr)), "noise_name": noise_idx, "noise_path": noise_path}
            # params_dict = {"result": result}
            params_dict_debug = {"nets": nets, "train_dataset": train_full_tensors, "test_dataset": test_full_tensors,"ar_coefs":None, "loss_array":loss_array, "loss_test_array": loss_test_array, "ar_noise": None, "noise_scaling": cur_noise_scaling, "snr": str(int(cur_snr)), "noise_name": noise_idx, "noise_path": noise_path}
            
            #save in name of 0
            pickle_path = cur_dir/(str(0)+"_"+"snr"+str(int(cur_snr))+"_"+str(noise_index)+"_models.pickle")

            tmp_pickle_path = cur_dir/(str(0)+"_"+"tmp_"+"snr"+str(int(cur_snr))+"_"+str(noise_index)+"_models.pickle")
            
            try:
                del train_full_tensors
            except:
                print(" del train_full_tensors failed")
            torch.cuda.empty_cache()
            print("torch.cuda.empty_cache()")

            try:
                with open(pickle_path, 'wb') as handle:
                    pickle.dump(params_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            except:
                print(f"failed in {i}")
                failed.append(i)
            try:
                with open(tmp_pickle_path, 'wb') as handle:
                    pickle.dump(params_dict_debug, handle, protocol=pickle.HIGHEST_PROTOCOL)
            except:
                print(f"failed in {i}")
                failed.append(i)
                continue
    print("failed: ", failed)
    print(pickle_path)



# def noise_waves(exp_root, snr_array=[5], num_train_seconds=9, num_test_seconds=10):
#     noiseroot = os.path.join(exp_root,"noises/")
#     noise_files =  glob(noiseroot+"/*")
#     print(noise_files)

#     dirs_ =  glob(exp_root+"/*")

#     idx = 0
#     sample_rate=16000
#     dict_ = {"snr":[],"noise_scaling": [], "noise_idx":[], "dir": [] }
#     df_snr = pd.DataFrame(data=dict_)
#     names_ = []

#     for dir_ in dirs_:
#         if os.path.basename(dir_)=="noises" or os.path.basename(dir_) =='5f_snrs.pickle' or os.path.basename(dir_) =='cleans':
#             continue
#         print("dir_", dir_)
#         print(glob(dir_+"/clean_wav/*"))
#         speech_file_ = glob(dir_+"/clean_wav/*")[0]
        
#         speech, sr = torchaudio.load(speech_file_)

#         if sr != sample_rate:
#             speech = F.resample(speech, sr, sample_rate)

#         simple_power =  1 / speech.shape[1] * torch.sum(speech**2)
#         print("simple_power: ", simple_power)

#         for i, snr in enumerate(snr_array):
#             snr=int(snr)
            
#             for j,noise_f in enumerate(noise_files):
#                 idx +=1        
#                 noise_idx = os.path.basename(noise_f).replace(".wav","")
#                 noise, sr = torchaudio.load(noise_f)
#                 noise = noise[:,(5*sr):(speech.shape[1]+(5*sr))] ###leave 5 sec for training
#                 speech2 = speech[:,:noise.shape[1]]
                
#                 noise_power = 1 / noise.shape[1] * torch.sum(noise**2)
#                 # noise_power = clean_power_noise
#                 speech_power = simple_power
#                 noise_power_target = speech_power * np.power(10, -snr / 10)
#                 noise_scaling = np.sqrt(noise_power_target / noise_power)
#                 print("speech.shape: ",speech2.shape)
#                 print("noise.shape: ",noise.shape)

#                 lossy_speech = speech2 + noise_scaling * noise
#                 print("noise_scaling: ", noise_scaling)
                
#                 noise_var = noise_scaling**2
                
#                 new_noise = noise_scaling * noise
#                 new_noise_power = float(1 / new_noise.shape[1] * torch.sum(new_noise**2))
                                
#                 # print("clean_power: ", yclean_power)
#                 print("snr: ", snr)
#                 df_snr.at[idx, "snr"] = snr
#                 df_snr.at[idx, "noise_scaling"] = float(noise_scaling)
#                 df_snr.at[idx, "dir"] = os.path.basename(os.path.normpath(dir_))
#                 df_snr.at[idx, "noise_idx"] = noise_idx
                
#                 filename = "noise{}_digits_snr{}_power{}_var{}.wav".format(noise_idx,snr, new_noise_power, noise_var)

#                 tarpath = os.path.join(os.path.join(dir_, "clean_wav"), filename)
#                 print("tarpath: ", tarpath)
#                 torchaudio.save(tarpath, speech2, sample_rate,encoding="PCM_F")
#                 df_snr.at[idx, "clean_wav"] = tarpath

#                 noisy_root = os.path.join(dir_, "noisy_wav")
#                 tar_noisypath =  os.path.join(noisy_root, filename)
#                 if not os.path.exists(noisy_root):
#                     os.mkdir(noisy_root)
#                 torchaudio.save(tar_noisypath, lossy_speech, sample_rate,encoding="PCM_F")
#                 df_snr.at[idx, "noisy_wav"] = tar_noisypath
                
#                 noise, sr = torchaudio.load(noise_f)
#                 scaled_noise = noise_scaling * noise
#                 noises_root = os.path.join(dir_, "noises")
#                 print("noise_scaling: ", noise_scaling)
#                 tar_noise =  os.path.join(noises_root, filename)
#                 if not os.path.exists(noises_root):
#                     os.mkdir(noises_root)
#                 df_snr.at[idx, "noise_path"] = tar_noise

#                 torchaudio.save(tar_noise, scaled_noise, sample_rate,encoding="PCM_F")
        

#     df_snr = df_snr.sort_values(by=['dir'])
#     print(df_snr)

#     with open(os.path.join(exp_root,'5f_snrs.pickle'), 'wb') as handle:
#         pickle.dump(df_snr, handle, protocol=pickle.HIGHEST_PROTOCOL)
#     return  names_, noises_names_, snr_array


def calculate_scaling_factor(clean_audio, noise_audio, target_snr):
    """Calculate the scaling factor to adjust noise to the target SNR level."""
    target_snr = float(target_snr)
    clean_power = torch.mean(clean_audio**2)
    noise_power = torch.mean(noise_audio**2)
    desired_noise_power = clean_power / (10 ** (target_snr / 10))
    scaling_factor = torch.sqrt(desired_noise_power / noise_power)
    return scaling_factor






def organize_wav_files_snr_levels(exp_root, output_pickle, snr_levels, num_train_seconds=0, num_test_seconds=10):
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
            print(f" 'clean_wav' directory not found in {exp_root}. Skipping processing.")
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
            print(f"Warning: Missing 'noise' or 'clean' file for file_id {file_id}. Skipping.")
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

            print(f"Processed file_id_{file_id} for SNR={snr}: Scaled Noise, Noisy WAV, Clean WAV, and Train Noise saved.")

            if file_idx not in noise_ids:
                noise_ids.append(file_idx)
            if snr not in snr_values    :
                snr_values.append(snr)

    # Save DataFrame to pickle
    df_snr.to_pickle(output_pickle)
    print(f"Metadata saved to {output_pickle}")

    return created_directories, noise_ids, snr_values





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="measure guided")
    parser.add_argument(
        "-config",
        default="exps_configs/n_chosen.yaml",
    )
    args = parser.parse_args()
    print(f"\nDir: {args.config}\n")

    with open(args.config, "r") as f:
        trials = yaml.safe_load(f)
    s_array = trials.get("s_array", [])
    snr_array = trials.get("snr_array", [])
    exp_root = trials.get("exp_root", "")
    network = trials["network"] 
    epochs = trials.get("epochs", 0) 
    batch_size = trials.get("batch_size", 0) 
    dataset_size = eval(trials.get("dataset_size", "0"))
    n_samples = trials["n_samples"] 
    scheduler_type = trials.get("scheduler_type","linear")
    num_steps = trials.get("num_steps",200)
    scheduler = trials.get("scheduler","60")
    
    betas=get_named_beta_schedule(scheduler_type, num_steps)
    alphas = 1.0 - betas
    alphas_cumprod =  torch.from_numpy(np.cumprod(alphas, axis=0))
    g_t = torch.sqrt((1-alphas_cumprod)/(alphas_cumprod))
    

    output_pickle_path = Path(exp_root)/"5f_snrs.pickle"
    # test_start_sec=9
    # test_end_sec=10
    test_start_sec=6
    test_end_sec=10
    save_audio = True
    if len(snr_array)==0:
        names, noises_names,snr_array = organize_wav_files(exp_root, output_pickle_path, num_train_seconds=test_start_sec, num_test_seconds=test_end_sec, save_audio=save_audio)
    else:
        names, noises_names,snr_array = organize_wav_files_snr_levels(exp_root, output_pickle_path, snr_array, num_train_seconds=test_start_sec, num_test_seconds=test_end_sec)
    
    train_noisemodel(exp_root, network=network,epochs=epochs, dataset_size=dataset_size, n_samples=n_samples,batch_size=batch_size,g_t=g_t,num_nets=num_steps)
    
    
    print("---run_exp---")
    run_exp(exp_root, dirnames=names, cuda_idx="3",s_array=s_array, reset=False, s_schedule=scheduler, scheduler_type=scheduler_type)
    
    storm_root = str(Path(exp_root)/"storm")
    run_storm(exp_root,storm_root) #200 steps
    
    print("---analyzing---")
    analyze_exp(exp_root,noises_names,snr_array,names)


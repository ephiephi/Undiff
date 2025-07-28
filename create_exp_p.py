import torchaudio
import numpy as np

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch
import numpy as np
from scipy import signal
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


import os
import pandas as pd
from torch.utils.data import Dataset

from analyze.analyze_exp import analyze_exp, choose_closest_to_median
from run_storm import run_storm
from create_exp_m import NetworkNoise2, NetworkNoise3,NetworkNoise4, get_named_beta_schedule,get_group_indices, plot_loss, get_group_indices
from torch.optim.lr_scheduler import StepLR
import torch.multiprocessing as mp
from network_factory2 import *
from create_exp_n_real import process_wav_file
        
import logging



class BatchNoiseDataset(Dataset):
    """
    data_tensor  –  y  (noisy signal)     shape  (B, 1, L)
    clean_tensor –  x0 (clean signal)     shape  (B, 1, L)
    g_t          –  scalar conditioning   (ignored by this fix)
    """
    def __init__(self, noisy_tensor, clean_tensor):
        self.y  = noisy_tensor.float()
        self.x0 = clean_tensor.float()


    def __len__(self):
        return self.y.size(0)

    def __getitem__(self, idx):
        return self.y[idx], self.x0[idx]      #  (y , x0)

    
    

def train_nets_process_p(network, train_loader,test_loader, device, idxes,trial, epochs=6000,batch_size=16,g_t=None,exp_root=None,min_epochs=400,slope_epochs=2,quarter_idx=None,mog=0,lr=None,one_network=False,scheduler=None):
    
    betas = torch.tensor(get_named_beta_schedule("linear", 200), dtype=torch.float32, device=device)
    alphas = 1 - betas
    a_bar = torch.cumprod(alphas, 0)
    sqrt_a_bar     = torch.sqrt(a_bar)           # (T,)
    sqrt_one_minus = torch.sqrt(1 - a_bar)
    
    setup_logging(exp_root)
    logging.info(f"--------  device: {device}")
    logging.info(f"epochs,min_epochs,slope_epochs: {epochs},{min_epochs},{slope_epochs}")
    if lr is None:
        lr=0.001
        
    def load_network(network, mog=None, num_instances=1):
        # Dynamically get the class from globals()
        if network in globals():
            if network.endswith("MoG") and mog is not None:
                return [globals()[network](num_mixtures=mog) for _ in range(num_instances)]
            else:
                return [globals()[network]() for _ in range(num_instances)]
        else:
            raise ValueError(f"Unknown network: {network}")
    nets = load_network(network, mog=mog, num_instances=len(idxes))
    
    # nets_min = copy.deepcopy(nets)
    nets_min = load_network(network, mog=mog, num_instances=len(idxes))
    mins=[1000000000 for i in range(len(idxes))]

    
    logging.info(f"idxes[0]: {idxes[0]}")
    logging.info(f"quarter_idx: {quarter_idx}")

    if one_network:
        raise ValueError("one_network is not supported in this function")

    loss_array = {}
    loss_test_array = {}
    a_array = {}
    b_array = {}

    net_counter=-1
    for i in idxes:
        
        
        net_counter+=1
        model = nets[net_counter]
        model.to(device)
        model.train()
        
        if i == 0:
            with torch.no_grad():
                #check if model have raw_a and raw_b
                if hasattr(model, 'raw_b'):
                    model.raw_b.copy_(torch.tensor(2.197)) #0.9

        cur_epochs = int((epochs-min_epochs)*(1-i/200)**slope_epochs+min_epochs)
        cur_lr = None
        if lr is not None:
            cur_lr = lr * (1+ (2*i) / 200) 
            optimizer = optim.AdamW(model.parameters(), lr=cur_lr,weight_decay=1e-2)
        else:
            lr = 0.001

            cur_lr = lr * (1+ (2*i) / 200) 
            optimizer = optim.AdamW(model.parameters(), lr=cur_lr,weight_decay=1e-2)
        if network== "GaussianARStepModel2b":
            # split parameters
            cur_lr = lr 
            scalar_params, conv_params = [], []
            for n, p in model.named_parameters():
                if n in {"raw_a", "raw_b"}:
                    scalar_params.append(p)
                else:
                    conv_params.append(p)

            optimizer = torch.optim.Adam([
                {"params": conv_params,  "lr": cur_lr},   # usual LR
                {"params": scalar_params, "lr": 15*cur_lr},  # ×30 faster for scalars
            ])
        if scheduler is not None:
            # scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min',            # because we want to minimize loss
                factor=0.5,            # reduce LR by this factor
                patience=5,            # epochs with no improvement before reducing LR
                verbose=True           # print updates
            )
                

        for epoch in range(cur_epochs):
            running_loss = 0.0
            
            if epoch == cur_epochs/2 and network== "GaussianARStepModel2b":
                # ---- freeze scalar params ------------------------------------
                optimizer.param_groups[1]["lr"] = 0.0           # no stepping
                for p in scalar_params:
                    p.requires_grad_(False)                    # gradient flow stops
                print(f"[epoch {epoch}]  raw_a/raw_b frozen (lr → 0)")
                
            
            for batch_idx, (y_batch, x0_batch) in enumerate(train_loader):
                optimizer.zero_grad()
                batch_tensor = y_batch.to(device, dtype=torch.float) #y
                # gt_tensor = gt_tensor.to(device, dtype=torch.float)
                x0 = x0_batch.to(device)
                y = batch_tensor
                eps = torch.randn_like(y)
                # x_t = sqrt_a_bar[i] * y + sqrt_one_minus[i] * eps
                x_t = sqrt_a_bar[i] * x0 + sqrt_one_minus[i] * eps 
                mu, log_sigma = model(x_t, batch_tensor)              # forward

                loss = model.casual_loss(mu, log_sigma, y)  
                loss.backward()
                optimizer.step()                
                
                running_loss += loss.item()
                
            if epoch%1==0:
                with torch.no_grad():
                    for batch_idx, (test_inputs, x0_test ) in enumerate(test_loader):
                        y_test = test_inputs.to(device, dtype=torch.float)
                        x0_test = x0_test.to(device)
                        eps = torch.randn_like(y_test)
                        # gt_test = gt_test.to(device, dtype=torch.float)
                        # x_t_test = sqrt_a_bar[i] * y_test + sqrt_one_minus[i] * eps
                        x_t_test = sqrt_a_bar[i] * x0_test + sqrt_one_minus[i] * eps
                        meanst, log_sigmat = model(x_t_test, y_test)
                        loss_t = model.casual_loss( meanst, log_sigmat, y_test)
                   
                if i in loss_test_array:
                    loss_test_array[i].append(float(loss_t))

                    # a_array[i].append(float())
                else:
                    loss_test_array[i] = [float(loss_t)]
                if hasattr(model, 'raw_b') and hasattr(model, 'raw_a'):
                    a_val = torch.sigmoid(model.raw_a).item()
                    b_val = torch.sigmoid(model.raw_b).item()
                    if i not in a_array:
                        a_array[i] = []
                        b_array[i] = []
                    a_array[i].append(float(a_val))
                    b_array[i].append(float(b_val))

                if loss_t < mins[net_counter]:
                    mins[net_counter] = loss_t
                    nets_min[net_counter] = model
            
            if i in loss_array:
                loss_array[i].append(float(loss))
            else:
                loss_array[i] = [float(loss)]
            
            if epoch %1000 == 1:
                if exp_root == None:
                    logging.info("dont have path for graph")
                else:
                    father_path = Path(exp_root)/ f"analysis/graphs/"
                    logging.info(f"{father_path}")
                    father_path.mkdir(parents=True, exist_ok=True)
                    imgpath = father_path/ f"loss_net_{network}_epoch{epoch}.jpg"
                    logging.info("plot_loss")
                    plot_loss(loss_array,loss_test_array,i, network,imgpath)
        if scheduler is not None:
            scheduler.step(loss_t)
        nets[net_counter].parameters = model.parameters
        logging.info(f"Model {i} Epoch {epoch+1}/{cur_epochs}, Loss: {running_loss}")
        
        if i in [0,1,5,10,20,50,51,75,100,125,150,175,199] or net_counter==0:
            if exp_root == None:
                logging.info("dont have path for graph")
            else:
                father_path = Path(exp_root)/ f"analysis/graphs/"
                logging.info(f"{father_path}")
                father_path.mkdir(parents=True, exist_ok=True)
                imgpath = father_path/ f"loss_net_{network}_i{i}.jpg"
                logging.info("plot_loss")
                plot_loss(loss_array,loss_test_array,i, network,imgpath)
                imgpath_a_val = father_path/"a_vals"/ f"a_val_net_{network}_i{i}.jpg"
                imgpath_b_val = father_path/"b_vals"/ f"b_val_net_{network}_i{i}.jpg"
                imgpath_a_val.parent.mkdir(parents=True, exist_ok=True)
                imgpath_b_val.parent.mkdir(parents=True, exist_ok=True)
                if network.startswith("GaussianARStepModel"):
                    if i in a_array:
                        plt.figure()
                        plt.plot(a_array[i], label='a values')
                        plt.xlabel('Epoch')
                        plt.ylabel('a value')
                        plt.title(f'a values for network {network} at i={i}')
                        plt.legend()
                        plt.savefig(imgpath_a_val)
                        plt.close()

                        plt.figure()
                        plt.plot(b_array[i], label='b values')
                        plt.xlabel('Epoch')
                        plt.ylabel('b value')
                        plt.title(f'b values for network {network} at i={i}')
                        plt.legend()
                        plt.savefig(imgpath_b_val)
                        plt.close()
    logging.info(f"test_loss mins array:  {mins}")
    
    return nets_min, loss_array, loss_test_array, quarter_idx



def train_nets_parralel(network,train_loader,test_loader,trial=0, epochs=100,num_nets=200,batch_size=16,g_t=None,exp_root=None,min_epochs=400,slope_epochs=2,mog=0,lr=None,one_network=False,scheduler=None):
    results = []
    gpu_num = torch.cuda.device_count()
    # gpu_num = 1
    if one_network:
        gpu_num=1
    devices = [f'cuda:{i}' for i in range(gpu_num)]
    
    logging.info("parralel")
    logging.info(f"epochs,min_epochs,slope_epochs: ,{epochs},{min_epochs},{slope_epochs}")
    

    numbers = [int((epochs-min_epochs)*(1-i/200)**slope_epochs+min_epochs) for i in range(200)]
    idxes_all = get_group_indices(numbers,num_groups=gpu_num)
    # idxes_all = [list(range(0,100)),list(range(100,200))]
    
    
    if len(devices) >1:
        with mp.get_context('spawn').Pool(processes=gpu_num) as pool:
            args = [(network, train_loader,test_loader,devices[i % gpu_num], idxes, trial,epochs,batch_size,g_t,exp_root,min_epochs,slope_epochs,i,mog,lr,one_network,scheduler) for i, idxes in enumerate(idxes_all)]
            results = pool.starmap(train_nets_process_p, args)

        loss_array= results[0][1]
        loss_test_array = results[0][2]
        nets = results[0][0]
        for i in range(1,gpu_num):
            nets.extend(results[i][0])
            loss_array.update(results[i][1])
            loss_test_array.update(results[i][2])
    else:
        nets, loss_array, loss_test_array, quarter_idx = train_nets_process_p(network, train_loader,test_loader,devices[0], idxes_all[0], trial,epochs,batch_size,g_t,exp_root,min_epochs,slope_epochs,0,mog,lr,one_network,scheduler)
    idxes_all = [list(range(0,200))]

    # nets, loss_array, loss_test_array, quarter_idx = train_nets_process(train_dataset, test_dataset,"cuda:1", idxes_all, trial,epochs=epochs)
    return nets,loss_array,loss_test_array


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
    
class PerClipNormDS(torch.utils.data.Dataset):
    """
    Wraps an existing Dataset that yields
        y  : noisy waveform  (1,L)
        x0 : clean waveform  (1,L)
    and returns normalised tensors *and* the per-clip stats so you can
    de-normalise if you ever need them during evaluation.
    """
    def __init__(self, base_ds, eps: float = 1e-8):
        self.ds  = base_ds
        self.eps = eps

    def __len__(self):                       # unchanged
        return len(self.ds)

    def __getitem__(self, idx):
        y, x0 = self.ds[idx]                 # both [1,L], float32
        # ----- compute clip stats on the CLEAN signal -------------
        mu  = x0.mean()                      # scalar tensor
        std = x0.std().clamp_min(self.eps)   # avoid /0

        y_n  = (y  - mu) / std               # normalised
        x0_n = (x0 - mu) / std
        return y_n, x0_n#, mu, std            # extra outputs
    
    
# -----------------------------------------------------------
# 1.  compute global mean / std once on the *training* set
# -----------------------------------------------------------
def estimate_mean_std(loader):
    """ Return waveform mean and std (scalar) over the whole training set."""
    s1 = 0.0
    s2 = 0.0
    n  = 0
    for noisy, clean in loader:           # each is [B, 1, L]
        wave = clean                     # choose clean or noisy – they have same scale
        s1 += wave.sum()
        s2 += (wave ** 2).sum()
        n  += wave.numel()
    mean = s1 / n
    std  = (s2 / n - mean**2).sqrt()
    return mean.item(), std.item()

# -----------------------------------------------------------
# 2.  use a small, reusable nn.Module
# -----------------------------------------------------------
class ZNorm(nn.Module):
    """
    Forward :  (B,1,L) → (x-mean)/std
    Inverse :  denormalise predictions at inference
    """
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std",  torch.tensor(std))

    def forward(self, x):
        return (x - self.mean) / self.std

    def inverse(self, z):
        return z * self.std + self.mean

# -----------------------------------------------------------
# 3.  wrap the existing dataset so every batch is normalised
# -----------------------------------------------------------
class NormalisedNoiseDS(Dataset):
    def __init__(self, base_ds, mean, std):
        self.ds   = base_ds
        self.norm = ZNorm(mean, std)

    def __len__(self): return len(self.ds)

    def __getitem__(self, idx):
        y, x0 = self.ds[idx]      # original tensors [1,L]
        return self.norm(y), self.norm(x0)


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
    train_noisy_dir = file_id_dir / "train_noisy"
    train_clean_dir = file_id_dir / "train_clean"
    train_noise_dir = file_id_dir / "train_noise"

    noisy_wav_dir.mkdir(parents=True, exist_ok=True)
    noises_dir_out.mkdir(parents=True, exist_ok=True)
    clean_wav_dir.mkdir(parents=True, exist_ok=True)
    train_noisy_dir.mkdir(parents=True, exist_ok=True)
    train_clean_dir.mkdir(parents=True, exist_ok=True)
    train_noise_dir.mkdir(parents=True, exist_ok=True)

    return noisy_wav_dir, noises_dir_out, clean_wav_dir, train_noisy_dir,train_clean_dir,train_noise_dir


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


# def process_wav_file(wav_path, start_seconds, end_seconds, sample_rate=16000):
#     """Load, trim, and calculate sample indices for a WAV file."""
#     audio, sr = torchaudio.load(wav_path)
#     if sr != sample_rate:
#         audio = F.resample(audio, sr, sample_rate)
#     sr=sample_rate
#     start_sample = int(start_seconds * sr)
#     end_sample = int(end_seconds * sr)
#     return audio, sr, start_sample, end_sample


def add_metadata(df, idx, snr, noise_scaling, file_id, noisy_path, noise_path, clean_path, train_noisy_path,train_clean_path):
    """Add metadata to the DataFrame."""
    df.at[idx, "snr"] = snr
    df.at[idx, "noise_scaling"] = noise_scaling
    df.at[idx, "dir"] = file_id
    df.at[idx, "noise_idx"] = file_id
    df.at[idx, "noisy_wav"] = str(noisy_path)
    df.at[idx, "noise_path"] = str(noise_path) if noise_path else None
    df.at[idx, "clean_wav"] = str(clean_path) if clean_path else None
    df.at[idx, "target_train_noisy_path"] = str(train_noisy_path) if train_noisy_path else None
    df.at[idx, "train_clean_path"] = str(train_clean_path) if train_clean_path else None


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



def train_noisemodel(root, network="NetworkNoise2",epochs=1800, dataset_size=1, n_samples=4,batch_size=2,g_t=None, num_nets=200,min_epochs=400,slope_epochs=2,noisy_val_len=1,mog=0,lr=None,one_network=False,scheduler=None,normalize_dataset=False):
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

            train_noisy_path = snr_df["target_train_noisy_path"][train_idx]

            noisy_path = snr_df["noisy_wav"][train_idx]

            
            train_noisy_dir = Path(cur_dir)/"train_noisy"
            train_clean_dir = Path(cur_dir)/"train_clean"
            



            # -------------------------------------------------------------
            def pad_to_length(tensor, target_len):
                return F.pad(tensor, (0, target_len - tensor.size(1))) if tensor.size(1) < target_len else tensor

            # gather matching filenames
            noisy_files  = {p.name: p for p in train_noisy_dir.glob("*.wav")}
            clean_files  = {p.name: p for p in train_clean_dir.glob("*.wav")}
            common_names = sorted(set(noisy_files) & set(clean_files))
            if not common_names:
                raise RuntimeError("No matching clean/noisy WAV pairs!")

            val_name   = common_names[0]   # choose ONE couple for validation
            train_names = common_names[1:] # all others for training

            train_y_list, train_x_list = [], []
            val_y_list,   val_x_list   = [], []

            # ------------------------ iterate -----------------------------
            for name in common_names:
                noisy_wave, sr = torchaudio.load(noisy_files[name])   # (1, L)
                clean_wave, _  = torchaudio.load(clean_files[name])   # (1, L)

                if name == val_name:                 # ---- VALIDATION ----
                    seg_len = int(noisy_val_len * sr)
                    noisy_val  = noisy_wave[:, :seg_len]              # keep first few seconds
                    clean_val  = clean_wave[:, :seg_len]
                    # chunk to (1, seg_len)  ->  reshape_signal gives (1, seg_len_ch)
                    val_y_list.append(torch.tensor(reshape_signal(noisy_val,  batch_size=1),
                                                dtype=torch.float32))
                    val_x_list.append(torch.tensor(reshape_signal(clean_val,  batch_size=1),
                                                dtype=torch.float32))
                else:                                # ---- TRAINING ----
                    # use the WHOLE waveform (no split); chunk into n_samples pieces
                    train_y_list.append(torch.tensor(
                        reshape_signal(noisy_wave, batch_size=n_samples),
                        dtype=torch.float32
                    ))
                    train_x_list.append(torch.tensor(
                        reshape_signal(clean_wave, batch_size=n_samples),
                        dtype=torch.float32
                    ))

            # ------------------- concat with padding ---------------------
            def concat_pad(list_of_tensors):
                max_len = max(t.size(1) for t in list_of_tensors)
                return torch.cat([pad_to_length(t, max_len) for t in list_of_tensors], dim=0)

            train_y_full = concat_pad(train_y_list)   # (B_train, L_max)
            train_x_full = concat_pad(train_x_list)
            test_y_full  = concat_pad(val_y_list)     # (1, L_val_padded)
            test_x_full  = concat_pad(val_x_list)

            # ------------------- DataLoader construction -----------------
            Btrain, Ltrain = train_y_full.shape
            Btest,  Ltest  = test_y_full.shape

            if not normalize_dataset:
                train_dataset = BatchNoiseDataset(
                    noisy_tensor=train_y_full.reshape(Btrain, 1, Ltrain),
                    clean_tensor=train_x_full.reshape(Btrain, 1, Ltrain)
                )
                test_dataset = BatchNoiseDataset(
                    noisy_tensor=test_y_full.reshape(Btest, 1, Ltest),
                    clean_tensor=test_x_full.reshape(Btest, 1, Ltest)
                )
            else:
                base_train = BatchNoiseDataset(train_y_full.reshape(Btrain,1,-1),
                               train_x_full.reshape(Btrain,1,-1))
                base_val   = BatchNoiseDataset(test_y_full.reshape(Btest,1,-1),
                                            test_x_full.reshape(Btest,1,-1))

                train_dataset   = PerClipNormDS(base_train)
                test_dataset     = PerClipNormDS(base_val)
                
                # # (a) build your *raw* dataset first
                # train_raw = BatchNoiseDataset(noisy_tensor=train_y_full.reshape(Btrain,1,-1),
                #                             clean_tensor=train_x_full.reshape(Btrain,1,-1))
                # train_loader_raw = DataLoader(train_raw, batch_size=64, shuffle=True)
                
                # test_raw = BatchNoiseDataset(noisy_tensor=test_y_full.reshape(Btest,1,-1),
                #                             clean_tensor=test_x_full.reshape(Btest,1,-1))
                # # train_loader_raw = DataLoader(train_raw, batch_size=64, shuffle=True)

                # # (b) compute global stats
                # mean, std = estimate_mean_std(train_loader_raw)
                # print(f"Training-set mean={mean:.4f}, std={std:.4f}")

                # # (c) rebuild loaders with normalisation
                # train_dataset  = NormalisedNoiseDS(train_raw,  mean, std)
                # test_dataset   = NormalisedNoiseDS(test_raw,   mean, std)      # **use same stats!**


            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader  = DataLoader(test_dataset,  batch_size=1, shuffle=False)

            
            
            logging.info("starting training")
            nets,loss_array,loss_test_array = train_nets_parralel(network,train_loader,test_loader,trial,epochs=epochs,num_nets=num_nets,batch_size=batch_size,g_t=g_t,exp_root=exp_root,min_epochs=min_epochs,slope_epochs=slope_epochs,mog=mog,lr=lr,one_network=one_network,scheduler=scheduler)
            logging.info("end 1 training")
            
            # params_dict = {"nets": nets, "train_dataset": None, "test_dataset": None,"ar_coefs":None, "loss_array":loss_array, "loss_test_array": loss_test_array, "ar_noise": None, "noise_scaling": cur_noise_scaling, "snr": str(int(cur_snr)), "noise_name": noise_idx, "noise_path": noise_path}
            params_dict = {"nets": [net.state_dict() for net in nets],"network":network, "train_dataset": None, "test_dataset": None,"ar_coefs":None, "loss_array":loss_array, "loss_test_array": loss_test_array, "ar_noise": None, "noise_scaling": cur_noise_scaling, "snr": str(int(cur_snr)), "noise_name": noise_idx, "noise_path": noise_path}
            
            # params_dict = {"result": result}
            params_dict_debug = {"nets": [net.state_dict() for net in nets],"network":network, "train_dataset": (train_dataset), "test_dataset": (test_dataset),"ar_coefs":None, "loss_array":loss_array, "loss_test_array": loss_test_array, "ar_noise": None, "noise_scaling": cur_noise_scaling, "snr": str(int(cur_snr)), "noise_name": noise_idx, "noise_path": noise_path}
            
            #save in name of 0
            pickle_path = cur_dir/(str(0)+"_"+"snr"+str(int(cur_snr))+"_"+str(noise_index)+"_models.pickle")

            tmp_pickle_path = cur_dir/(str(0)+"_"+"tmp_"+"snr"+str(int(cur_snr))+"_"+str(noise_index)+"_models.pickle")
            
    
            try:
                del train_y_full_tensors,train_x_full_tensors
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





def organize_wav_files_snr_levels(exp_root, output_pickle, snr_levels, num_train_seconds=0, num_test_seconds=None, num_train_seconds_noise=0, num_test_seconds_noise=None,train_on_test=False,adapt_train_scale=False):
    """
    Organize files into directories based on file IDs (X), calculate and scale noise for multiple SNR levels,
    save the results, and save metadata to a DataFrame.
    """
    # Directories
    # noisy_dir = Path(exp_root) / "noisy_wav"
    noises_dir = Path(exp_root) / "noises"
    clean_dir = Path(exp_root) / "cleans"
    clean_train_dir = Path(exp_root) / "clean_train"
    output_dir = Path(exp_root)

    # Check if necessary directories exist
    # if not noisy_dir.exists():
    #     raise FileNotFoundError(f"'noisy_wav' directory not found in {exp_root}")
    if not clean_train_dir.exists():
        logging.info(f" 'clean_train' directory not found in {exp_root}. Skipping processing.")
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
        clean_file = next((f for f in os.listdir(clean_dir) if f.endswith(f"fileid_{file_idx}.wav")), None)
        
        # Create directories
        noisy_wav_dir, noises_dir_out, clean_wav_dir, noisy_train_dir,train_clean_dir,noise_train_dir = create_directories(output_dir, file_id)
        if file_id not in created_directories:
            created_directories.append(file_id)

        # File paths
        noise_file = next((f for f in os.listdir(noises_dir) if f.endswith(f"fileid_{file_id}.wav")), None)
        clean_file = next((f for f in os.listdir(clean_dir) if f.endswith(f"fileid_{file_id}.wav")), None)

        if not noise_file or not clean_file:
            logging.info(f"Warning: Missing 'noise' or 'clean' file for file_id {file_id}. Skipping.")
            continue

        # Load clean and noise WAVs
        clean_audio, clean_sr, clean_start, clean_end = process_wav_file(clean_dir / clean_file, num_train_seconds, num_test_seconds)
        if num_test_seconds_noise is None:
            whole_noise_audio, sr_original = torchaudio.load(noises_dir / noise_file)
            num_test_seconds_noise = whole_noise_audio.shape[1] / sr_original
            # num_test_seconds_noise = len_test_seconds_noise - num_train_seconds_noise
            
        if num_train_seconds_noise==None:
            clean_len_in_seconds = (clean_end - clean_start) / clean_sr
            num_train_seconds_noise = num_test_seconds_noise-(clean_len_in_seconds)
        noise_audio, _, _, noise_end = process_wav_file(noises_dir / noise_file, num_train_seconds_noise, num_test_seconds_noise)
        noise_start = noise_end-(clean_end-clean_start)
        
        # Process training noise
        train_noise_audio, n_sr, train_start, train_end = process_wav_file(noises_dir / noise_file, 0, num_train_seconds_noise)
        # train_clean_audio, _, train_clean_start, train_clean_end = process_wav_file(clean_dir / clean_file, 0, num_train_seconds)
        
        sr=16000
        for snr in snr_levels:
            # Scale noise to target SNR
            scaling_factor = calculate_scaling_factor(clean_audio[:, clean_start:clean_end], noise_audio[:, noise_start:noise_end], snr)
            scaled_noise_audio = noise_audio[:, noise_start:noise_end] * scaling_factor
            
            scaled_noise_audio_train = train_noise_audio[:, train_start:train_end] * scaling_factor
            
            # original_snr = noisy_file.split("snr")[-1].split("_")[0]
            new_noisy_name = clean_file.split("fileid")[0] + "snr" +  snr + "_" + "file_id"+ "_"+file_id + ".wav"

            # Save scaled noise
            scaled_noise_name = f"noise{file_idx}_{new_noisy_name}"
            target_noise_path = noises_dir_out / scaled_noise_name
            torchaudio.save(target_noise_path, scaled_noise_audio, sr, encoding="PCM_F")

            # Add scaled noise to clean audio
            noisy_audio = clean_audio[:, clean_start:clean_end] + scaled_noise_audio
            target_noisy_path = noisy_wav_dir / scaled_noise_name
            torchaudio.save(target_noisy_path, noisy_audio, sr, encoding="PCM_F")
            
            clean_save_path = clean_wav_dir / f"noise{file_idx}_{new_noisy_name}"
            torchaudio.save(clean_save_path, clean_audio[:, clean_start:clean_end], sr, encoding="PCM_F")
            
            clean_train_save_dir = train_clean_dir / f"noise{file_idx}_{new_noisy_name}".replace(".wav", "")

            # Scale and save training noisy
            # scaled_train_noise_audio = train_noise_audio[:, train_start:train_end] * scaling_factor
            # noisy_audio_train = train_clean_audio[:,train_clean_start:train_clean_end] + scaled_noise_audio_train
            # if train_on_test:
            #     noisy_audio_train = noisy_audio
            train_noisy_name = f"noise{file_idx}_{new_noisy_name}"
            target_train_noisy_dir = noisy_train_dir# / train_noisy_name.replace(".wav", "")
            target_train_noise_dir = noise_train_dir#/ train_noisy_name.replace(".wav", "")
            target_clean_train_save_dir = train_clean_dir# / train_noisy_name.replace(".wav", "")
            target_train_noisy_dir.mkdir(parents=True, exist_ok=True)
            target_train_noise_dir.mkdir(parents=True, exist_ok=True)
            target_clean_train_save_dir.mkdir(parents=True, exist_ok=True)
            clean_train_idx_dir =  clean_train_dir/("fileid_" + file_id   )
            offset = 0
            
            total_noise_len = scaled_noise_audio_train.shape[1]
            
            # ------------------------------------------------------------------
            # slice training clips — guarantees equal length by trimming only
            # ------------------------------------------------------------------
            noise_root_dir    = Path(exp_root) / "noise_train"
            use_parallel_noise  = noise_root_dir.exists()
            offset = 0
            total_noise_len = scaled_noise_audio_train.shape[1]

            for clean_train_path in clean_train_idx_dir.glob("*.wav"):
                # ---- load clean wav, force mono ----
                clean_waveform, sr, _, _ = process_wav_file(clean_train_path, 0, None)
                if clean_waveform.dim() == 1:           # [L] → [1, L]
                    clean_waveform = clean_waveform.unsqueeze(0)

                # ---- (optional) scale clean for loudness alignment ----
                if adapt_train_scale:
                    peak      = clean_waveform.abs().max().item()
                    ref_peak  = clean_audio[:, clean_start:clean_end].abs().max().item()
                    scaling_factor = ref_peak / (peak + 1e-9)
                else:
                    scaling_factor = 1.0
                clean_waveform *= scaling_factor

                # ---------------- I/O paths ----------------
                noise_train_save_path = target_train_noise_dir / clean_train_path.name
                noisy_train_save_path = target_train_noisy_dir / clean_train_path.name
                clean_train_save_path = target_clean_train_save_dir / clean_train_path.name

                # save the (possibly rescaled) clean training chunk
                torchaudio.save(clean_train_save_path, clean_waveform, sr, encoding="PCM_F")
                
                if use_parallel_noise:
                    # expected parallel noise location:  noise_train/fileid_<file_id>/<same filename>.wav
                    parallel_dir = noise_root_dir / f"fileid_{file_id}"
                    parallel_file = parallel_dir / clean_train_path.name
                    if parallel_file.exists():
                        noise_segment, _, _, _ = process_wav_file(parallel_file, 0, None)
                        if noise_segment.dim() == 1:
                            noise_segment = noise_segment.unsqueeze(0)
                        # Trim both tensors to the shorter length
                        L = min(clean_waveform.size(1), noise_segment.size(1))
                        clean_waveform = clean_waveform[:, :L]
                        noise_segment  = noise_segment[:,  :L]
                    else:
                        logging.warning(f"Missing parallel noise: {parallel_file}. Falling back to wrap-around.")
                        noise_segment = None   # triggers fallback
                
                if not use_parallel_noise:
                    # ---- pull a noise slice of ≥ desired length, wrap if needed ----
                    Lc = clean_waveform.shape[1]                      # target length
                    if Lc <= total_noise_len - offset:
                        noise_segment = scaled_noise_audio_train[:, offset:offset + Lc]
                    else:
                        part1     = scaled_noise_audio_train[:, offset:]
                        remain    = Lc - part1.shape[1]
                        part2     = scaled_noise_audio_train[:, :remain]
                        noise_segment = torch.cat([part1, part2], dim=1)
                    offset = (offset + Lc) % total_noise_len          # advance pointer

                # ---- FINAL safety-trim so both tensors are exactly equal ----
                L = min(clean_waveform.shape[1], noise_segment.shape[1])
                clean_waveform = clean_waveform[:, :L]
                noise_segment  = noise_segment[:,  :L]

                # ---- scale noise to requested SNR ----
                scaling_factor = calculate_scaling_factor(clean_waveform, noise_segment, snr)
                noise_segment *= scaling_factor

                # ---- save noise & noisy mixture ----
                torchaudio.save(noise_train_save_path, noise_segment, sr, encoding="PCM_F")
                noisy_waveform = clean_waveform + noise_segment
                torchaudio.save(noisy_train_save_path, noisy_waveform, sr, encoding="PCM_F")

            # for clean_train_path in clean_train_idx_dir.glob("*.wav"):
            #     clean_waveform, sr, _, _ = process_wav_file(clean_train_path, 0, None)
            #     if clean_waveform.dim() == 1:           # ensure [1, L]
            #         clean_waveform = clean_waveform.unsqueeze(0)

            #     # print(0)
            #     noise_train_save_path = target_train_noise_dir / clean_train_path.name
            #     noisy_train_save_path = target_train_noisy_dir / clean_train_path.name
            #     clean_train_save_path = target_clean_train_save_dir/ clean_train_path.name
            #     # Skip if this file was already processed
            #     # if noise_train_save_path.exists():
            #     #     continue

            #     # Load the clean waveform and sampling rate
            #     # clean_waveform, sr,_,_= process_wav_file(clean_train_path, 0, None)

            #     # Scale the clean waveform if a scaling factor is provided
            #     if adapt_train_scale:
            #         # scaling_factor = calculate_scaling_factor(clean_audio[:, clean_start:clean_end], clean_waveform, 0)
            #         peak = clean_waveform.abs().max().item()
            #         ref_peak = clean_audio[:, clean_start:clean_end].abs().max().item()
            #         scaling_factor = ref_peak / peak 
            #     else:
            #         scaling_factor = torch.tensor(1.0)
            #     clean_waveform = clean_waveform * scaling_factor
            #     torchaudio.save(clean_train_save_path, clean_waveform, sr, encoding="PCM_F")

            #     # Select a noise segment of the same length as the clean waveform (wrap around if needed)
            #     length = clean_waveform.shape[1]
            #     total_noise_len = scaled_noise_audio_train.shape[1]
            #     if length <= total_noise_len - offset:
            #         noise_segment = scaled_noise_audio_train[:, offset:offset + length]
            #     else:
            #         part1 = scaled_noise_audio_train[:, offset:]
            #         remaining = length - (total_noise_len - offset)
            #         part2 = scaled_noise_audio_train[:, :remaining]
            #         noise_segment = torch.cat((part1, part2), dim=1)
            #     offset = (offset + length) % total_noise_len

            #     scaling_factor = calculate_scaling_factor( clean_waveform, noise_segment, snr)
            #     noise_segment = noise_segment * scaling_factor
            #     # Save the noise segment and the combined noisy waveform
            #     torchaudio.save(noise_train_save_path, noise_segment, sample_rate=sr, encoding='PCM_F')
            #     noisy_waveform = clean_waveform + noise_segment
            #     torchaudio.save(noisy_train_save_path, noisy_waveform, sample_rate=sr, encoding='PCM_F')
                                                        
            # Add metadata
            df_idx += 1 
            add_metadata(
                df_snr, df_idx, snr, float(scaling_factor.item()), file_id, target_noisy_path,
                target_noise_path, clean_save_path, target_train_noisy_dir,clean_train_save_dir
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
        default="exps_configs/libr20BBC_p_net2_6_snr5.yaml",
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
    noisy_val_len = trials.get("val_len","1")
    adapt_train_scale = trials.get("adapt_train_scale", False)
    normalize_dataset = trials.get("normalize_dataset", False)
    loss_model = "y_model"
    if normalize_dataset == True:
        loss_model = "y_model_norm"
     
    # noisy_test_len = test_end_sec- test_start_sec #test end is the clean and the noisy
    
    test_start_sec_noise = trials.get("test_start_sec_noise",test_start_sec)
    test_end_sec_noise = trials.get("test_end_sec_noise",test_end_sec)
    trained_model_path = trials.get("trained_model_path","0")
    
    mog = trials.get("mog", 0) 
    lr = trials.get("lr", 0.001 )
    one_network = trials.get("one_network", False) 
    training_scheduler = trials.get("training_scheduler", None) 
    train_on_test = trials.get("train_on_test", False)
    print(train_on_test)
    
    log_file = setup_logging(exp_root)
    logging.info(f"\nDir: {args.config}\n")
    logging.info(f"Logging to: {log_file}")

    
    betas=get_named_beta_schedule(scheduler_type, num_steps)
    alphas = 1.0 - betas
    alphas_cumprod =  torch.from_numpy(np.cumprod(alphas, axis=0))
    g_t = torch.sqrt((1-alphas_cumprod)/(alphas_cumprod))
    

    output_pickle_path = Path(exp_root)/"5f_snrs.pickle"
    
    
    from pathlib import Path
    import shutil

    root_dir = Path(exp_root)  # Replace with your actual path

            
    for wav_path in root_dir.rglob("*.WAV"):
        new_path = wav_path.with_suffix(".wav")
        
        # if not new_path.exists():
            # Load the original .WAV file
        audio, sr = torchaudio.load(wav_path)
        
        # Multiply waveform by 5
        audio_scaled = audio * 4
        
        # Save to .wav with PCM_F encoding
        torchaudio.save(new_path.as_posix(), audio_scaled, sr, encoding="PCM_F")
        print(f"Processed and saved: {new_path}")
        # else:
        #     print(f"Skipped (already exists): {new_path}")

    
    save_audio = False
    if len(snr_array)==0:
        names, noises_names,snr_array = organize_wav_files(exp_root, output_pickle_path, num_train_seconds=test_start_sec, num_test_seconds=test_end_sec, save_audio=save_audio)
    else:
        names, noises_names,snr_array = organize_wav_files_snr_levels(exp_root, output_pickle_path, snr_array, num_train_seconds=test_start_sec, num_test_seconds=test_end_sec,  num_train_seconds_noise=test_start_sec_noise, num_test_seconds_noise=test_end_sec_noise,train_on_test=train_on_test,adapt_train_scale=adapt_train_scale)
    
    run_network=None 
    if trained_model_path == "0":
        logging.info("---startin training---")
        train_noisemodel(exp_root, network=network,epochs=epochs, dataset_size=dataset_size, n_samples=n_samples,batch_size=batch_size,g_t=g_t,num_nets=num_steps,min_epochs=min_epochs,slope_epochs=slope_epochs,noisy_val_len=noisy_val_len,mog=mog,lr=lr,one_network=one_network,scheduler=training_scheduler,normalize_dataset=normalize_dataset)
    else:
        run_network=  network
    # names=["6","18"]
    # noises_names = ["6","18"]
    
    logging.info("---run_exp---")
    run_exp(exp_root, dirnames=names,s_array=s_array, reset=False, s_schedule=scheduler, scheduler_type=scheduler_type,noise_mosel_path=trained_model_path, network=run_network,mog=mog,loss_model=loss_model)
    
    storm_root = str(Path(exp_root)/"storm")
    run_storm(exp_root,storm_root) #200 steps
    from run_storm_measure import measure_storm
    storm_root = os.path.join(exp_root, "storm")
    measure_storm(storm_root)
    
    logging.info("---analyzing---")
    analyze_exp(exp_root,noises_names,snr_array,names,specific_s=None)
    
    ours_results_path = Path(exp_root)/"analysis"/"ours_all.xlsx"
    winner_s = choose_closest_to_median(ours_results_path)
    analyze_exp(exp_root,noises_names,snr_array,names,specific_s=winner_s,output_namedir="analysis_specific_s",)



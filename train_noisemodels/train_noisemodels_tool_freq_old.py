import glob
import pathlib
import pandas as pd
import seaborn as sns
# import tensorflow as tf
import os
import random

import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
from IPython import display
from IPython import display
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
import torch.utils.data
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
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


#calc likelihood:
def calc_real_likelihood(inputs):
    SIGMA=1
    sum_arg=0
    wav_data2 = inputs.squeeze()

    for i in range(len(wav_data2)-1):
        if i==0:
            sum_arg += (wav_data2[0] - 0)**2
        else:
            sum_arg += (wav_data2[i] - 0.9*wav_data2[i-1])**2

    likelihood = sum_arg*(-0.5)*((1/SIGMA)**2) + len(wav_data2)*np.log(1/(np.sqrt(2*np.pi)*SIGMA))

    return likelihood

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


def create_ar_noise():
    AR_COEF_MIN = 0.95
    AR_COEF_MAX = 0.95
    order=1
    # AR_COEF = 0.9

    n_samples=8000
    mu=0
    sigma=1
    """Generate auto-regressive noise."""
    # Generate white noise
    white_noise = np.random.normal(mu, sigma, n_samples + order)
    # return np.random.normal(mu, sigma, n_samples)
    # # Initialize AR coefficients randomly
    # # ar_coefs = np.random.uniform(-0.5, 0.5, order)
    ar_coefs = np.random.uniform(AR_COEF_MIN,AR_COEF_MAX, order)

    # Generate AR noise
    ar_noise = np.zeros_like(white_noise)
    for i in range(order, n_samples + order):
        ar_noise[i] = np.dot(ar_coefs, ar_noise[i - order:i]) + white_noise[i]
    ar_noise = ar_noise[order:]  # Discard initial transient
    return ar_noise

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear": ###chosen in default
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        # scale = 1000 / num_diffusion_timesteps
        beta_start = 0.0001  # scale * 0.0001
        beta_end = 0.02  # scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
            
        )
        
def create_dataset(cur_noise_scaling):
    ar_noise = cur_noise_scaling*create_ar_noise()
    ar_noise2 = cur_noise_scaling*create_ar_noise()

    # Create datasets and data loaders
    train_tensor = torch.tensor(ar_noise, dtype=torch.float32).view(1,1,-1)
    test_tensor = torch.tensor(ar_noise2, dtype=torch.float32).view(1,1,-1)


    betas=get_named_beta_schedule("linear", 200)

    alphas = 1.0 - betas
    alphas_cumprod =  torch.from_numpy(np.cumprod(alphas, axis=0))
    # white_noise_diffusion = torch.normal(0,1,train_tensor.shape)
    g_t = -torch.sqrt((1-alphas_cumprod)/(alphas_cumprod))

    train_full_tensors = train_tensor.squeeze().repeat(200,1).view(200,1,-1)
    for i in range(200):
        cur_white_noise_diffusion = torch.normal(0,1,train_tensor.shape)
        train_full_tensors[i,:,:] = train_full_tensors[i,:,:]+cur_white_noise_diffusion*g_t[i]

    test_full_tensors = test_tensor.squeeze().repeat(200,1).view(200,1,-1)
    for i in range(200):
        cur_white_noise_diffusion = torch.normal(0,1,test_tensor.shape)
        test_full_tensors[i,:,:] = test_full_tensors[i,:,:]+cur_white_noise_diffusion*g_t[i]
        

    #Create TensorDatasets
    train_dataset = NoiseDataset(train_full_tensors,g_t)
    test_dataset = NoiseDataset(test_full_tensors,g_t)

    #Create DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    return train_dataset, test_dataset, ar_noise, ar_noise2



    
def train_nets(train_dataset):
    n_fft = 512 
    win_length = n_fft 
    hop_length = int(win_length/2) 
    window = torch.hann_window(win_length) 
    models = [{} for i in range(200)]
    for i,model in enumerate(models):
        batch_tensor, gt_tensor = train_dataset.__getitem__(i)
        batch_tensor = batch_tensor.view(-1)#.to("cuda", dtype=torch.float)

        stft = torch.stft(batch_tensor, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)
        magnitude = torch.abs(stft)
        stds = torch.std(magnitude,dim=1)
        means = torch.mean(magnitude,dim=1)
        stats = (means, stds)

        models[i]["stats"] = stats
        models[i]["params"] = (n_fft,hop_length,win_length,window)
    return models

# def create_noisy_files(root):
    
def create_dataset_real(noise_whole, sr):
    noise_sample1 = noise_whole[:, 0:int(sr/2)] ############todo: change
    noise_sample2 = noise_whole[:, int(sr/2):sr]

    # Create datasets and data loaders
    train_tensor = torch.tensor(noise_sample1, dtype=torch.float32).view(1,1,-1)
    test_tensor = torch.tensor(noise_sample2, dtype=torch.float32).view(1,1,-1)


    betas=get_named_beta_schedule("linear", 200)

    alphas = 1.0 - betas
    alphas_cumprod =  torch.from_numpy(np.cumprod(alphas, axis=0))
    # white_noise_diffusion = torch.normal(0,1,train_tensor.shape)
    g_t = -torch.sqrt((1-alphas_cumprod)/(alphas_cumprod))

    train_full_tensors = train_tensor.squeeze().repeat(200,1).view(200,1,-1)
    for i in range(200):
        cur_white_noise_diffusion = torch.normal(0,1,train_tensor.shape)
        train_full_tensors[i,:,:] = train_full_tensors[i,:,:]+cur_white_noise_diffusion*g_t[i]

    test_full_tensors = test_tensor.squeeze().repeat(200,1).view(200,1,-1)
    for i in range(200):
        cur_white_noise_diffusion = torch.normal(0,1,test_tensor.shape)
        test_full_tensors[i,:,:] = test_full_tensors[i,:,:]+cur_white_noise_diffusion*g_t[i]
        

    #Create TensorDatasets
    train_dataset = NoiseDataset(train_full_tensors,g_t)
    test_dataset = NoiseDataset(test_full_tensors,g_t)

    #Create DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    return train_dataset, test_dataset, noise_sample1, noise_sample2




if __name__ == "__main__":
    print("starting ----")
    # root = "/data/ephraim/datasets/known_noise/undiff/exp_ar_i_real_freq/"
    root = "/data/ephraim/datasets/known_noise/undiff/exp_ar_i_real_freq_begin/"
    

    with open(Path(root)/'5f_snrs.pickle', 'rb') as handle:
        snr_df = pickle.load(handle)
    print(snr_df)
    
    failed = []
    
    # for i in tqdm(range(len(snr_df["snr"]))):
        # print(i+1)
        # if i>0:
        #     break
    
    failed = {}
    i=snr_df[snr_df["dir"]=="b"].index[2]
    # for trial in [3,1,2]:
    for i in snr_df.index:
        for trial in [0]:
                
            train_idx = i
            noise_idx = snr_df["noise_idx"][train_idx]
            cur_dir = Path(root)/ snr_df["dir"][train_idx]
            print(f"______{cur_dir}_______")

            cur_snr = snr_df["snr"][train_idx]
            cur_noise_scaling = snr_df["noise_scaling"][train_idx]
            noise_index = snr_df["noise_idx"][train_idx]
            print(cur_snr,cur_dir, noise_index, cur_noise_scaling)

            noise_path = snr_df["noise_path"][train_idx]
            speech_path = snr_df["clean_wav"][train_idx]
            noisy_path = snr_df["noisy_wav"][train_idx]
            noise_whole, sr = torchaudio.load(noise_path)

            train_dataset, test_dataset, noise_sample1, noise_sample2 = create_dataset_real(noise_whole, sr)
                
            # nets, loss_array, loss_test_array = train_nets(train_dataset, test_dataset)
            models = train_nets(train_dataset)
            
            params_dict = {"models": models}

            
            pickle_path = cur_dir/(str(trial)+"_"+"snr"+str(int(cur_snr))+"_"+str(noise_index)+"_models.pickle")


            try:
                with open(pickle_path, 'wb') as handle:
                    pickle.dump(params_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print(pickle_path)
            except:
                print(f"failed in {i}")
                failed.append(i)
                # continue
    print("failed: ", failed)
    print(pickle_path)
    
    # command = "python run_exp_ar_i_95.py"
    # os.system(command)
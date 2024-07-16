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


def CausalConv1d(in_channels, out_channels, kernel_size, dilation=1, **kwargs):
   pad = (kernel_size - 1) * dilation +1
   return nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad, dilation=dilation, **kwargs)


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = CausalConv1d(1, 2, kernel_size=3, dilation=1)

    def forward(self, x, cur_gt):
        x = self.conv1(x)
        # print("self.conv1.padding: ", self.conv1.padding)
        if self.conv1.padding[0] != 0:
            x = x[:, :, :-self.conv1.padding[0]-1]  # remove trailing padding
        means = x[:,0,:]
        log_var = x[:,1,:]
        stds = torch.exp(0.5 *log_var)
        return means, stds
    
    def calc_model_likelihood(self, expected_means, expected_stds, wav_tensor, verbose=False):
        wav_tensor = wav_tensor.squeeze()
        means_=expected_means.squeeze()
        stds_ = expected_stds.squeeze()

        exp_all = -(1/2)*((torch.square(wav_tensor-means_)/torch.square(stds_)))
        param_all = 1/(np.sqrt(2*np.pi)*stds_)
        model_likelihood1 = torch.sum(torch.log(param_all), axis=-1) 
        model_likelihood2 = torch.sum(exp_all, axis=-1) 

        if verbose:
            print("model_likelihood1: ", model_likelihood1)
            print("model_likelihood2: ", model_likelihood2)
        return model_likelihood1 + model_likelihood2
    
    def casual_loss(self, expected_means, expected_stds, wav_tensor):
        model_likelihood = self.calc_model_likelihood(expected_means, expected_stds, wav_tensor)
        return -model_likelihood


def calc_model_likelihood(tensor_, model, curgt):
    with torch.no_grad():
        output = (model(tensor_.to("cuda"), curgt.to("cuda", dtype=torch.float)))


    wav_tensor = tensor_.squeeze()
    means_=output[0].squeeze()
    stds_ = output[1].squeeze()

    exp_all = -(1/2)*((torch.square(wav_tensor-means_)/torch.square(stds_)))
    param_all = 1/(np.sqrt(2*np.pi)*stds_)
    model_likelihood1 = torch.sum(torch.log(param_all), axis=-1) 
    model_likelihood2 = torch.sum(exp_all, axis=-1) 
    model_likelihood = model_likelihood1 + model_likelihood2
    
    return model_likelihood

    
def train_nets(train_dataset, test_dataset,epochs=6000):
    nets = [Network() for i in range(200)]


    loss_array = {}
    loss_test_array = {}

    for i,model in enumerate(nets):
        model.to("cuda")
        model.train()

        optimizer = optim.Adam(model.parameters())
        for epoch in range(epochs):
            running_loss = 0.0
            # for batch_idx, (batch_tensor, gt_tensor) in enumerate(train_loader):
            batch_tensor, gt_tensor = train_dataset.__getitem__(i)
            # for data in train_loader:
            optimizer.zero_grad()
            batch_tensor = batch_tensor.view(1,1,-1).to("cuda", dtype=torch.float)
            gt_tensor = gt_tensor.to("cuda", dtype=torch.float)
            # print("batch_tensor.shape:",batch_tensor.shape)
            # print()
            means, stds = model(batch_tensor, gt_tensor)


            loss = model.casual_loss( means, stds, wav_tensor=batch_tensor)
            # print("loss",loss)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
                
                
            if epochs%100==0:
                with torch.no_grad():
                    test_inputs, gt_test = test_dataset.__getitem__(i)
                    test_inputs = test_inputs.view(1,1,-1).to("cuda", dtype=torch.float)
                    gt_test = gt_test.to("cuda", dtype=torch.float)
                    meanst, stdst = model(test_inputs, gt_test)
                loss_t = model.casual_loss( meanst, stdst, wav_tensor=test_inputs)
                if i in loss_test_array:
                    loss_test_array[i].append(float(loss_t))
                else:
                    loss_test_array[i] = [float(loss_t)]
            
                # print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss}")  #/ len(train_loader
            # loss_array[i].append(float(loss))
            if i in loss_array:
                loss_array[i].append(float(loss))
            else:
                loss_array[i] = [float(loss)]
        # print("gt_tensor:", float(gt_tensor))
        nets[i].parameters = model.parameters
        print(f"Model {i} Epoch {epoch+1}/{epochs}, Loss: {running_loss}")
    return nets, loss_array, loss_test_array

# def create_noisy_files(root):
    



if __name__ == "__main__":
    print("starting ----")


    with open('/data/ephraim/datasets/known_noise/undiff/exp_ar_i_095/scaling_snrs.pickle', 'rb') as handle:
        snr_df = pickle.load(handle)
    root = "/data/ephraim/datasets/known_noise/undiff/exp_ar_i_095/"
    snr_df
    
    failed = []
    for i in tqdm(range(len(snr_df["snr"]))):
        print(i)
        train_idx = i+1
        if snr_df["dir"][train_idx] not in ["a", "b","c", "d","e"]:
            continue
        cur_dir = Path(root +snr_df["dir"][train_idx])
        cur_snr = snr_df["snr"][train_idx]
        cur_noise_scaling = snr_df["noise_scaling"][train_idx]
        print(cur_snr,cur_dir, cur_noise_scaling)
        pickle_path = cur_dir/("snr"+str(int(cur_snr))+"_models.pickle")

        train_dataset, test_dataset, ar_noise, ar_noise2 = create_dataset(cur_noise_scaling)
        
        nets, loss_array, loss_test_array = train_nets(train_dataset, test_dataset)
        
        params_dict = {"nets": nets, "train_dataset": None, "test_dataset": None,"ar_coefs":None, "loss_array":loss_array, "loss_test_array": loss_test_array, "ar_noise": ar_noise, "noise_scaling": cur_noise_scaling, "snr": str(int(cur_snr))}
        try:
            with open(pickle_path, 'wb') as handle:
                pickle.dump(params_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except:
            print(f"failed in {i}")
            failed.append(i)
            continue
    print("failed: ", failed)
    
    
    command = "python run_exp_ar_i_95.py"
    os.system(command)
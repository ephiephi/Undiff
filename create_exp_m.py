import torchaudio
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
from IPython import display
from matplotlib import pyplot as plt
from typing import Optional
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
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
import IPython
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

from analyze.analyze_exp import analyze_exp
from run_storm import run_storm


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
        
# betas=get_named_beta_schedule("linear", 200)

# alphas = 1.0 - betas
# alphas_cumprod =  torch.from_numpy(np.cumprod(alphas, axis=0))
# g_t = torch.sqrt((1-alphas_cumprod)/(alphas_cumprod))



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
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)
        




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



class BatchNoiseDataset(Dataset):
    def __init__(self, data_tensor, gt_tensor):
        self.data_tensor =data_tensor
        self.gt_tensor = gt_tensor

    def __len__(self):
        return self.data_tensor.shape[0]

    def __getitem__(self, idx):
        item = self.data_tensor[idx,:,:]
        cur_gt = self.gt_tensor
        return item, cur_gt


def create_ar_noise_batch(batch_size, n_samples, ar_coefs, order=1, dtype=torch.float32):
    """
    Efficiently generate a batch of AR noise samples using matrix operations.
    
    Parameters:
        batch_size (int): Number of AR noise samples in the batch.
        n_samples (int): Length of each AR noise sample.
        AR_COEF_MIN (float): Minimum value for AR coefficients.
        AR_COEF_MAX (float): Maximum value for AR coefficients.
        order (int): The order of the AR process.
        dtype (torch.dtype): The desired PyTorch tensor data type.

    Returns:
        torch.Tensor: Batch of AR noise samples with shape (batch_size, n_samples).
    """
    # Initialize white noise for the entire batch
    white_noise = torch.normal(0, 1, size=(batch_size, n_samples + order), dtype=dtype)

    # Convert AR coefficients to a tensor if needed
    if not isinstance(ar_coefs, torch.Tensor):
        ar_coefs = torch.tensor(ar_coefs, dtype=dtype)
    
    # Ensure AR coefficients are of the correct order
    assert ar_coefs.shape[0] == order, f"AR coefficients must have length {order}."
    
    # Expand AR coefficients for batch processing
    ar_coefs = ar_coefs.unsqueeze(0).expand(batch_size, -1)  # Shape: (batch_size, order)


    # Initialize the AR noise tensor
    ar_noise = torch.zeros_like(white_noise)

    # Iterate across time steps to compute AR noise
    for t in range(order, n_samples + order):
        # Slice the relevant previous values for all samples
        past_values = ar_noise[:, t - order:t]  # Shape: (batch_size, order)
        # Compute AR noise using matrix multiplication
        ar_noise[:, t] = (past_values * ar_coefs).sum(dim=1) + white_noise[:, t]

    # Remove the first `order` values to discard transient
    return ar_noise[:, order:]




class CausalConv1dClassS(nn.Conv1d):
    def __init__(self,in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        pad = (kernel_size - 1) * dilation
        super().__init__(in_channels, out_channels, kernel_size, padding=pad, dilation=dilation, **kwargs)
    
    def forward(self, inputs):
        output = super().forward(inputs)
        if self.padding[0] != 0:
            output = output[:, :, :-self.padding[0]]
        return output
    
class NetworkNoise2(nn.Module):
    def __init__(self, kernel_size=9):
        super().__init__()
        self.kernel_size=kernel_size
        self.conv0 = CausalConv1dClassS(1, 2, kernel_size=kernel_size, dilation=8)
        self.tanh0 = nn.Tanh()
        self.conv1 = CausalConv1dClassS(1, 2, kernel_size=kernel_size, dilation=1)
        self.tanh1 = nn.Tanh()
        self.conv2 = CausalConv1dClassS(2, 2, kernel_size=kernel_size, dilation=2)
        self.tanh2 = nn.Tanh()
        self.conv3 = CausalConv1dClassS(2, 2, kernel_size=kernel_size, dilation=4)
        self.tanh3 = nn.Tanh()
        self.conv4 = CausalConv1dClassS(2, 2, kernel_size=kernel_size, dilation=8)
        # self.b = nn.Parameter(torch.tensor(0.5))  # Initial value of 'b'


    def forward(self, x, cur_gt):

        x1 = self.conv1(x)
        x2 = self.conv0(x)
        x = self.tanh1(x1)
        x = self.conv2(x)
        x = self.tanh2(x)
        x = self.conv3(x)
        x = self.tanh3(x)
        x = self.conv4(x)
        x = x+x1 + x2


        means = x[:,0,:]
        log_var = x[:,1,:]
        stds = torch.exp(0.5 *log_var)
        # stds = torch.ones_like(means)*self.b
        return means, stds
    
    def calc_model_likelihood(self, expected_means, expected_stds, wav_tensor, verbose=False):
        wav_tensor = wav_tensor.squeeze(axis=1)[:,self.kernel_size+1:]
        means_=expected_means.squeeze(axis=1)[:,self.kernel_size:-1]
        stds_ = expected_stds.squeeze(axis=1)[:,self.kernel_size:-1]
        # print(wav_tensor.shape)
        # print(means_.shape)

        exp_all = -(1/2)*((torch.square(wav_tensor-means_)/torch.square(stds_)))
        param_all = 1/(np.sqrt(2*np.pi)*stds_)
        model_likelihood1 = torch.sum(torch.log(param_all), axis=-1) 
        model_likelihood2 = torch.sum(exp_all, axis=-1) 

        if verbose:
            print("model_likelihood1: ", model_likelihood1)
            print("model_likelihood2: ", model_likelihood2)
        likelihood = model_likelihood1 + model_likelihood2
        return likelihood.mean()
    
    def casual_loss(self, expected_means, expected_stds, wav_tensor):
        model_likelihood = self.calc_model_likelihood(expected_means, expected_stds, wav_tensor)
        return -model_likelihood   


class NetworkNoise3(nn.Module):
    def __init__(self, kernel_size=9):
        super().__init__()
        self.kernel_size=kernel_size
        self.conv1 = CausalConv1dClassS(1, 2, kernel_size=kernel_size, dilation=1)
        self.tanh1 = nn.Tanh()
        self.conv2 = CausalConv1dClassS(2, 2, kernel_size=kernel_size, dilation=2)
        self.tanh2 = nn.Tanh()
        self.conv3 = CausalConv1dClassS(2, 2, kernel_size=kernel_size, dilation=4)
        self.tanh3 = nn.Tanh()
        self.conv4 = CausalConv1dClassS(2, 2, kernel_size=kernel_size, dilation=8)
        # self.b = nn.Parameter(torch.tensor(0.5))  # Initial value of 'b'


    def forward(self, x, cur_gt):

        x1 = self.conv1(x)
        x = self.tanh1(x1)
        x = self.conv2(x)
        x = self.tanh2(x)
        x = self.conv3(x)
        x = self.tanh3(x)
        x = self.conv4(x)
        x = x+x1

        means = x[:,0,:]
        log_var = x[:,1,:]
        stds = torch.exp(0.5 *log_var)
        # stds = torch.ones_like(means)*self.b
        return means, stds
    
    def calc_model_likelihood(self, expected_means, expected_stds, wav_tensor, verbose=False):
        wav_tensor = wav_tensor.squeeze(axis=1)[:,self.kernel_size+1:]
        means_=expected_means.squeeze(axis=1)[:,self.kernel_size:-1]
        stds_ = expected_stds.squeeze(axis=1)[:,self.kernel_size:-1]
        # print(wav_tensor.shape)
        # print(means_.shape)

        exp_all = -(1/2)*((torch.square(wav_tensor-means_)/torch.square(stds_)))
        param_all = 1/(np.sqrt(2*np.pi)*stds_)
        model_likelihood1 = torch.sum(torch.log(param_all), axis=-1) 
        model_likelihood2 = torch.sum(exp_all, axis=-1) 

        if verbose:
            print("model_likelihood1: ", model_likelihood1)
            print("model_likelihood2: ", model_likelihood2)
        likelihood = model_likelihood1 + model_likelihood2
        return likelihood.mean()
    
    def casual_loss(self, expected_means, expected_stds, wav_tensor):
        model_likelihood = self.calc_model_likelihood(expected_means, expected_stds, wav_tensor)
        return -model_likelihood   
    

class NetworkNoise4(nn.Module):
    def __init__(self, kernel_size=9):
        super().__init__()
        self.kernel_size=kernel_size
        self.conv1 = CausalConv1dClassS(1, 2, kernel_size=kernel_size, dilation=1)
        self.tanh1 = nn.Tanh()
        self.conv2 = CausalConv1dClassS(2, 2, kernel_size=kernel_size, dilation=2)
        self.tanh2 = nn.Tanh()
        self.conv3 = CausalConv1dClassS(2, 2, kernel_size=kernel_size, dilation=4)
        self.tanh3 = nn.Tanh()
        self.conv4 = CausalConv1dClassS(2, 2, kernel_size=kernel_size, dilation=8)
        self.tanh4 = nn.Tanh()
        
        self.conv5 = CausalConv1dClassS(2, 2, kernel_size=kernel_size, dilation=8)
        self.tanh5 = nn.Tanh()
        self.conv6 = CausalConv1dClassS(2, 2, kernel_size=kernel_size, dilation=4)
        self.tanh6 = nn.Tanh()
        self.conv7 = CausalConv1dClassS(2, 2, kernel_size=kernel_size, dilation=2)
        self.tanh7 = nn.Tanh()
        self.conv8 = CausalConv1dClassS(2, 2, kernel_size=kernel_size, dilation=1)
        # self.b = nn.Parameter(torch.tensor(0.5))  # Initial value of 'b'


    def forward(self, x, cur_gt):

        x1 = self.conv1(x)
        x = self.tanh1(x1)
        x = self.conv2(x)
        x = self.tanh2(x)
        x = self.conv3(x)
        x = self.tanh3(x)
        x2 = self.conv4(x)
        x = x2+x1
        x = self.tanh4(x)
        x = self.conv5(x)
        x = self.tanh5(x)
        x = self.conv6(x)
        x = self.tanh6(x)
        x = self.conv7(x)
        x = self.tanh7(x)
        x = self.conv8(x)
        x = x1+x2+x

        means = x[:,0,:]
        log_var = x[:,1,:]
        stds = torch.exp(0.5 *log_var)
        # stds = torch.ones_like(means)*self.b
        return means, stds
    
    def calc_model_likelihood(self, expected_means, expected_stds, wav_tensor, verbose=False):
        wav_tensor = wav_tensor.squeeze(axis=1)[:,self.kernel_size+1:]
        means_=expected_means.squeeze(axis=1)[:,self.kernel_size:-1]
        stds_ = expected_stds.squeeze(axis=1)[:,self.kernel_size:-1]
        # print(wav_tensor.shape)
        # print(means_.shape)

        exp_all = -(1/2)*((torch.square(wav_tensor-means_)/torch.square(stds_)))
        param_all = 1/(np.sqrt(2*np.pi)*stds_)
        model_likelihood1 = torch.sum(torch.log(param_all), axis=-1) 
        model_likelihood2 = torch.sum(exp_all, axis=-1) 

        if verbose:
            print("model_likelihood1: ", model_likelihood1)
            print("model_likelihood2: ", model_likelihood2)
        likelihood = model_likelihood1 + model_likelihood2
        return likelihood.mean()
    
    def casual_loss(self, expected_means, expected_stds, wav_tensor):
        model_likelihood = self.calc_model_likelihood(expected_means, expected_stds, wav_tensor)
        return -model_likelihood   
    





class GatedResidualBlock(nn.Module):
    """
    One block of WaveNet-like gating + skip + residual connection.

    in_channels:  number of input channels
    residual_channels: the internal channels for residual computation
    skip_channels:     channels to be added to the skip-connection path
    dilation:          convolution dilation
    kernel_size:       convolution kernel size
    """
    def __init__(
        self,
        in_channels,
        residual_channels,
        skip_channels,
        dilation,
        kernel_size
    ):
        super().__init__()
        # 1x1 to transform from in_channels -> residual_channels
        self.conv_in = nn.Conv1d(in_channels, residual_channels, kernel_size=1)

        # Gated convolution
        self.conv_filter = CausalConv1dClassS(
            residual_channels,
            residual_channels,
            kernel_size=kernel_size,
            dilation=dilation
        )
        self.conv_gate = CausalConv1dClassS(
            residual_channels,
            residual_channels,
            kernel_size=kernel_size,
            dilation=dilation
        )

        # 1x1 convolution for skip connection
        self.conv_skip = nn.Conv1d(residual_channels, skip_channels, kernel_size=1)
        # 1x1 convolution for residual connection
        self.conv_out = nn.Conv1d(residual_channels, in_channels, kernel_size=1)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Project up to residual_channels
        residual = self.conv_in(x)

        # Gated activation
        filter_out = self.tanh(self.conv_filter(residual))
        gate_out   = self.sigmoid(self.conv_gate(residual))
        gated = filter_out * gate_out

        # Skip-connection
        skip = self.conv_skip(gated)

        # Residual update (project back to in_channels)
        out = self.conv_out(gated) + x

        return out, skip



class WaveNetCausalModel(nn.Module):
    def __init__(
        self,
        kernel_size=9,
        residual_channels=16,
        skip_channels=16,
        dilation_depth=4,
        num_stacks=2
    ):
        """
        Args:
            kernel_size (int): size of the convolution kernel
            residual_channels (int): the number of channels in the residual/gated block
            skip_channels (int): the number of skip-connection channels
            dilation_depth (int): how many layers per stack (dilation = 1,2,4,...2^(d-1))
            num_stacks (int): how many times we repeat the stack of dilated convolutions
        """
        super().__init__()
        self.kernel_size = kernel_size

        # The input has dimension 1 (single waveform channel)
        # The output has dimension 2 (mu, log_var)
        in_channels = 1
        out_channels = 2

        self.blocks = nn.ModuleList()
        self.skip_channels = skip_channels

        # Create multiple stacks of layers with exponentially increasing dilation
        # e.g. for dilation_depth=4 we get dilations [1, 2, 4, 8].
        # We repeat that 'num_stacks' times.
        for _ in range(num_stacks):
            for i in range(dilation_depth):
                dilation = 2 ** i
                block = GatedResidualBlock(
                    in_channels=in_channels,
                    residual_channels=residual_channels,
                    skip_channels=skip_channels,
                    dilation=dilation,
                    kernel_size=kernel_size
                )
                self.blocks.append(block)

        # Final skip-output projection:
        # We'll accumulate all skip connections from each block
        self.final_conv1 = nn.Conv1d(skip_channels, skip_channels, kernel_size=1)
        self.final_conv2 = nn.Conv1d(skip_channels, out_channels, kernel_size=1)

    def forward(self, x,cur_gt):
        """
        x: shape (B, 1, T)
        Returns:
          means:   shape (B, T)      [the mean of the distribution]
          stds:    shape (B, T)      [the std of the distribution]
        """
        skip_connections = 0
        out = x
        for block in self.blocks:
            out, skip = block(out)
            skip_connections = skip_connections + skip  # accumulate skip

        # Pass through final skip conv
        skip = self.final_conv1(skip_connections)
        skip = nn.ReLU()(skip)
        output = self.final_conv2(skip)

        # output has shape (B, 2, T)
        means = output[:, 0, :]
        log_var = output[:, 1, :]

        # std = exp(0.5 * log_var)
        stds = torch.exp(0.5 * log_var)

        return means, stds

    def calc_model_likelihood(self, expected_means, expected_stds, wav_tensor, verbose=False):
        wav_tensor = wav_tensor.squeeze(axis=1)[:,self.kernel_size+1:]
        means_=expected_means.squeeze(axis=1)[:,self.kernel_size:-1]
        stds_ = expected_stds.squeeze(axis=1)[:,self.kernel_size:-1]
        # print(wav_tensor.shape)
        # print(means_.shape)

        exp_all = -(1/2)*((torch.square(wav_tensor-means_)/torch.square(stds_)))
        param_all = 1/(np.sqrt(2*np.pi)*stds_)
        model_likelihood1 = torch.sum(torch.log(param_all), axis=-1) 
        model_likelihood2 = torch.sum(exp_all, axis=-1) 

        if verbose:
            print("model_likelihood1: ", model_likelihood1)
            print("model_likelihood2: ", model_likelihood2)
        likelihood = model_likelihood1 + model_likelihood2
        return likelihood.mean()

    def casual_loss(self, expected_means, expected_stds, wav_tensor):
        """
        The negative of the average log-likelihood (so we can do a gradient descent).
        """
        model_likelihood = self.calc_model_likelihood(expected_means, expected_stds, wav_tensor)
        return -model_likelihood



# def create_noisy_files(root):
def plot_loss(loss_array,loss_test_array,j, netname,imgpath):
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(loss_array[j])
    axs[0].set_title(f'loss_array i={j}')
    axs[1].plot(loss_test_array[j])
    axs[1].set_title(f'loss_test_array i={j} nn {netname}')
    fig.savefig(imgpath)
    print(f"Plot saved to {imgpath}")
    plt.close(fig)


import torch.multiprocessing as mp

def train_nets_process(network, train_full_tensors, test_full_tensors, device, idxes,trial, epochs=6000,batch_size=16,g_t=None,exp_root=None):
    if network=="NetworkNoise2":
        nets = [NetworkNoise2() for i in range(len(idxes))]
    elif network=="NetworkNoise3":
        nets = [NetworkNoise3() for i in range(len(idxes))]
    elif network=="NetworkNoise4":
        nets = [NetworkNoise4() for i in range(len(idxes))]
    elif network=="WaveNetCausalModel":
        nets = [WaveNetCausalModel() for i in range(len(idxes))]
    else:
        print("network unknown")
        raise Exception
    # elif trial==1:
    #     nets = [Network2() for i in range(len(idxes))]
    # elif trial==2:
    #     nets = [Network3() for i in range(len(idxes))]
    # elif trial==3:
    #     nets = [Network4() for i in range(len(idxes))]
    
    print("idxes[0]:",idxes[0])
    if idxes[0] == 0:
        quarter_idx=0
    elif idxes[0] == 50:
        quarter_idx=1
    elif idxes[0] == 100:
        quarter_idx=2
    elif idxes[0] == 150:
        quarter_idx=3
    else:
        print ("no identifiesd quarter")
        raise Exception
    
    cur_epochs = epochs
    

    loss_array = {}
    loss_test_array = {}

    net_counter=-1
    for i in idxes:
        
        cur_white_noise_diffusion = torch.normal(0,1,train_full_tensors[:,0,:].shape)
        cur_train_full_tensors = train_full_tensors[:,0,:]+cur_white_noise_diffusion*g_t[i]
        
        cur_white_noise_diffusion = torch.normal(0,1,test_full_tensors[:,0,:].shape)
        cur_test_full_tensors = test_full_tensors[:,0,:]+cur_white_noise_diffusion*g_t[i]
        
        #Create TensorDatasets
        dataset_size_ = train_full_tensors.shape[0]
        train_dataset = BatchNoiseDataset(cur_train_full_tensors.reshape(dataset_size_,1,-1),g_t[i])
        test_dataset = BatchNoiseDataset(cur_test_full_tensors.reshape(dataset_size_,1,-1),g_t[i])

        #Create DataLoaders
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) #todo: numbers
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        
        net_counter+=1
        model = nets[net_counter]
        model.to(device)
        model.train()

        optimizer = optim.Adam(model.parameters())
        
        cur_epochs = epochs

        for epoch in range(cur_epochs):
            running_loss = 0.0
            for batch_idx, (batch_tensor, gt_tensor) in enumerate(train_loader):
                optimizer.zero_grad()
                batch_tensor = batch_tensor.to(device, dtype=torch.float)
                gt_tensor = gt_tensor.to(device, dtype=torch.float)
                
                means, stds = model(batch_tensor, gt_tensor)
                loss = model.casual_loss( means, stds, wav_tensor=batch_tensor).mean()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
            if epoch%1==0:
                with torch.no_grad():
                    for batch_idx, (test_inputs, gt_test) in enumerate(test_loader):
                        test_inputs = test_inputs.to(device, dtype=torch.float)
                        gt_test = gt_test.to(device, dtype=torch.float)
                        meanst, stdst = model(test_inputs, gt_test)
                loss_t = model.casual_loss( meanst, stdst, wav_tensor=test_inputs).mean()
                if i in loss_test_array:
                    loss_test_array[i].append(float(loss_t))
                else:
                    loss_test_array[i] = [float(loss_t)]
            
            if i in loss_array:
                loss_array[i].append(float(loss))
            else:
                loss_array[i] = [float(loss)]
        nets[net_counter].parameters = model.parameters
        print(f"Model {i} Epoch {epoch+1}/{epochs}, Loss: {running_loss}")
        
        if i in [0,20,50,100,150]:
            if exp_root == None:
                print("dont have path for graph")
            else:
                father_path = Path(exp_root)/ f"analysis/graphs/"
                print(father_path)
                father_path.mkdir(parents=True, exist_ok=True)
                imgpath = father_path/ f"loss_net_{network}_i{i}.jpg"
                print("plot_loss")
                plot_loss(loss_array,loss_test_array,i, network,imgpath)
    
    return nets, loss_array, loss_test_array, quarter_idx


def train_nets_parralel(network,train_dataset, test_dataset,trial=0, epochs=100,num_nets=200,batch_size=16,g_t=None,exp_root=None):
    results = []
    gpu_num=4
    idxes_all = [list(range(0,50)),list(range(50,100)),list(range(100,150)),list(range(150,200))]
    # idxes_all = [list(range(0,100)),list(range(100,200))]
    
    devices = [f'cuda:{i}' for i in range(gpu_num)]
    with mp.get_context('spawn').Pool(processes=4) as pool:
        args = [(network, train_dataset, test_dataset,devices[i % gpu_num], idxes, trial,epochs,batch_size,g_t,exp_root) for i, idxes in enumerate(idxes_all)]
        results = pool.starmap(train_nets_process, args)

    loss_array= results[0][1]
    loss_test_array = results[0][2]
    nets = results[0][0]
    for i in range(1,gpu_num):
        nets.extend(results[i][0])
        loss_array.update(results[i][1])
        loss_test_array.update(results[i][2])
    # idxes_all = [list(range(0,200))]

    # nets, loss_array, loss_test_array, quarter_idx = train_nets_process(train_dataset, test_dataset,"cuda:1", idxes_all, trial,epochs=epochs)
    return nets,loss_array,loss_test_array




def train_noisemodel(root, network="NetworkNoise2",epochs=60, dataset_size=128*8, n_samples=640000,batch_size=16,g_t=None,num_steps=200):
    print("starting ----")

    print(root)
    
    with open(Path(root)/'5f_snrs.pickle', 'rb') as handle:
        snr_df = pickle.load(handle)
    print(snr_df)
    
    failed = []
    
    snr_df2 = snr_df
    # snr_df2 = snr_df[(snr_df["dir"]=="b") | (snr_df["dir"]=="a") | (snr_df["dir"]=="c")]
    # for trial in [3,1,2]:
    for i in snr_df2.index:
        # if i==0:
        #     continue
        for trial in [0]:
            
            
                
            train_idx = i
            cur_snr = snr_df["snr"][train_idx]
            if cur_snr != 5.0:
                print(cur_snr, "continue")
                continue
            
            noise_idx = snr_df["noise_idx"][train_idx]
            print("noise_idx", noise_idx)
            # if noise_idx!="Babble":
            #     print("noise_idx", noise_idx)
            #     continue
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
            
            if noise_idx=="1":
                ar_coefs = [0.9]
            elif noise_idx=="2":
                ar_coefs = [0.6,-0.1, 0.2]
                order=3
            elif noise_idx=="3":
                ar_coefs = [-0.9]
            else:
                print("unknown noise")
                continue

            
            # n_samples = 64000
            ar_noise_batch = create_ar_noise_batch( batch_size=dataset_size, n_samples=n_samples, ar_coefs=ar_coefs, order=len(ar_coefs))
            stop = math.floor(n_samples/sr-1)
            # stop = 2
            train_ar = ar_noise_batch[:,  int(0*sr):int(stop*sr)] *cur_noise_scaling
            test_ar = ar_noise_batch[:,  int(stop*sr):int((stop+1)*sr)]*cur_noise_scaling
            
            # sr = 16000
            train_tensor = torch.tensor(train_ar, dtype=torch.float32)#.view(1,1,-1)
            test_tensor = torch.tensor(test_ar, dtype=torch.float32)#.view(1,1,-1)
            
            
            print("creating tensors")

            train_full_tensors = train_tensor.squeeze().view(dataset_size,1,-1)
            test_full_tensors = test_tensor.reshape(dataset_size,200,-1)


            
            print("starting training")
            nets,loss_array,loss_test_array = train_nets_parralel(network,train_full_tensors, test_full_tensors,trial,epochs=epochs,num_nets=200,batch_size=batch_size,g_t=g_t)
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







def calc_vad(f, verbose=False):
    test_file=f
    fs,s = read_wav(test_file)
    win_len = int(fs*0.025)
    hop_len = int(fs*0.010)
    sframes = enframe(s,win_len,hop_len) # rows: frame index, cols: each frame
    if verbose:
        plot_this(compute_log_nrg(sframes))

    # percent_high_nrg is the VAD context ratio. It helps smooth the
    # output VAD decisions. Higher values are more strict.
    percent_high_nrg = 0.5

    vad = nrg_vad(sframes,percent_high_nrg)

    if verbose:
        plot_these(deframe(vad,win_len,hop_len),s)
    return deframe(vad,win_len,hop_len) 


def create_ar_noise(n_samples, ar_coefs,order=1, dtype=torch.float32):

    mu=0
    sigma=1
    white_noise = np.random.normal(mu, sigma, n_samples + order)

    # Generate AR noise
    ar_noise = np.zeros_like(white_noise)
    for i in range(order, n_samples + order):
        ar_noise[i] = np.dot(ar_coefs, ar_noise[i - order:i]) + white_noise[i]
    ar_noise = ar_noise[order:]  # Discard initial transient
    return torch.from_numpy(ar_noise).to(dtype).view(1,-1)



def create_wav_dirs(exp_root):
    cleans_root = os.path.join(exp_root,"cleans")
    clean_files =  glob(cleans_root+"/*")
    names=[]
    for f in clean_files:
        if f.endswith(".wav"):
            try:
                name = os.path.basename(f).split("_")[0]
                print(name)
                names.append(name)
            except:
                print("no name in filename")
            dir_path = os.path.join(exp_root,name)
            if os.path.exists(dir_path):
                print(name + " exist")
                raise Exception
            else:
                os.mkdir(dir_path)
                path_ = Path(dir_path)
                clean_path = path_/"clean_wav"
                os.mkdir(clean_path)
                os.mkdir(path_/"noisy_wav")
                os.mkdir(path_/"noises")
                namewav = name+".wav"
                shutil.copyfile(f, clean_path/namewav)
    return names


def create_noises(exp_root):
    noisees_rooot = Path(exp_root)/"noises"
    n_samples = 16000*6
    sample_rate = 16000
    ar_coefs = [0.9]
    noise1 = create_ar_noise(n_samples, ar_coefs,order=1, dtype=torch.float32)

    ar_coefs = [0.6,-0.1, 0.2]
    noise2 = create_ar_noise(n_samples, ar_coefs,order=3, dtype=torch.float32)

    ar_coefs = [-0.9]
    noise3 = create_ar_noise(n_samples, ar_coefs,order=1, dtype=torch.float32)


    tar_noisypath = (noisees_rooot)/"1.wav"
    torchaudio.save(tar_noisypath, noise1, sample_rate,encoding="PCM_F")

    tar_noisypath = (noisees_rooot)/"2.wav"
    torchaudio.save(tar_noisypath, noise2, sample_rate,encoding="PCM_F")

    tar_noisypath = (noisees_rooot)/"3.wav"
    torchaudio.save(tar_noisypath, noise3, sample_rate,encoding="PCM_F")
    noises = ["1","2","3"]
    return noises


def noise_waves(exp_root, snr_array=[5]):
    noiseroot = os.path.join(exp_root,"noises/")
    noise_files =  glob(noiseroot+"/*")
    print(noise_files)

    dirs_ =  glob(exp_root+"/*")

    idx = 0
    sample_rate=16000
    dict_ = {"snr":[],"noise_scaling": [], "noise_idx":[], "dir": [] }
    df_snr = pd.DataFrame(data=dict_)

    for dir_ in dirs_:
        if os.path.basename(dir_)=="noises" or os.path.basename(dir_) =='5f_snrs.pickle' or os.path.basename(dir_) =='cleans':
            continue
        print("dir_", dir_)
        print(glob(dir_+"/clean_wav/*"))
        speech_file_ = glob(dir_+"/clean_wav/*")[0]
        
        speech, sr = torchaudio.load(speech_file_)
        speech_file_ = speech_file_.replace("WAV", "wav")

        if sr != sample_rate:
            speech = F.resample(speech, sr, sample_rate)
        speech, sr = torchaudio.load(speech_file_)


        vaded_signal = calc_vad(speech_file_)[0:speech.shape[1],:]
        vaded_signal_torch = (speech[0][vaded_signal.T[0]>0])
        vaded_signal_torch = torch.unsqueeze(vaded_signal_torch, dim=0)
        clean_power = float( 1 / vaded_signal_torch.shape[1] * torch.sum(vaded_signal_torch**2))
        print("clean_power: ", clean_power)
        simple_power =  1 / speech.shape[1] * torch.sum(speech**2)
        print("simple_power: ", simple_power)
        # snr_array = [5]

        for i, snr in enumerate(snr_array):
            snr=int(snr)
            
            for j,noise_f in enumerate(noise_files):
                idx +=1        
                noise_idx = os.path.basename(noise_f).replace(".wav","")
                noise, sr = torchaudio.load(noise_f)
                noise = noise[:,(5*sr):(speech.shape[1]+(5*sr))] ###leave 5 sec for training
                speech2 = speech[:,:noise.shape[1]]
                
                noise_power = 1 / noise.shape[1] * torch.sum(noise**2)
                # noise_power = clean_power_noise
                speech_power = clean_power
                noise_power_target = speech_power * np.power(10, -snr / 10)
                noise_scaling = np.sqrt(noise_power_target / noise_power)
                print("speech.shape: ",speech2.shape)
                print("noise.shape: ",noise.shape)

                lossy_speech = speech2 + noise_scaling * noise
                print("noise_scaling: ", noise_scaling)
                
                noise_var = noise_scaling**2
                
                new_noise = noise_scaling * noise
                new_noise_power = float(1 / new_noise.shape[1] * torch.sum(new_noise**2))
                
                y_power =  1 / lossy_speech.shape[1] * torch.sum(lossy_speech**2)
                
                vaded_signal = calc_vad(speech_file_)[0:lossy_speech.shape[1],:]
                vaded_lossy_speech_torch = (lossy_speech[0][vaded_signal.T[0]>0])
                vaded_lossy_speech_torch = torch.unsqueeze(vaded_lossy_speech_torch, dim=0)
                yclean_power = float( 1 / vaded_lossy_speech_torch.shape[1] * torch.sum(vaded_lossy_speech_torch**2))
                # print("clean_power: ", yclean_power)
                print("snr: ", snr)
                df_snr.at[idx, "snr"] = snr
                df_snr.at[idx, "noise_scaling"] = float(noise_scaling)
                df_snr.at[idx, "dir"] = dir_[-1]
                df_snr.at[idx, "noise_idx"] = noise_idx
                
                filename = "noise{}_digits_snr{}_power{}_var{}.wav".format(noise_idx,snr, new_noise_power, noise_var)

                tarpath = os.path.join(os.path.join(dir_, "clean_wav"), filename)
                print("tarpath: ", tarpath)
                torchaudio.save(tarpath, speech2, sample_rate,encoding="PCM_F")
                df_snr.at[idx, "clean_wav"] = tarpath

                noisy_root = os.path.join(dir_, "noisy_wav")
                tar_noisypath =  os.path.join(noisy_root, filename)
                if not os.path.exists(noisy_root):
                    os.mkdir(noisy_root)
                torchaudio.save(tar_noisypath, lossy_speech, sample_rate,encoding="PCM_F")
                df_snr.at[idx, "noisy_wav"] = tar_noisypath
                
                noise, sr = torchaudio.load(noise_f)
                scaled_noise = noise_scaling * noise
                noises_root = os.path.join(dir_, "noises")
                print("noise_scaling: ", noise_scaling)
                tar_noise =  os.path.join(noises_root, filename)
                if not os.path.exists(noises_root):
                    os.mkdir(noises_root)
                df_snr.at[idx, "noise_path"] = tar_noise
                # if noise_idx=="1" and  dir_[-1] == "b":
                #     raise Exception

                torchaudio.save(tar_noise, scaled_noise, sample_rate,encoding="PCM_F")
        

    df_snr = df_snr.sort_values(by=['dir'])
    print(df_snr)

    with open(os.path.join(exp_root,'5f_snrs.pickle'), 'wb') as handle:
        pickle.dump(df_snr, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return snr_array


if __name__ == '__main__':
    with open("exps_configs/m_ar_short_much.yaml", "r") as f:
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
    
    betas=get_named_beta_schedule(scheduler_type, num_steps)
    alphas = 1.0 - betas
    alphas_cumprod =  torch.from_numpy(np.cumprod(alphas, axis=0))
    g_t = torch.sqrt((1-alphas_cumprod)/(alphas_cumprod))
    
    
    names = create_wav_dirs(exp_root)
    noises_names = create_noises(exp_root)
    
    snr_array = noise_waves(exp_root, snr_array=snr_array)

    train_noisemodel(exp_root, network=network,epochs=epochs, dataset_size=dataset_size, n_samples=n_samples,batch_size=batch_size,g_t=g_t,num_steps=num_steps)#128*8

    print("---run_exp---")
    run_exp(exp_root, dirnames=names, cuda_idx="1",s_array=s_array)
    
    storm_root = str(Path(exp_root)/"storm")
    run_storm(exp_root,storm_root)
    
    analyze_exp(exp_root,noises_names,snr_array,names)


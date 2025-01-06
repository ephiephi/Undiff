import collections
import datetime
import glob
import pathlib
import pandas as pd
import seaborn as sns
# import tensorflow as tf
import os
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
from IPython import display
from IPython import display
from matplotlib import pyplot as plt
from typing import Optional
from pathlib import Path
import numpy as np
import torch.utils.data
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm

from torch.autograd import Variable
import torch
import numpy as np
from scipy import signal

def plot_specgram(waveform, sample_rate):
    f, t, Sxx = signal.spectrogram(waveform, sample_rate)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Intensity [dB]')
    plt.show()
    
    
    


import torchaudio
# noise_whole = r"/data/ephraim/datasets/known_noise/undiff/exp_ar_j_real_2sec_divided/noises/machine.wav"
# noise_whole, sr = torchaudio.load(noise_whole)

import numpy as np
import torch


def normalize_tensor(tensor):
    """
    Normalize the input tensor using Z-score normalization.
    
    Args:
        tensor (torch.Tensor): Input tensor to be normalized.
        
    Returns:
        normalized_tensor (torch.Tensor): The normalized tensor.
        mean (float): Mean of the input tensor.
        std (float): Standard deviation of the input tensor.
    """
    mean = tensor.mean()
    std = tensor.std()
    
    # Avoid division by zero
    std = std if std > 0 else 1e-6
    
    normalized_tensor = (tensor - mean) / std
    return normalized_tensor, mean, std


def denormalize_tensor(normalized_tensor, mean, std):
    """
    Recover the original tensor from the normalized tensor.
    
    Args:
        normalized_tensor (torch.Tensor): Normalized tensor.
        mean (float): Mean used for normalization.
        std (float): Standard deviation used for normalization.
    
    Returns:
        torch.Tensor: The original tensor.
    """
    return normalized_tensor * std + mean


def create_ar_noise(n_samples, ar_coefs,order=1, dtype=torch.float32):
    
    mu=0
    sigma=1
    # Generate white noise
    white_noise = np.random.normal(mu, sigma, n_samples + order)

    # Initialize AR coefficients randomly
    # ar_coefs = np.random.uniform(AR_COEF_MIN,AR_COEF_MAX, order)

    # Generate AR noise
    ar_noise = np.zeros_like(white_noise)
    for i in range(order, n_samples + order):
        ar_noise[i] = np.dot(ar_coefs, ar_noise[i - order:i]) + white_noise[i]
    ar_noise = ar_noise[order:]  # Discard initial transient
    return torch.from_numpy(ar_noise).to(dtype).view(1,-1)
# ar_noise1 = create_ar_noise(16000, 0.9, 0.9)


sr=16000
# noise_whole = create_ar_noise(10000000, 0.9, 0.9, 1)
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

#### Display tools
def plot_this(s,title=''):
    """

    """
    import pylab
    s = s.squeeze()
    if s.ndim ==1:
        pylab.plot(s)
    else:
        pylab.imshow(s,aspect='auto')
        pylab.title(title)
    pylab.show()
    

dataset_size=128*2
n_samples =640000
ar_noise_batch = create_ar_noise_batch(batch_size=dataset_size, n_samples=n_samples, ar_coefs=[0.9], order=1)
ar_noise_batch, mean_all, std_all = normalize_tensor(ar_noise_batch)

sr = 16000
new_sample_rate = 16000
# resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=new_sample_rate)
# noise_whole = resampler(noise_whole)
# print("old sr:", sr)
# print("new sr:", new_sample_rate)

print(ar_noise_batch.shape)
stop = 35
noise_sample1 = ar_noise_batch[0,  int(0*sr):int(stop*sr)]
noise_sample2 = ar_noise_batch[0,  int(stop*sr):int((stop+1)*sr)]
train_ar = ar_noise_batch[:,  int(0*sr):int(stop*sr)]
test_ar = ar_noise_batch[:,  int(stop*sr):int((stop+1)*sr)]
print(noise_sample1.shape)
print(noise_sample2.shape)


import torch
from torch.utils.data import DataLoader, TensorDataset


# Create datasets and data loaders
train_tensor = torch.tensor(train_ar, dtype=torch.float32)#.view(1,1,-1)
test_tensor = torch.tensor(test_ar, dtype=torch.float32)#.view(1,1,-1)
train_tensor.shape



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

betas=get_named_beta_schedule("linear", 200)

alphas = 1.0 - betas
alphas_cumprod =  torch.from_numpy(np.cumprod(alphas, axis=0))
white_noise_diffusion = torch.normal(0,1,train_tensor.shape)
white_noise_diffusion2 = torch.normal(0,1,test_tensor.shape)
g_t = torch.sqrt((1-alphas_cumprod)/(alphas_cumprod))
t = np.random.randint(1,200,1)
cur_g_t = g_t[t]
real_Vtrain = train_tensor + white_noise_diffusion*g_t[t]
real_Vtest = test_tensor + white_noise_diffusion2*g_t[t]
# train_tensor = real_Vtrain
# test_tensor = real_Vtest
real_Vtrain.shape



# train_full_tensors = train_tensor.squeeze().repeat(200,1,1).view(dataset_size,200,-1)
train_full_tensors = train_tensor.squeeze().view(dataset_size,1,-1)



# train_full_tensors = train_tensor.reshape(dataset_size,200,-1)

for i in [0]:
    cur_white_noise_diffusion = torch.normal(0,1,train_full_tensors[:,i,:].shape)
    train_full_tensors = train_full_tensors[:,i,:] #+cur_white_noise_diffusion*g_t[i]

# test_full_tensors = test_tensor.squeeze().repeat(200,1).view(200,1,-1)
test_full_tensors = test_tensor.squeeze().view(dataset_size,1,-1)

for i in [0]:
    cur_white_noise_diffusion = torch.normal(0,1,test_full_tensors[:,i,:].shape)
    test_full_tensors = test_full_tensors[:,i,:] #+cur_white_noise_diffusion*g_t[i]

print(train_full_tensors.shape)
print(test_full_tensors.shape)


import os
import pandas as pd
from torch.utils.data import Dataset


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


def calc_stft(tensor):

    # Parameters
    sample_rate = 16000  # Sample rate in Hz
    n_fft = 512  # Number of FFT points
    win_length = n_fft  # Window length
    hop_length = int(win_length/2)  # Number of samples between frames
    window = torch.hann_window(win_length)  # Window function

    signal_ = tensor.view(-1)
    duration = max(tensor.shape)

    stft = torch.stft(signal_, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)
    return stft, duration, sample_rate

def plot_waveform(waveform, sr, title="Waveform", ax=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    if ax is None:
        _, ax = plt.subplots(num_channels, 1)
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)


def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")


def plot_fbank(fbank, title=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Filter bank")
    axs.imshow(fbank, aspect="auto")
    axs.set_ylabel("frequency bin")
    axs.set_xlabel("mel bin")
    
from scipy.signal import firwin
def fir_filter(tensor, high=False, cutoff=6000,num_taps=101,device="cpu",sample_rate=16000):
    cutoff_freq = cutoff  # Cutoff frequency for the high-pass filter in Hz
    num_taps = num_taps  # Filter order (number of filter coefficients)
    if high:
        pass_zero = False
    else:
        pass_zero = True
    coefficients = firwin(num_taps, cutoff=cutoff_freq, pass_zero=pass_zero, fs=sample_rate)

    # Convert filter coefficients to a PyTorch tensor
    coefficients = torch.tensor(coefficients, dtype=torch.float32).to(device)

    pad_length = (num_taps - 1) // 2
    signal_padded = torch.nn.functional.pad(tensor.view(1, 1, -1), (pad_length, pad_length), mode='constant').to(device)

    # Apply the FIR filter to the signal using convolution
    filtered_signal = torch.nn.functional.conv1d(signal_padded.view(1, 1, -1), coefficients.view(1, 1, -1))
    return filtered_signal


from torch import nn



loss_array = {}
loss_test_array = {}

def train(nets, steps,train_loader,test_loader ):
    device="cuda:3"
    print("start ")
    for i in steps:
        model = nets[i]
        min_test_loss = 1000000000
        # if i ==1:
        #     break
        #Create TensorDatasets
        
        model.to(device)
        model.train()
        ii=0
        optimizer = optim.Adam(model.parameters())
        for epoch in tqdm(range(epochs)):
            running_loss = 0.0
            for batch_idx, (batch_tensor, gt_tensor) in enumerate(train_loader):
                ii+=1
                # data = data[0]
                # for data in train_loader:
                optimizer.zero_grad()
                batch_tensor = batch_tensor.to(device, dtype=torch.float)
                gt_tensor = gt_tensor.to(device, dtype=torch.float)
                
                means, stds = model(batch_tensor,gt_tensor )


                # print(model.casual_loss)
                loss = model.casual_loss( means, stds, wav_tensor=batch_tensor).mean() ##.mean()
                print("loss ",loss)
                # print("loss",loss)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                    
            if epoch%50==0:
                print(epoch)
            if epoch%1==0 or i > 90:
                with torch.no_grad():
                    for batch_idx, (test_inputs, gt_test) in enumerate(test_loader):
                        # test_inputs, gt_test = test_dataset.__getitem__(i)
                        # test_inputs = fir_filter(test_inputs,high=True)
                        test_inputs = test_inputs.to(device, dtype=torch.float)
                        gt_test = gt_test.to(device, dtype=torch.float)
                        meanst, stdst = model(test_inputs, gt_test)
                loss_t = model.casual_loss( meanst, stdst, wav_tensor=test_inputs).mean()
                if i in loss_test_array:
                    loss_test_array[i].append(float(loss_t))
                else:
                    loss_test_array[i] = [float(loss_t)]
                if loss_t<=min_test_loss or epoch<10:
                    min_test_loss = loss_t
                # else:
                #     break
            

            if i in loss_array:
                loss_array[i].append(float(loss))
            else:
                loss_array[i] = [float(loss)]
        nets[i].parameters = model.parameters
        print(f"Model {i} Epoch {epoch+1}/{epochs}, Loss: {running_loss}, test Loss: {loss_t}")
    print(ii)
    return nets, loss_array, loss_test_array



class SimpleAR(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.01))  # Initial value of 'a'
        self.b = nn.Parameter(torch.tensor(0.01))  # Initial value of 'b'

    def forward(self, x, cur_gt):
        means = torch.zeros_like(x)
        stds = torch.zeros_like(x)
        for i in range(x.shape[-1]):
            if i==1:
                continue
            means[:,:,i] = self.a* x[:,:,i-1] 
            stds[:,:,i] = self.b 
        return means, stds
    
    def forward_once(self, x, cur_gt):
        means = self.a* x[:,:,i] 
        stds = self.b 
        return means, stds
    
    def calc_model_likelihood(self, expected_means, expected_stds, wav_tensor, verbose=False):
        wav_tensor = wav_tensor.squeeze(axis=1)[:,2:]
        means_=expected_means.squeeze(axis=1)[:,2:]
        stds_ = expected_stds.squeeze(axis=1)[:,2:]


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
    





import matplotlib.pyplot as plt
j=199
def plot_loss(loss_array,loss_test_array,j):
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(loss_array[j])
    axs[0].set_title('loss_array')
    axs[1].plot(loss_test_array[j])
    axs[1].set_title('loss_test_array')
    plt.show()
    
    


import torch.optim as optim
nets_g = [SimpleAR()]
epochs = 1

loss_array = {}
loss_test_array = {}
i=0
# train_full_tensors_n, mean, std = normalize_tensor(train_full_tensors)
# test_full_tensors_n, mean, std = normalize_tensor(test_full_tensors)
train_dataset = BatchNoiseDataset(train_full_tensors[:,:20].reshape(dataset_size,1,-1),g_t[i])
test_dataset = BatchNoiseDataset(test_full_tensors[:,:20].reshape(dataset_size,1,-1),g_t[i])

#Create DataLoaders
train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True) #todo: numbers
test_loader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=False)


nets_g,loss_array_high,loss_test_array_high = train(steps=[0],nets=nets_g,train_loader=train_loader,test_loader=test_loader)

plot_loss(loss_array_high,loss_test_array_high,0)
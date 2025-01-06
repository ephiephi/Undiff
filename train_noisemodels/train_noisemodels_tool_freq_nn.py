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
        


from torch.nn.modules.utils import _pair
from torch import nn
import torch.nn.functional as F

class CausalConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, dilation=1, groups=1, bias=True):
        stride = _pair(stride)
        dilation = (dilation,1)
        # print("dilation:", dilation)
        if padding is None:
            padding = int((kernel_size[1] -1) * dilation[1] +1) 
        else:
           padding = padding * 2
        # print("padding:",padding)
        self._pad= (padding)
        self._pad_h =   int(np.floor(kernel_size[0]/2))
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias)
    def forward(self, inputs):
        # print("self._pad:",self._pad)
        inputs = F.pad(inputs, (self._pad, 0,self._pad_h , self._pad_h))
        output = super().forward(inputs)
        if self._pad != 0:
            output = output[:, :, :-1]
        return output
    
    
class NetworkFreq(nn.Module):
    def __init__(self, kernel_size=(5,5)):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad_h = int(np.floor(kernel_size[0]/2))
        # BCHW - https://discuss.pytorch.org/t/applying-separate-convolutions-to-each-row-of-input/152597
        self.conv1 = CausalConv2d(257, 257*2, kernel_size=kernel_size, groups=257)

    def forward(self, x, cur_gt):
        
        assert len(x.shape) ==4
        B = x.shape[0]
        C = x.shape[1]
        H = x.shape[2]
        W = x.shape[3]
        # print("B,C,H:", B,C,H)
        # print("input.shape: ", input.shape)
        x = self.conv1(x.transpose(1, 2).reshape (H * C ,B, -1))
        x = x.reshape (B, H, C*2, W).transpose (1, 2)

        means = x[:,0,:,:]
        log_var = x[:,1,:,:]
        stds = torch.exp(0.5 *log_var)
        return means, stds
    
    def calc_model_likelihood(self, expected_means, expected_stds, wav_tensor, verbose=False):
        wav_tensor = wav_tensor.squeeze()
        # print("wav_tensor shape: ", wav_tensor.shape)
        means_=expected_means.squeeze()
        stds_ = expected_stds.squeeze()
        exp_all = -(1/2)*((torch.square(wav_tensor-means_)/torch.square(stds_)))
        param_all = 1/(np.sqrt(2*np.pi)*stds_)
        model_likelihood1 = torch.sum(torch.log(param_all)) #, axis=-1 
        model_likelihood2 = torch.sum(exp_all) #, axis=-1

        if verbose:
            print("model_likelihood1: ", model_likelihood1)
            print("model_likelihood2: ", model_likelihood2)
        return model_likelihood1 + model_likelihood2
    
    def casual_loss(self, expected_means, expected_stds, wav_tensor):
        model_likelihood = self.calc_model_likelihood(expected_means, expected_stds, wav_tensor)
        return -model_likelihood


# def create_noisy_files(root):
    
def create_dataset_real(noise_whole, sr):
    # noise_sample1 = noise_whole[:, 0:int(sr/2)]
    # noise_sample2 = noise_whole[:, int(sr/2):sr]
    # noise_sample1 = noise_whole[:, -int(sr/2):]
    # noise_sample2 = noise_whole[:, -sr:-int(sr/2)]
    
    # noise_sample1 = noise_whole[:, 0:int(5*sr)]
    # noise_sample2 = noise_whole[:, int(5*sr):int(10*sr)]
    
    # noise_sample1 = noise_whole[:, 0:int(5*sr)]
    # noise_sample2 = noise_whole[:,  int(5*sr):int(10*sr)]
    
    noise_sample1 = noise_whole[:, int(3*sr):int(5*sr)]
    noise_sample2 = noise_whole[:,  int(1*sr):int(3*sr)]

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

import torch.multiprocessing as mp

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


def train_nets(train_dataset, test_dataset,trial,epochs=1000,num_nets=200, device="cuda"):
    # if trial==0:
    #     nets = [Network() for i in range(len(idxes))]
    # elif trial==1:
    #     nets = [Network2() for i in range(len(idxes))]
    # elif trial==2:
    #     nets = [Network3() for i in range(len(idxes))]
    # elif trial==3:
    #     nets = [Network4() for i in range(len(idxes))]
    nets = [NetworkFreq() for i in range(num_nets)]
    loss_array = {}
    loss_test_array = {}

    net_counter=-1
    for i,model in enumerate(nets):
        net_counter+=1
        model = nets[net_counter]
        model.to(device)
        model.train()
        min_test_loss = 1000000000

        optimizer = optim.Adam(model.parameters(), lr=0.05)
        
        for epoch in range(epochs):
            running_loss = 0.0
            cur_train_tensor, gt_tensor = train_dataset.__getitem__(i)
            optimizer.zero_grad()
            
            stft, duration, sample_rate = calc_stft(cur_train_tensor)
            magnitude = torch.log(torch.abs(stft))
            
            batch_tensor = magnitude.view(1,1,stft.shape[0],stft.shape[1]).to(device, dtype=torch.float)
            # batch_tensor = batch_tensor.repeat(10,1,1,1)
            gt_tensor = gt_tensor.to(device, dtype=torch.float)
            # print("batch_tensor.shape:",batch_tensor.shape)
            # print()
            means, stds = model(batch_tensor, gt_tensor)


            loss = model.casual_loss( means, stds, wav_tensor=batch_tensor)
            # print("loss",loss)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
                
            
            if epoch%10==0:
                with torch.no_grad():
                    cur_test_inputs, gt_test = test_dataset.__getitem__(i)
                    stft_test, duration, sample_rate = calc_stft(cur_test_inputs)
                    magnitude_test =  torch.log(torch.abs(stft_test))
                    test_inputs = magnitude_test.view(1,1,stft.shape[0],stft.shape[1]).to(device, dtype=torch.float)
                    gt_test = gt_test.to(device, dtype=torch.float)
                    meanst, stdst = model(test_inputs, gt_test)
                loss_t = model.casual_loss( meanst, stdst, wav_tensor=test_inputs)
                if i in loss_test_array:
                    loss_test_array[i].append(float(loss_t))
                else:
                    loss_test_array[i] = [float(loss_t)]
                if loss_t<=min_test_loss:
                    min_test_loss = loss_t
                else:
                    break
            
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



if __name__ == "__main__":
    print("starting ----")
    root = "/data/ephraim/datasets/known_noise/undiff/exp_ar_j_real_freq_nn/"
    print(root)
    

    with open(Path(root)/'5f_snrs.pickle', 'rb') as handle:
        snr_df = pickle.load(handle)
    print(snr_df)
    
    failed = []

    
    # i=snr_df[snr_df["dir"]=="b"].index[2]
    snr_df2 = snr_df[(snr_df["dir"]=="b") | (snr_df["dir"]=="a") | (snr_df["dir"]=="c")]
    # for trial in [3,1,2]:
    for i in tqdm(snr_df2.index):
        # if i >1:
        #     break
        for trial in [0]:
            
        # for i in snr_df.index:
        #     print(i)
        # raise Exception
            
                
            train_idx = i
            cur_snr = snr_df["snr"][train_idx]
            if cur_snr != 5.0:
                print(cur_snr, "continue")
                continue
            
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
            nets,loss_array,loss_test_array = train_nets(train_dataset, test_dataset,trial,epochs=1000,num_nets=200)
                            
            params_dict = {"nets": nets, "train_dataset": None, "test_dataset": None,"ar_coefs":None, "loss_array":loss_array, "loss_test_array": loss_test_array, "ar_noise": noise_sample1, "noise_scaling": cur_noise_scaling, "snr": str(int(cur_snr)), "noise_name": noise_idx, "noise_path": noise_path}
            # params_dict = {"result": result}
            params_dict_debug = {"nets": nets, "train_dataset": train_dataset, "test_dataset": test_dataset,"ar_coefs":None, "loss_array":loss_array, "loss_test_array": loss_test_array, "ar_noise": noise_sample1, "noise_scaling": cur_noise_scaling, "snr": str(int(cur_snr)), "noise_name": noise_idx, "noise_path": noise_path}
            
            #save in name of 0
            pickle_path = cur_dir/(str(0)+"_"+"snr"+str(int(cur_snr))+"_"+str(noise_index)+"_models.pickle")

            tmp_pickle_path = cur_dir/(str(0)+"_"+"tmp_"+"snr"+str(int(cur_snr))+"_"+str(noise_index)+"_models.pickle")

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
                # continue
    print("failed: ", failed)
    print(pickle_path)
    
    
    
    
    # command = "python run_exp_ar_i_middle_real1short.py"
    # os.system(command)
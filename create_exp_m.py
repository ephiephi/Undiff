import torchaudio
import numpy as np
import gc

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
from torch.optim.lr_scheduler import StepLR
import argparse

import pathlib
import pandas as pd
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
from network_factory import *
from network_factory2 import *


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





# def create_noisy_files(root):
def plot_loss(loss_array,loss_test_array,j, netname,imgpath):
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(loss_array[j])
    axs[0].set_title(f'loss_array')
    axs[1].plot(loss_test_array[j])
    axs[1].set_title(f'loss_test_array min={min(loss_test_array[j])}')
    fig.suptitle(f'{netname} loss')
    fig.savefig(imgpath)
    logging.info(f"Plot saved to {imgpath}")
    plt.close(fig)




from torch.utils.data import Dataset

class CombinedNoiseDataset(Dataset):
    def __init__(self, clean_wave_tensor, g_t, idxes=None, alpha=0.9):
        """
        clean_wave_tensor: shape (N, 1, T) - the base signals
        g_t: array of noise scales, shape (M,). g_t[i] = noise scale
        idxes: a list of valid indices into g_t. If None, we'll use all [0..len(g_t)-1].
        alpha: decay factor for p(i)=alpha^i distribution, i=0..len(idxes)-1
        """
        self.clean_wave_tensor = clean_wave_tensor  # (N,1,T)
        self.N = clean_wave_tensor.shape[0]
        self.T = clean_wave_tensor.shape[-1]
        self.g_t = g_t
        if idxes is None:
            idxes = list(range(len(g_t)))
        self.idxes = sorted(idxes)
        self.n_idxes = len(self.idxes)

        # 1) "first_round": each index exactly once
        self.first_round = self.idxes[:]  # shallow copy
        random.shuffle(self.first_round)

        # 2) after we exhaust first_round, we use exponential distribution
        self.alpha = alpha
        self.probs = self._build_probs()  # array of length n_idxes

    def _build_probs_decaying(self):
        """Return normalized p(i)=alpha^i / sum_j alpha^j for i in [0..n_idxes-1]."""
        # We'll assume self.idxes is sorted ascending
        # rank i => alpha^i
        weights = []
        for rank in range(self.n_idxes):
            weights.append(self.alpha**rank)
        sumw = sum(weights)
        probs = [w/sumw for w in weights]
        return probs  # length n_idxes
    
    def _build_probs(self,N=200, sigma=100.0):
        """
        Returns a list of length N of probabilities p(i),
        where p(i) ~ exp( - (i^2) / (2 * sigma^2) ), i=0..N-1.
        We then normalize so that sum(p(i))=1.
        """
        weights = []
        for i in range(N):
            w = math.exp(- (i**2) / (2 * sigma**2))
            weights.append(w)
        total = sum(weights)
        probs = [w/total for w in weights]
        return probs

    def __len__(self):
        # We'll define length to be large, or we can define exactly self.N
        # If you want a bigger dataset, you can do multiple epochs
        return self.N

    def __getitem__(self, index):
        """
        We'll pick a single waveform from clean_wave_tensor at 'index'.
        Then pick noise-level index in two phases:
          1) If self.first_round not empty -> pop one
          2) else pick from exponential distribution
        Then add noise.
        """
        # 1) get the base wave
        clean_wave = self.clean_wave_tensor[index].clone()  # shape (1, T)

        # 2) pick a noise-level index
        if len(self.first_round) > 0:
            i = self.first_round.pop()  # pick the last or pop(0) if you prefer
        else:
            # use the exponential distribution
            # random.choices(population, weights, k=1) picks one index from population
            # But we want to sample among the possible ranks [0..n_idxes-1]
            # Then map rank-> actual idx
            rank = random.choices(range(self.n_idxes), weights=self.probs, k=1)[0]
            i = self.idxes[rank]

        # 3) get g_t[i] as noise scale
        sigma = self.g_t[i]
        # 4) add noise
        white_noise = torch.randn_like(clean_wave)
        noised_wave = clean_wave + sigma*white_noise

        # 5) build the noise_level tensor shape (1, T)
        noise_level_tensor = torch.full_like(clean_wave, sigma)

        return noised_wave, noise_level_tensor
    

def train_one_net_for_all_noise_levels(network, train_full_tensors, test_full_tensors, device, idxes,trial, epochs=6000,batch_size=16,g_t=None,exp_root=None,min_epochs=400,slope_epochs=2,quarter_idx=None,mog=0,lr=None):
    """
    Train a single noise-aware network on a range of noise levels, all at once.
    - `train_full_tensors` shape (N, 1, T)
    - `test_full_tensors`  shape (M, 1, T)
    - `g_t`: array of possible noise levels, shape (G,).
    - `idxes`: optional subset of indices into g_t to pick from. if None, use entire g_t.
    """
    # 1) Create the single noise-aware network
    # model = network_cls()  # e.g. model = NetworkNoise6(...)
    model = network
    model.to(device)
    logging.info(f"starting training for all noise levels")

    # 2) Create combined dataset that picks random noise levels from `idxes`
    train_dataset = CombinedNoiseDataset(train_full_tensors, g_t, idxes=idxes)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Optionally also create a test dataset similarly:
    test_dataset = CombinedNoiseDataset(test_full_tensors, g_t, idxes=idxes)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 3) An optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    loss_array=[[float(-1)]]
    loss_test_array = [[float(-1)]]

    # 4) Training loop
    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss = 0.0
        for j_idx, (noised_wave, noise_level_tensor) in enumerate(train_loader):
            noised_wave = noised_wave.to(device, dtype=torch.float)
            noise_level_tensor = noise_level_tensor.to(device, dtype=torch.float)

            optimizer.zero_grad()
            if mog==0:
                means, stds = model(noised_wave, noise_level_tensor)
                loss = model.casual_loss(means, stds, noised_wave).mean()
            else:
                logits, means, log_sig = model(noised_wave, cur_gt=None)
                loss = model.casual_loss(logits, means, log_sig, noised_wave).mean()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if j_idx in [0,100]:
                logging.info(f"j_idx: {j_idx}, loss: {loss.item()}")
                
            
        loss_array[0].append(running_loss/len(train_loader))

        # 5) Evaluate on test set
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for (noised_wave_test, noise_level_test) in test_loader:
                noised_wave_test = noised_wave_test.to(device, dtype=torch.float)
                noise_level_test = noise_level_test.to(device, dtype=torch.float)
                if mog==0:
                    means_t, stds_t = model(noised_wave_test, noise_level_test)
                    loss_t = model.casual_loss(means_t, stds_t, noised_wave_test).mean()
                else:
                    logits, means, log_sig = model(noised_wave_test, cur_gt=noise_level_test)
                    loss = model.casual_loss(logits, means, log_sig, noised_wave).mean()
                test_loss += loss_t.item()
        loss_test_array[0].append(test_loss/len(test_loader))

        test_loss /= len(test_loader)
        if epoch%500==0 or epoch==0:
            logging.info(f"Epoch {epoch+1}/{epochs}, train_loss={running_loss/len(train_loader):.3f}, test_loss={test_loss:.3f}")
            father_path = Path(exp_root)/ f"analysis/graphs/"
            logging.info(f"{father_path}")
            father_path.mkdir(parents=True, exist_ok=True)
            networkname = str(network).split("(")[0]
            imgpath = father_path/ f"loss_net_{networkname}_i{0}_epoch{epoch}.jpg"
            logging.info("plot_loss")
            plot_loss(loss_array,loss_test_array,0, network,imgpath)
        
    father_path = Path(exp_root)/ f"analysis/graphs/"
    logging.info(f"{father_path}")
    father_path.mkdir(parents=True, exist_ok=True)
    networkname = str(network).split("(")[0]
    imgpath = father_path/ f"loss_net_{networkname}_i{0}.jpg"
    logging.info("plot_loss")
    plot_loss(loss_array,loss_test_array,0, network,imgpath)

    return [model for i in range(len(idxes))], loss_array, loss_test_array, quarter_idx
    
    



import torch.multiprocessing as mp
import copy
from create_exp_m import NetworkNoiseWaveNetMoG2
def train_nets_process(network, train_full_tensors, test_full_tensors, device, idxes,trial, epochs=6000,batch_size=16,g_t=None,exp_root=None,min_epochs=400,slope_epochs=2,quarter_idx=None,mog=0,lr=None,one_network=False,scheduler=None):
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
    # elif trial==1:
    #     nets = [Network2() for i in range(len(idxes))]
    # elif trial==2:
    #     nets = [Network3() for i in range(len(idxes))]
    # elif trial==3:
    #     nets = [Network4() for i in range(len(idxes))]
    
    logging.info(f"idxes[0]: {idxes[0]}")
    logging.info(f"quarter_idx: {quarter_idx}")
    # if idxes[0] == 0:
    #     quarter_idx=0
    # elif idxes[0] == 50:
    #     quarter_idx=1
    # elif idxes[0] == 100:
    #     quarter_idx=2
    # elif idxes[0] == 150:
    #     quarter_idx=3
    # else:
    #     logging.info ("no identifiesd quarter")
    #     raise Exception
    
    # cur_epochs = epochs
    if one_network:
        return train_one_net_for_all_noise_levels(nets[0], train_full_tensors, test_full_tensors, device, idxes,trial, epochs,batch_size,g_t,exp_root,min_epochs,slope_epochs,quarter_idx,mog,lr,scheduler)
    

    loss_array = {}
    loss_test_array = {}

    net_counter=-1
    for i in idxes:
        
        cur_white_noise_diffusion = torch.normal(0,1,train_full_tensors[:,0,:].shape)
        cur_train_full_tensors = train_full_tensors[:,0,:]+cur_white_noise_diffusion*g_t[i]
        
        cur_white_noise_diffusion = torch.normal(0,1,test_full_tensors[:,0,:].shape)
        cur_test_full_tensors = test_full_tensors[:,0,:]+cur_white_noise_diffusion*g_t[i]
        
        #Create TensorDatasets
        train_dataset_size_ = train_full_tensors.shape[0]
        test_dataset_size_ = test_full_tensors.shape[0]
        train_dataset = BatchNoiseDataset(cur_train_full_tensors.reshape(train_dataset_size_,1,-1),g_t[i])
        test_dataset = BatchNoiseDataset(cur_test_full_tensors.reshape(test_dataset_size_,1,-1),g_t[i])

        #Create DataLoaders
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) #todo: numbers
        test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
        
        net_counter+=1
        model = nets[net_counter]
        model.to(device)
        model.train()

        cur_epochs = int((epochs-min_epochs)*(1-i/200)**slope_epochs+min_epochs)
        
        if lr is not None:
            optimizer = optim.Adam(model.parameters(), lr=lr)
        else:
            optimizer = optim.Adam(model.parameters())
        if scheduler is not None:
            scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
        
        

        for epoch in range(cur_epochs):
            running_loss = 0.0
            for batch_idx, (batch_tensor, gt_tensor) in enumerate(train_loader):
                optimizer.zero_grad(set_to_none=True)
                batch_tensor = batch_tensor.to(device, dtype=torch.float)
                gt_tensor = gt_tensor.to(device, dtype=torch.float)
                
                if mog==0:
                    means, stds = model(batch_tensor, gt_tensor)
                    # means, stds = torch.utils.checkpoint.checkpoint_sequential( model.layers, segments=4, input=(batch_tensor,gt_tensor))
                    loss = model.casual_loss( means, stds, wav_tensor=batch_tensor).mean()
                else:
                    logits, means, log_sig = model(batch_tensor, cur_gt=None)
                    loss = model.casual_loss(logits, means, log_sig, batch_tensor).mean()
                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()
                
                # gc.collect()
                running_loss += loss.item()
                
            if epoch%1==0:
                with torch.no_grad():
                    for batch_idx, (test_inputs, gt_test) in enumerate(test_loader):
                        test_inputs = test_inputs.to(device, dtype=torch.float)
                        gt_test = gt_test.to(device, dtype=torch.float)
                        if mog==0:
                            meanst, stdst = model(test_inputs, gt_test)
                            loss_t = model.casual_loss( meanst, stdst, wav_tensor=test_inputs).mean()
                        else:
                            logits_t, meanst, log_sigt = model(test_inputs, cur_gt=None)
                            loss_t = model.casual_loss(logits_t, meanst, log_sigt, test_inputs).mean()                          
                if i in loss_test_array:
                    loss_test_array[i].append(float(loss_t))
                else:
                    loss_test_array[i] = [float(loss_t)]

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
            scheduler.step()
        nets[net_counter].parameters = model.parameters
        logging.info(f"Model {i} Epoch {epoch+1}/{cur_epochs}, Loss: {running_loss}")
        
        if i in [0,20,50,100,150] or net_counter==0:
            if exp_root == None:
                logging.info("dont have path for graph")
            else:
                father_path = Path(exp_root)/ f"analysis/graphs/"
                logging.info(f"{father_path}")
                father_path.mkdir(parents=True, exist_ok=True)
                imgpath = father_path/ f"loss_net_{network}_i{i}.jpg"
                logging.info("plot_loss")
                plot_loss(loss_array,loss_test_array,i, network,imgpath)
    logging.info(f"test_loss mins array:  {mins}")
    # gc.collect()
    return nets_min, loss_array, loss_test_array, quarter_idx




def get_group_indices(numbers, num_groups=4):
    """
    Divides the given list of numbers into num_groups sequential groups
    such that the sum of numbers in each group is as close as possible.
    
    Returns the indices for each group.
    """
    total_sum = sum(numbers)
    target_sum = total_sum / num_groups
    groups = []
    current_group = []
    current_sum = 0

    for i, num in enumerate(numbers):
        current_sum += num
        current_group.append(i)

        # If current group sum exceeds or is close to the target, finalize the group
        if current_sum >= target_sum and len(groups) < num_groups - 1:
            groups.append(current_group)
            current_group = []
            current_sum = 0

    # Add remaining indices to the last group
    groups.append(current_group)
    
    return groups




def train_nets_parralel(network,train_dataset, test_dataset,trial=0, epochs=100,num_nets=200,batch_size=16,g_t=None,exp_root=None,min_epochs=400,slope_epochs=2,mog=0,lr=None,one_network=False,scheduler=None):
    results = []
    gpu_num=torch.cuda.device_count()
    if one_network:
        gpu_num=1
    # gpu_num = 1 #todo
    logging.info("parralel")
    logging.info(f"epochs,min_epochs,slope_epochs: ,{epochs},{min_epochs},{slope_epochs}")
    

    numbers = [int((epochs-min_epochs)*(1-i/200)**slope_epochs+min_epochs) for i in range(200)]
    idxes_all = get_group_indices(numbers,num_groups=gpu_num)
    # idxes_all = [list(range(0,100)),list(range(100,200))]
    
    devices = [f'cuda:{i}' for i in range(gpu_num)]
    # devices = ["cuda:0"]
    if len(devices) >1:
        with mp.get_context('spawn').Pool(processes=gpu_num) as pool:
            args = [(network, train_dataset, test_dataset,devices[i % gpu_num], idxes, trial,epochs,batch_size,g_t,exp_root,min_epochs,slope_epochs,i,mog,lr,one_network,scheduler) for i, idxes in enumerate(idxes_all)]
            results = pool.starmap(train_nets_process, args)

        loss_array= results[0][1]
        loss_test_array = results[0][2]
        nets = results[0][0]
        for i in range(1,gpu_num):
            nets.extend(results[i][0])
            loss_array.update(results[i][1])
            loss_test_array.update(results[i][2])
    else:
        nets, loss_array, loss_test_array, quarter_idx = train_nets_process(network, train_dataset, test_dataset,devices[0], idxes_all[0], trial,epochs,batch_size,g_t,exp_root,min_epochs,slope_epochs,0,mog,lr,one_network,scheduler)
    idxes_all = [list(range(0,200))]

    # nets, loss_array, loss_test_array, quarter_idx = train_nets_process(train_dataset, test_dataset,"cuda:1", idxes_all, trial,epochs=epochs)
    return nets,loss_array,loss_test_array



def train_noisemodel(root, network="NetworkNoise2",epochs=60, dataset_size=128*8, n_samples=640000,batch_size=16,g_t=None,num_nets=200,min_epochs=400,slope_epochs=2,noise_type=None,mog=0):
    logging.info("starting ----")

    logging.info(f"{root}")
    
    with open(Path(root)/'5f_snrs.pickle', 'rb') as handle:
        snr_df = pickle.load(handle)
    logging.info(f"{snr_df}")
    
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
                logging.info(f"{cur_snr}, continue")
                continue
            
            noise_idx = snr_df["noise_idx"][train_idx]
            logging.info(f"noise_idx, {noise_idx}")
            # if noise_idx!="Babble":
            #     logging.info("noise_idx", noise_idx)
            #     continue
            cur_dir = Path(root)/ snr_df["dir"][train_idx]
            logging.info(f"______{cur_dir}_______")

            cur_snr = snr_df["snr"][train_idx]
            cur_noise_scaling = snr_df["noise_scaling"][train_idx]
            noise_index = snr_df["noise_idx"][train_idx]
            logging.info(f"{cur_snr},{cur_dir}, {noise_index}, {cur_noise_scaling}")

            noise_path = snr_df["noise_path"][train_idx]
            speech_path = snr_df["clean_wav"][train_idx]
            noisy_path = snr_df["noisy_wav"][train_idx]
            noise_whole, sr = torchaudio.load(noise_path)
            if noise_type=="simple_ar":
                if noise_idx=="1":
                    ar_coefs = [0.9]
                elif noise_idx=="2":
                    ar_coefs = [0.6,-0.1, 0.2]
                    order=3
                elif noise_idx=="3":
                    ar_coefs = [-0.9]
                else:
                    logging.info("unknown noise")
                    continue
            elif noise_type=="complicated_ar":
                if noise_idx=="1":
                    ar_coefs = [ 0.4,-0.1, 0.1, -0.05,0.03]
                else:
                    logging.info("unknown noise")
            elif noise_type=="complicated2_ar":
                if noise_idx=="1":
                    ar_coefs = [ 0.4,-0.1, 0.1, -0.05,0.03, -0.01,0.06, -0.05,0.003,0.09,0.01,0.03, -0.05,0.02, -0.01,0.03, -0.005,0.009, -0.05,0.03]
                else:
                    logging.info("unknown noise")
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
            
            
            logging.info("creating tensors")

            train_full_tensors = train_tensor.squeeze().view(dataset_size,1,-1)
            test_full_tensors = test_tensor.reshape(dataset_size,200,-1)


            
            logging.info("starting training")
            # nets,loss_array,loss_test_array = train_nets_parralel(network,train_full_tensors, test_full_tensors,trial,epochs=epochs,num_nets=200,batch_size=batch_size,g_t=g_t)#
            nets,loss_array,loss_test_array = train_nets_parralel(network,train_full_tensors, test_full_tensors,trial,epochs=epochs,num_nets=num_nets,batch_size=batch_size,g_t=g_t,exp_root=exp_root,min_epochs=min_epochs,slope_epochs=slope_epochs,mog=mog)

            logging.info("end 1 training")
            
            params_dict = {"nets": nets, "train_dataset": None, "test_dataset": None,"ar_coefs":None, "loss_array":loss_array, "loss_test_array": loss_test_array, "ar_noise": None, "noise_scaling": cur_noise_scaling, "snr": str(int(cur_snr)), "noise_name": noise_idx, "noise_path": noise_path}
            # params_dict = {"result": result}
            params_dict_debug = {"nets": nets, "train_dataset": train_full_tensors, "test_dataset": test_full_tensors,"ar_coefs":None, "loss_array":loss_array, "loss_test_array": loss_test_array, "ar_noise": None, "noise_scaling": cur_noise_scaling, "snr": str(int(cur_snr)), "noise_name": noise_idx, "noise_path": noise_path}
            
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
    logging.info(f"failed:  {failed}")
    logging.info(f"{pickle_path}")







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
                logging.info(f"{name}")
                names.append(name)
            except:
                logging.info("no name in filename")
            dir_path = os.path.join(exp_root,name)
            if os.path.exists(dir_path):
                logging.info(f"{name}  exist")
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


def create_noises(exp_root, noise_type,seconds=4):
    noisees_rooot = Path(exp_root)/"noises"
    sample_rate = 16000
    n_samples = sample_rate*(5+seconds)
    
    if noise_type=="simple_ar":
        ar_coefs = [0.9]
        noise1 = create_ar_noise(n_samples, ar_coefs,order=len(ar_coefs), dtype=torch.float32)

        ar_coefs = [0.6,-0.1, 0.2]
        noise2 = create_ar_noise(n_samples, ar_coefs,order=len(ar_coefs), dtype=torch.float32)

        ar_coefs = [-0.9]
        noise3 = create_ar_noise(n_samples, ar_coefs,order=len(ar_coefs), dtype=torch.float32)


        tar_noisypath = (noisees_rooot)/"1.wav"
        torchaudio.save(tar_noisypath, noise1, sample_rate,encoding="PCM_F")

        tar_noisypath = (noisees_rooot)/"2.wav"
        torchaudio.save(tar_noisypath, noise2, sample_rate,encoding="PCM_F")

        tar_noisypath = (noisees_rooot)/"3.wav"
        torchaudio.save(tar_noisypath, noise3, sample_rate,encoding="PCM_F")
        noises = ["1","2","3"]
    elif noise_type=="complicated_ar":
        ar_coefs = [ 0.4,-0.1, 0.1, -0.05,0.03]
        noise1 = create_ar_noise(n_samples, ar_coefs,order=len(ar_coefs), dtype=torch.float32)

        tar_noisypath = (noisees_rooot)/"1.wav"
        torchaudio.save(tar_noisypath, noise1, sample_rate,encoding="PCM_F")

        noises = ["1"]
    elif noise_type=="complicated2_ar":
        ar_coefs = [ 0.4,-0.1, 0.1, -0.05,0.03, -0.01,0.06, -0.05,0.003,0.09,0.01,0.03, -0.05,0.02, -0.01,0.03, -0.005,0.009, -0.05,0.03]
        noise1 = create_ar_noise(n_samples, ar_coefs,order=len(ar_coefs), dtype=torch.float32)

        tar_noisypath = (noisees_rooot)/"1.wav"
        torchaudio.save(tar_noisypath, noise1, sample_rate,encoding="PCM_F")

        noises = ["1"]
    return noises


def noise_waves(exp_root, snr_array=[5]):
    noiseroot = os.path.join(exp_root,"noises/")
    noise_files =  glob(noiseroot+"/*")
    logging.info(f"{noise_files}")

    dirs_ =  glob(exp_root+"/*")

    idx = 0
    sample_rate=16000
    dict_ = {"snr":[],"noise_scaling": [], "noise_idx":[], "dir": [] }
    df_snr = pd.DataFrame(data=dict_)

    for dir_ in dirs_:
        if os.path.basename(dir_)=="noises" or os.path.basename(dir_) =='5f_snrs.pickle' or os.path.basename(dir_) =='cleans' or os.path.basename(dir_) =='analysis':
            continue
        logging.info(f"dir_ {dir_}")
        clean_wavs_paths = glob(dir_+"/clean_wav/*")
        logging.info(f"{clean_wavs_paths}")
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
        logging.info(f"clean_power:  {clean_power}")
        simple_power =  1 / speech.shape[1] * torch.sum(speech**2)
        logging.info(f"simple_power: , {simple_power}")
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
                logging.info(f"speech.shape: {speech2.shape}")
                logging.info(f"noise.shape: {noise.shape}")

                lossy_speech = speech2 + noise_scaling * noise
                logging.info(f"noise_scaling: {noise_scaling}")
                
                noise_var = noise_scaling**2
                
                new_noise = noise_scaling * noise
                new_noise_power = float(1 / new_noise.shape[1] * torch.sum(new_noise**2))
                
                y_power =  1 / lossy_speech.shape[1] * torch.sum(lossy_speech**2)
                
                vaded_signal = calc_vad(speech_file_)[0:lossy_speech.shape[1],:]
                vaded_lossy_speech_torch = (lossy_speech[0][vaded_signal.T[0]>0])
                vaded_lossy_speech_torch = torch.unsqueeze(vaded_lossy_speech_torch, dim=0)
                yclean_power = float( 1 / vaded_lossy_speech_torch.shape[1] * torch.sum(vaded_lossy_speech_torch**2))

                logging.info(f"snr: {snr}")
                df_snr.at[idx, "snr"] = snr
                df_snr.at[idx, "noise_scaling"] = float(noise_scaling)
                df_snr.at[idx, "dir"] = dir_[-1]
                df_snr.at[idx, "noise_idx"] = noise_idx
                
                filename = "noise{}_digits_snr{}_power{}_var{}.wav".format(noise_idx,snr, new_noise_power, noise_var)

                tarpath = os.path.join(os.path.join(dir_, "clean_wav"), filename)
                logging.info(f"tarpath: {tarpath}")
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
                scaled_noise = scaled_noise[:,(5*sr):(speech.shape[1]+(5*sr))] ###leave 5 sec for training
                noises_root = os.path.join(dir_, "noises")
                logging.info(f"noise_scaling: {noise_scaling}")
                tar_noise =  os.path.join(noises_root, filename)
                if not os.path.exists(noises_root):
                    os.mkdir(noises_root)
                df_snr.at[idx, "noise_path"] = tar_noise
                # if noise_idx=="1" and  dir_[-1] == "b":
                #     raise Exception

                torchaudio.save(tar_noise, scaled_noise, sample_rate,encoding="PCM_F")
        

    df_snr = df_snr.sort_values(by=['dir'])
    logging.info(f"{df_snr}")

    with open(os.path.join(exp_root,'5f_snrs.pickle'), 'wb') as handle:
        pickle.dump(df_snr, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return snr_array


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="train noise for guidence")
    parser.add_argument(
        "-config",
        default="exps_configs/m_ar_net6mog5.yaml",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        trials = yaml.safe_load(f)
    
    
    s_array = trials.get("s_array", [])
    snr_array = trials.get("snr_array", [])
    exp_root = trials.get("exp_root", "")
    network = trials["network"] 
    epochs = trials.get("epochs", 0) 
    min_epochs=trials.get("min_epochs", 1500) 
    slope_epochs=trials.get("slope_epochs", 2) 
    batch_size = trials.get("batch_size", 0) 
    dataset_size = eval(trials.get("dataset_size", "0"))
    n_samples = trials["n_samples"] 
    scheduler_type = trials.get("scheduler_type","linear")
    num_steps = trials.get("num_steps",200)
    scheduler = trials.get("scheduler","60")
    # test_start_sec = trials.get("test_start_sec",6)
    # test_end_sec = trials.get("test_end_sec",10)
    trained_model_path = trials.get("trained_model_path","0")
    noise_type = trials.get("noise_type", "simple_ar") 
    mog = trials.get("mog", 0) 
    
    betas=get_named_beta_schedule(scheduler_type, num_steps)
    alphas = 1.0 - betas
    alphas_cumprod =  torch.from_numpy(np.cumprod(alphas, axis=0))
    g_t = torch.sqrt((1-alphas_cumprod)/(alphas_cumprod))
    
    log_file = setup_logging(exp_root)
    logging.info(f"\nDir: {args.config}\n")
    logging.info(f"Logging to: {log_file}")
    
    
    names = create_wav_dirs(exp_root)
    noises_names = create_noises(exp_root, noise_type=noise_type)
    snr_array = noise_waves(exp_root, snr_array=snr_array)
    # names=["p"]
    # noises_names=["1"]
    # snr_array=["5"]

    train_noisemodel(exp_root, network=network,epochs=epochs, dataset_size=dataset_size, n_samples=n_samples,batch_size=batch_size,g_t=g_t,num_nets=num_steps,min_epochs=min_epochs,slope_epochs=slope_epochs,noise_type=noise_type,mog=mog)
    run_network = network

    logging.info("---run_exp---")
    # run_exp(exp_root, dirnames=names, cuda_idx="1",s_array=s_array)
    run_exp(exp_root, dirnames=names,s_array=s_array, reset=False, s_schedule=scheduler, scheduler_type=scheduler_type,noise_mosel_path=trained_model_path,mog=mog ) #network=run_network 

    
    storm_root = str(Path(exp_root)/"storm")
    run_storm(exp_root,storm_root)
    
    analyze_exp(exp_root,noises_names,snr_array,names)


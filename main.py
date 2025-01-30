import os

import hydra
import torch

import improved_diffusion
from improved_diffusion.tasks import TaskType

from  improved_diffusion.script_util import create_gaussian_diffusion

###########################

import os
import pandas as pd
from torch.utils.data import Dataset
from torch import nn
import torch
import numpy as np


from torch.nn.modules.utils import _pair
from torch import nn
import torch.nn.functional as F

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
        return likelihood.mean()#/(stds_.shape[-1])
    
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

# class CausalConv2d(nn.Conv2d):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, dilation=1, groups=1, bias=True):
#         # kernel_size = (kernel_size[0],kernel_size[1])
#         stride = _pair(stride)
#         dilation = (dilation,1)
#         # print("dilation:", dilation)
#         if padding is None:
#             padding = int((kernel_size[1] -1) * dilation[1] +1) 
#         else:
#            padding = padding * 2
#         # print("padding:",padding)
#         self._pad= (padding)
#         self._pad_h =   int(np.floor(kernel_size[0]/2))
#         super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias)
#     def forward(self, inputs):
#         # print("inputs.shape:",inputs.shape)
#         inputs = F.pad(inputs, (self._pad, 0,self._pad_h , self._pad_h))
#         output = super().forward(inputs)
#         if self._pad != 0:
#             output = output[:, :, :-1]
#         return output
    
    
# class NetworkFreq(nn.Module):
#     def __init__(self, kernel_size=(11,5)):
#         super().__init__()
#         self.kernel_size = kernel_size
#         self.pad_h = int(np.floor(kernel_size[0]/2))
#         dilation=(1,1)
#         self.padding = int((kernel_size[1] -1) * dilation[1] +1)
#         # BCHW - https://discuss.pytorch.org/t/applying-separate-convolutions-to-each-row-of-input/152597
#         # grouped_conv = CausalConv2d(H * C, H * C, kernel_size = k, groups = H)
#         self.conv1 = CausalConv2d(257, 257*2, kernel_size=kernel_size, groups=257)
        
#         self.tanh = nn.Tanh()
#         self.conv2 = CausalConv2d(257*2, 257*2, kernel_size=kernel_size, groups=257)


#     def forward(self, x, cur_gt):
        
#         assert len(x.shape) ==4
#         B = x.shape[0]
#         C = x.shape[1]
#         H = x.shape[2]
#         W = x.shape[3]
#         # print("B,C,H:", B,C,H)
#         # print("input.shape: ", input.shape)
#         x = self.conv1(x.transpose(1, 2).reshape (H * C ,B, -1))
#         x = x.reshape (B, H, C*2, W).transpose (1, 2)
#         identity = x
#         x = self.tanh(x)
#         x = self.conv2(x.transpose(1, 2).reshape (2*H * C ,B, -1))
#         x = x.reshape (B, H, C*2, W).transpose (1, 2)
#         x = x + identity

#         means = x[:,0,:,:]
#         log_var = x[:,1,:,:]
#         stds = torch.exp(0.5 *log_var)
#         return means, stds
    
#     def calc_model_likelihood(self, expected_means, expected_stds, wav_tensor, verbose=False):
#         # print("wav_tensor.shape ", wav_tensor.squeeze().shape)
#         padding = self.padding
#         wav_tensor = wav_tensor.squeeze()[self.pad_h:-self.pad_h,padding:]
#         means_=expected_means.squeeze()[self.pad_h:-self.pad_h,padding:]
#         stds_ = expected_stds.squeeze()[self.pad_h:-self.pad_h,padding:]
#         # print("wav_tensor shape: ", wav_tensor.shape)
#         # print("means_ shape: ", means_.shape)
#         # print("stds_ shape: ", stds_.shape)
#         eps = 0.000000001
#         exp_all = -(1/2)*((torch.square(wav_tensor-means_)/(torch.square(stds_)+eps)))
#         param_all = 1/(np.sqrt(2*np.pi)*stds_+eps)

#         model_likelihood1 = torch.mean(torch.sum(torch.log(param_all),axis=1)) #, axis=-1 
#         model_likelihood2 = torch.mean(torch.sum(exp_all,axis=1)) #, axis=-1

#         if verbose:
#             print("model_likelihood1: ", model_likelihood1)
#             print("model_likelihood2: ", model_likelihood2)
#         return model_likelihood1 + model_likelihood2
    
#     def casual_loss(self, expected_means, expected_stds, wav_tensor):
#         model_likelihood = self.calc_model_likelihood(expected_means, expected_stds, wav_tensor)
#         return -model_likelihood



# def CausalConv1d(in_channels, out_channels, kernel_size, dilation=1, **kwargs):
#    pad = (kernel_size - 1) * dilation +1
#    return nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad, dilation=dilation, **kwargs)


# class CausalConv1dClass(nn.Conv1d):
#     def __init__(self,in_channels, out_channels, kernel_size, dilation=1, **kwargs):
#         pad = (kernel_size - 1) * dilation +1
#         super().__init__(in_channels, out_channels, kernel_size, padding=pad, dilation=dilation, **kwargs)
    
#     def forward(self, inputs):
#         output = super().forward(inputs)
#         if self.padding[0] != 0:
#             output = output[:, :, :-self.padding[0]-1]
#         return output


# class NetworkLong(nn.Module):
#     def __init__(self, kernel_size=50):
#         super().__init__()   
#         self.conv0 = CausalConv1dClass(1, 2, kernel_size=kernel_size, dilation=10)
#         self.tanh0 = nn.Tanh()
#         self.tanh0_1 = nn.Tanh()
#         self.sig0 = nn.Sigmoid()
#         self.conv1 = CausalConv1dClass(1, 2, kernel_size=kernel_size, dilation=1)
#         self.tanh = nn.Tanh()
#         self.tanh1_1 = nn.Tanh()
#         self.sig1 = nn.Sigmoid()
#         self.conv2 = CausalConv1dClass(2, 2, kernel_size=kernel_size, dilation=2)
#         self.tanh2 = nn.Tanh()
#         self.tanh2_1 = nn.Tanh()
#         self.sig2 = nn.Sigmoid()
#         self.conv3 = CausalConv1dClass(2, 2, kernel_size=kernel_size, dilation=4)
#         self.tanh3 = nn.Tanh()
#         self.tanh3_1 = nn.Tanh()
#         self.sig3 = nn.Sigmoid()
#         self.conv4 = CausalConv1dClass(2, 2, kernel_size=kernel_size, dilation=8)
#         self.param = nn.Parameter(torch.randn(1))
        
#         # fc_layer = nn.Linear(in_features=128, out_features=64)

#     def forward(self, x, cur_gt):
#         input = x[:]
#         x1 = self.conv0(x)
        
#         x = self.conv1(x)

#         identity=x[:]
#         x = x+x1
#         x = self.tanh(x)
#         x = self.conv2(x)

#         identity2=x[:]
#         x = identity+x
#         x = self.tanh2(x)
#         x = self.conv3(x)

#         identity3=x[:]
#         x = x+identity2+identity
#         x = self.tanh3(x)
#         x = self.conv4(x)

#         x = x+identity2+identity+identity3

#         means = x[:,0,:]
#         log_var = x[:,1,:]
#         stds = torch.exp(0.5 *log_var)

#         return means, stds
    
#     def calc_model_likelihood(self, expected_means, expected_stds, wav_tensor, verbose=False):
#         wav_tensor = wav_tensor.squeeze()
#         means_=expected_means.squeeze()
#         stds_ = expected_stds.squeeze()

#         exp_all = -(1/2)*((torch.square(wav_tensor-means_)/torch.square(stds_)))
#         param_all = 1/(np.sqrt(2*np.pi)*stds_)
#         model_likelihood1 = torch.sum(torch.log(param_all), axis=-1) 
#         model_likelihood2 = torch.sum(exp_all, axis=-1) 

#         if verbose:
#             print("model_likelihood1: ", model_likelihood1)
#             print("model_likelihood2: ", model_likelihood2)
        
            
#         return model_likelihood1 + model_likelihood2
    
#     def casual_loss(self, expected_means, expected_stds, wav_tensor):
#         model_likelihood = self.calc_model_likelihood(expected_means, expected_stds, wav_tensor)
#         loss3 = torch.sum(expected_stds,axis=-1)
#         return -model_likelihood + self.param*loss3/wav_tensor.shape[-1]


# class Network(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = CausalConv1d(1, 2, kernel_size=3, dilation=1)

#     def forward(self, x, cur_gt):
#         x = self.conv1(x)
#         # print("self.conv1.padding: ", self.conv1.padding)
#         if self.conv1.padding[0] != 0:
#             x = x[:, :, :-self.conv1.padding[0]-1]  # remove trailing padding
#         means = x[:,0,:]
#         log_var = x[:,1,:]
#         stds = torch.exp(0.5 *log_var)
#         return means, stds
    
#     def calc_model_likelihood(self, expected_means, expected_stds, wav_tensor, verbose=False):
#         wav_tensor = wav_tensor.squeeze()
#         means_=expected_means.squeeze()
#         stds_ = expected_stds.squeeze()

#         exp_all = -(1/2)*((torch.square(wav_tensor-means_)/torch.square(stds_)))
#         param_all = 1/(np.sqrt(2*np.pi)*stds_)
#         model_likelihood1 = torch.sum(torch.log(param_all), axis=-1) 
#         model_likelihood2 = torch.sum(exp_all, axis=-1) 

#         if verbose:
#             print("model_likelihood1: ", model_likelihood1)
#             print("model_likelihood2: ", model_likelihood2)
#         return model_likelihood1 + model_likelihood2
    
#     def casual_loss(self, expected_means, expected_stds, wav_tensor):
#         model_likelihood = self.calc_model_likelihood(expected_means, expected_stds, wav_tensor)
#         return -model_likelihood

# class Network(nn.Module):
#     def __init__(self, cutoff=6000,high=True, kernel_size=5):
#         self.high = high
#         self.cutoff = cutoff
#         super().__init__()
#         self.conv1 = CausalConv1d(1, 2, kernel_size=kernel_size, dilation=1)

#     def forward(self, x, cur_gt):
#         x = self.conv1(x)
#         # print("self.conv1.padding: ", self.conv1.padding)
#         if self.conv1.padding[0] != 0:
#             x = x[:, :, :-self.conv1.padding[0]-1]  # remove trailing padding
#         means = x[:,0,:]
#         log_var = x[:,1,:]
#         stds = torch.exp(0.5 *log_var)
#         return means, stds
    
#     def calc_model_likelihood(self, expected_means, expected_stds, wav_tensor, verbose=False):
#         # model_likelihood=0
#         wav_tensor = wav_tensor.squeeze()
#         means_=expected_means.squeeze()
#         stds_ = expected_stds.squeeze()
#         if self.high:
#             wav_tensor = wav_tensor[self.cutoff:]
#             means_=means_[self.cutoff:]
#             stds_ = stds_[self.cutoff:]
#         else:
#             wav_tensor = wav_tensor[:self.cutoff]
#             means_=means_[:self.cutoff]
#             stds_ = stds_[:self.cutoff]
#         exp_all = -(1/2)*((torch.square(wav_tensor-means_)/torch.square(stds_)))
#         param_all = 1/(np.sqrt(2*np.pi)*stds_)
#         model_likelihood1 = torch.sum(torch.log(param_all), axis=-1) 
#         model_likelihood2 = torch.sum(exp_all, axis=-1) 

#         # model_likelihood2 = torch.sum(torch.log(1/(np.sqrt(2*np.pi)*stds_)), axis=-1) 
#         # model_likelihood = model_likelihood + model_likelihood2
#         if verbose:
#             print("model_likelihood1: ", model_likelihood1)
#             print("model_likelihood2: ", model_likelihood2)
            
#         return model_likelihood1 + model_likelihood2
    
#     def casual_loss(self, expected_means, expected_stds, wav_tensor):
#         model_likelihood = self.calc_model_likelihood(expected_means, expected_stds, wav_tensor)
#         return -model_likelihood
    

# class NetworkDivided(nn.Module):
#     def __init__(self, kernel_size=5):
#         super().__init__()
#         self.conv1 = CausalConv1d(1, 2, kernel_size=kernel_size, dilation=1)

#     def forward(self, x, cur_gt):
#         x = self.conv1(x)
#         # print("self.conv1.padding: ", self.conv1.padding)
#         if self.conv1.padding[0] != 0:
#             x = x[:, :, :-self.conv1.padding[0]-1]  # remove trailing padding
#         means = x[:,0,:]
#         log_var = x[:,1,:]
#         stds = torch.exp(0.5 *log_var)
#         return means, stds
    
#     def calc_model_likelihood(self, expected_means, expected_stds, wav_tensor, verbose=False):
#         # model_likelihood=0
#         wav_tensor = wav_tensor.squeeze()
#         means_=expected_means.squeeze()
#         stds_ = expected_stds.squeeze()

#         exp_all = -(1/2)*((torch.square(wav_tensor-means_)/torch.square(stds_)))
#         param_all = 1/(np.sqrt(2*np.pi)*stds_)
#         model_likelihood1 = torch.sum(torch.log(param_all), axis=-1) 
#         model_likelihood2 = torch.sum(exp_all, axis=-1) 

#         if verbose:
#             print("model_likelihood1: ", model_likelihood1)
#             print("model_likelihood2: ", model_likelihood2)
            
#         return model_likelihood1 + model_likelihood2
    
#     def casual_loss(self, expected_means, expected_stds, wav_tensor):
#         model_likelihood = self.calc_model_likelihood(expected_means, expected_stds, wav_tensor)
#         return -model_likelihood
########################

def instantiate_model_and_diffusion(cfg, device):
    model = hydra.utils.instantiate(cfg.model.model)

    for param in model.parameters():
        param.requires_grad = False

    # load checkpoint
    pl_ckpt = torch.load(cfg.model.ckpt_path, map_location="cpu")
    model_state = improved_diffusion.remove_prefix_from_state_dict(
        pl_ckpt["state_dict"], j=1
    )

    # load ema
    if cfg.use_ema:
        ema_params = pl_ckpt["ema_params"][0]
        model_state = improved_diffusion.create_state_dict_from_ema(
            model_state, model, ema_params
        )

    # load state_dict
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    # define diffusion
    diffusion = hydra.utils.call(cfg.diffusion)

    return model, diffusion


@hydra.main(config_path="configs", config_name="inference_cfg", version_base=None)
def inference(cfg):
    DEVICE = cfg.device

    metrics_list = [
        hydra.utils.instantiate(cfg.task.metrics[metric], device=DEVICE)
        for metric in cfg.task.metrics
    ]

    task = hydra.utils.instantiate(
        cfg.task.name, output_dir=cfg.output_dir, metrics=metrics_list
    )

    model, diffusion = instantiate_model_and_diffusion(cfg, DEVICE)

    files_or_num = (
        list(map(lambda x: os.path.join(cfg.audio_dir, x), os.listdir(cfg.audio_dir)))
        if task.task_type != TaskType.UNCONDITIONAL
        else cfg.audio_dir
    )

    task.inference(
        files_or_num, model, diffusion, cfg.sampling_rate, cfg.segment_size, DEVICE, cfg.guidance, cfg.guid_s, cfg.cur_noise_var,cfg.y_noisy, cfg.outpath, cfg.clean_wav, cfg.s_schedule, cfg.noise_type, cfg.noise_model_path, cfg.l_low, cfg.network
    )


if __name__ == "__main__":
    inference()

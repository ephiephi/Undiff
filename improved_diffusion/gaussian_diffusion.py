"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py
Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math

import numpy as np
import torch
import torch as th
import torch

from .losses import discretized_gaussian_log_likelihood, normal_kl
from .nn import mean_flat
from .tasks import TaskType

import torchaudio
import pickle
from torch import nn
from torch.utils.data import Dataset

from torch.nn.modules.utils import _pair
from torch import nn
import torch.nn.functional as F


from scipy.signal import firwin
from network_factory import *
from network_factory2 import *


def calculate_rolling_std_with_means(audio_tensor, means_tensor, window_size=10):
    """
    Efficiently calculate the rolling standard deviation using precomputed rolling means.

    Parameters:
        audio_tensor (torch.Tensor): Input tensor of shape (1, n).
        means_tensor (torch.Tensor): Precomputed rolling means tensor of shape (1, n).
        window_size (int): The size of the sliding window (default is 10).

    Returns:
        torch.Tensor: Tensor of shape (1, n) with rolling standard deviations.
    """
    # Ensure the input is 2D
    if len(audio_tensor.shape) != 2 or audio_tensor.shape[0] != 1:
        raise ValueError("Input audio_tensor must be of shape (1, n)")
    if len(means_tensor.shape) != 2 or means_tensor.shape[0] != 1:
        raise ValueError("Input means_tensor must be of shape (1, n)")
    if audio_tensor.shape != means_tensor.shape:
        raise ValueError("audio_tensor and means_tensor must have the same shape")

    n = audio_tensor.shape[1]
    std_tensor = torch.zeros_like(audio_tensor)

    # Flatten for efficient processing
    audio_flat = audio_tensor.flatten()
    means_flat = means_tensor.flatten()

    # Compute cumulative sums of squared values
    cumsum_sq = torch.cumsum(torch.cat((torch.tensor([0.0], device=audio_tensor.device), audio_flat**2)), dim=0)

    if n >= window_size:
        rolling_sum_sq = cumsum_sq[window_size:] - cumsum_sq[:-window_size]

        # Compute variance using the formula Var = (E[X^2] - E[X]^2)
        rolling_var = (rolling_sum_sq / window_size) - means_flat[window_size - 1:]**2
        rolling_std = torch.sqrt(torch.clamp(rolling_var, min=0.0))  # Avoid negative due to precision issues

        # Assign to the result for indices where full window is available
        std_tensor[0, window_size - 1:] = rolling_std

    # For the first `window_size - 1` elements, compute manually
    for i in range(window_size - 1):
        segment = audio_flat[max(0, i - window_size + 1): i + 1]
        mean_segment = torch.mean(segment)
        std_tensor[0, i] = torch.sqrt(torch.mean((segment - mean_segment)**2))

    return std_tensor

def fir_filter(tensor, high=False, cutoff=6000,num_taps=101,sr=16000):
    cutoff_freq = cutoff  # Cutoff frequency for the high-pass filter in Hz
    num_taps = num_taps  # Filter order (number of filter coefficients)
    if high:
        pass_zero = False
    else:
        pass_zero = True
    coefficients = firwin(num_taps, cutoff=cutoff_freq, pass_zero=pass_zero, fs=sr)

    # Convert filter coefficients to a PyTorch tensor
    coefficients = torch.tensor(coefficients, dtype=torch.float32)

    pad_length = (num_taps - 1) // 2
    signal_padded = torch.nn.functional.pad(tensor.view(1, 1, -1), (pad_length, pad_length), mode='constant')

    # Apply the FIR filter to the signal using convolution
    filtered_signal = torch.nn.functional.conv1d(signal_padded.view(1, 1, -1).to("cuda"), coefficients.view(1, 1, -1).to("cuda"))
    return filtered_signal

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

# def CausalConv1d(in_channels, out_channels, kernel_size, dilation=1, **kwargs):
#    pad = (kernel_size - 1) * dilation +1
#    return nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad, dilation=dilation, **kwargs)

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
#         # model_likelihood=0
#         wav_tensor = wav_tensor.squeeze()
#         means_=expected_means.squeeze()
#         stds_ = expected_stds.squeeze()
#         # for i in range(len(wav_tensor)):
#         #     exp_ = torch.exp(-(1/(2*stds_[i]**2))*(wav_tensor[i]-means_[i])**2)
#         #     param_ = 1/(np.sqrt(2*np.pi)*stds_[i])
#         #     model_likelihood_dot += torch.log(exp_*param_)
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
    

def calc_stft(tensor):
    # Parameters
    sample_rate = 16000  # Sample rate in Hz
    n_fft = 512  # Number of FFT points
    win_length = n_fft  # Window length
    hop_length = int(win_length/2)  # Number of samples between frames
    window = torch.hann_window(win_length).to("cuda")  # Window function

    signal_ = tensor.view(-1)
    duration = max(tensor.shape)

    stft = torch.stft(signal_, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)
    return stft, duration, sample_rate



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


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.
    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    L1 = enum.auto()
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.
    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42
    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
        input_sigma_t=False,
        cond_strength=-1,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps
        self.input_sigma_t = input_sigma_t

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        self.beta_variance = self.betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas = alphas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)
        
        self.s_guid_scheduler_linear = np.linspace(0, 1, num=len(alphas))
        self.s_guid_scheduler_linear2 = np.linspace(0.04, 1, num=len(alphas))
        
        x_ = np.linspace(0, np.pi, len(alphas))  
        self.s_guid_scheduler_sinusoidal = (np.sin(x_ - np.pi/2) + 1) / 2
        
        x_2 = np.linspace(0, 10, num=len(alphas))  
        self.s_guid_scheduler_exponential = (np.exp(x_2))/(np.exp(x_2[-1]))
        x_3 = np.linspace(0, 20, num=len(alphas))  
        self.s_guid_scheduler_exponential2 = (np.exp(x_3))/(np.exp(x_3[-1]))
        x_4 = np.linspace(0, 5, num=len(alphas))  
        self.s_guid_scheduler_exponential3 = (np.exp(x_4))/(np.exp(x_4[-1]))
        #sinexp
        def sin_exp(alphas, sinus_idx = 100):
            #sinus
            x_ = np.linspace(0, np.pi, len(alphas))  
            s_guid_scheduler_sinusoidal = (np.sin(x_ - np.pi/2) + 1) / 2

            # exponent 
            start_value = 1.08
            end_value = 1.48
            exponential_array = np.logspace(np.log(start_value), np.log(end_value), sinus_idx, base=np.exp(20))
            exponential_array = exponential_array/exponential_array[-1]*(s_guid_scheduler_sinusoidal[sinus_idx])
            new_scheduler = np.concatenate((exponential_array,  s_guid_scheduler_sinusoidal[sinus_idx:]))
            return new_scheduler
        self.s_guid_scheduler_sin_exp = sin_exp(alphas)
        #convex
        def convex(alphas,basenum=50):
            start_value = 1.48
            end_value = 1.08
            exponential_array = np.logspace(np.log(start_value), np.log(end_value), len(alphas), base=np.exp(basenum))
            exponential_array = exponential_array/exponential_array[0]#*(s_guid_scheduler_sinusoidal[sinus_idx])
            # new_scheduler = np.concatenate((exponential_array,  s_guid_scheduler_sinusoidal[sinus_idx:]))
            # new_scheduler = np.concatenate((exponential_array,  s_guid_scheduler_sinusoidal[150:]))
            return 1-exponential_array
        self.s_guid_scheduler_convex = convex(alphas)
        self.constant_scheduler = np.ones(len(alphas))
        #sinus_increase
        x_ = np.linspace(0, np.pi, len(alphas))  
        scheduler_sinusoidal =1.2- (np.sin(x_ - np.pi/2) + 1) / 10 
        self.sinus_increase = scheduler_sinusoidal
        


        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        self.cond_strength = cond_strength

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).
        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        degradation=None,
        orig_x=None,
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)

        if self.input_sigma_t:
            model_output = model(
                x, _extract_into_tensor(self.betas, t, t.shape), **model_kwargs
            )
        else:
            model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                ############
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            if degradation is not None and orig_x is not None:
                pred_xstart = (
                    pred_xstart + degradation(orig_x) - degradation(pred_xstart)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        cond_fn=None,
        degradation=None,
        orig_x=None,
        guidance=False,
        guid_s=0,
        cur_noise_var=None,
        y_noisy=None,
        s_schedule=None,
        noise_type=None, 
        noise_model=None,
        diffusion_idx=None,
        l_low=0.8
    ): ########
        """
        Sample x_{t-1} from the model at the given timestep.
        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            degradation=degradation,
            orig_x=orig_x,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0

        # if cond_fn is not None:
        #     out["mean"] = self.condition_mean(
        #         cond_fn, out, x, t, model_kwargs=model_kwargs
        #     )
        cur_mean = out["mean"]
        sigma_t = th.exp(0.5 * out["log_variance"])
        if guidance:
            alpha_bar_t = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
            # c3 = 1 / torch.sqrt(Alpha_bar[t])
            c3 = _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape)
            # c4 = (1 - Alpha_bar[t]) / Alpha_bar[t]
            c4 = torch.div(1- alpha_bar_t, alpha_bar_t)
            g_t = torch.sqrt(c4)
            # print("c3: ", c3)
            # print("c4: ", c4)
            # print("x: ", x)
            # print("y_noisy: ", y_noisy)
            # grad_log_p = c3 * (y_noisy - c3 * x) / (c4 + cur_noise_var) ** 2  #bug
            # self.s_guid_scheduler_linear
            # self.s_guid_scheduler_sinusoidal print("t: ", t)
            # print("t: ", t)
            s_scheduler = self.s_guid_scheduler_linear
            if s_schedule=="linear":
                s_scheduler = self.s_guid_scheduler_linear
            elif s_schedule=="linear2":
                s_scheduler = self.s_guid_scheduler_linear2
            elif s_schedule=="exponential":
                s_scheduler = self.s_guid_scheduler_exponential
            elif s_schedule=="exponential2":
                s_scheduler = self.s_guid_scheduler_exponential2
            elif s_schedule=="sinusoidal":
                s_scheduler = self.s_guid_scheduler_sinusoidal
            elif s_schedule=="exponential3":
                s_scheduler = self.s_guid_scheduler_exponential3
            elif s_schedule=="sin_exp_150":
                s_scheduler = self.s_guid_scheduler_sin_exp
            elif s_schedule=="convex":
                s_scheduler = self.s_guid_scheduler_convex
            elif s_schedule=="constant":
                s_scheduler = self.constant_scheduler
            elif s_schedule=="sinus_increase":
                s_scheduler = self.sinus_increase
            else:
                def convex_(alphas,basenum=50):
                    start_value = 1.48
                    end_value = 1.08
                    exponential_array = np.logspace(np.log(start_value), np.log(end_value), len(alphas), base=np.exp(basenum))
                    exponential_array = exponential_array/exponential_array[0]#*(s_guid_scheduler_sinusoidal
                    return 1-exponential_array
                print("s_schedule: ", s_schedule)
                s_scheduler = convex_(self.alphas, basenum=float(s_schedule))
                
                # def sinus_increase_range(alphas, range_=8):
                #     range_ = range_*2
                #     x_ = np.linspace(0, np.pi, len(alphas))
                #     return 1+2/range_- (np.sin(x_ - np.pi/2) + 1) / range_
                # s_scheduler = sinus_increase_range(self.alphas, range_=float(s_schedule))
                
            cur_s = _extract_into_tensor(guid_s*s_scheduler, t, x.shape)
            # cur_s = _extract_into_tensor(guid_s*10*np.ones(s_scheduler.shape), t, x.shape)

            if noise_type == "y_model":
                # print("--------y_model----------")
                with torch.enable_grad():
                    mu_theta = cur_mean[:].detach()#.requires_grad_(True)
                    y_model = noise_model.to("cuda")
                    #n_t = y_noisy - c3 * x_t
                    mu_theta = mu_theta.requires_grad_(True)
                    # mu, sig = noise_model(n_t,"none" )
                    mu, log_sigma = y_model(mu_theta,y_noisy) ###x_t|x=mu_theta
                    
                    try:
                        loss = y_model.casual_loss(mu, log_sigma, y_noisy, offset=False).to("cuda")
                    except:
                        loss = y_model.casual_loss(mu, log_sigma, y_noisy).to("cuda")
                    grad_log_p = -c3 * torch.autograd.grad(loss, mu_theta)[0]
            elif noise_type == "loss_model":
                # print("--------loss_model----------")
                with torch.enable_grad():
                    x_t = cur_mean[:].detach()#.requires_grad_(True)
                    noise_model = noise_model.to("cuda")
                    n_t = y_noisy - c3 * x_t
                    n_t = n_t.requires_grad_(True)
                    # mu, sig = noise_model(n_t,"none" )
                    mu, sig = noise_model(n_t,g_t)
                    
                    # mu, sig = noise_model(n_t)
                    # loss = noise_model.gaussian_nll_loss(mu, sig, n_t).to("cuda")
                    try:
                        loss = noise_model.calc_model_likelihood(mu, sig, n_t, offset=False).to("cuda")###false
                    except:
                        loss = noise_model.calc_model_likelihood(mu, sig, n_t).to("cuda")
                    # loss = loss*len()
                    grad_log_p = -c3 * torch.autograd.grad(loss, n_t)[0]
            elif noise_type == "loss_model_x":
                # print("--------loss_model_x----------")
                with torch.enable_grad():
                    x_t = cur_mean[:].detach()#.requires_grad_(True)
                    noise_model = noise_model.to("cuda")
                    n_t = y_noisy - c3 * x
                    n_t = n_t.requires_grad_(True)
                    # mu, sig = noise_model(n_t,"none" )
                    mu, sig = noise_model(n_t,g_t)
                    
                    # mu, sig = noise_model(n_t)
                    # loss = noise_model.gaussian_nll_loss(mu, sig, n_t).to("cuda")
                    try:
                        loss = noise_model.calc_model_likelihood(mu, sig, n_t, offset=False).to("cuda")###false
                    except:
                        loss = noise_model.calc_model_likelihood(mu, sig, n_t).to("cuda")
                    # loss = loss*len()
                    grad_log_p = -c3 * torch.autograd.grad(loss, n_t)[0]
            elif noise_type == "loss_model_mog":
                # print("--------loss_model_mog----------")
                with torch.enable_grad():
                    x_t = cur_mean[:].detach()#.requires_grad_(True)
                    # print("noise_model:", noise_model)
                    noise_model = noise_model.to("cuda")
                    n_t = y_noisy - c3 * x_t
                    n_t = n_t.requires_grad_(True)
                    # mu, sig = noise_model(n_t,g_t)
                    logits, mu, log_sig = noise_model(n_t, g_t)
                    try:
                        loss = noise_model.casual_loss(logits, mu, log_sig, n_t, offset=False).to("cuda")
                    except:
                        loss = noise_model.casual_loss(logits, mu, log_sig, n_t).to("cuda")
                        
                    grad_log_p = -c3 * torch.autograd.grad(loss, n_t)[0]
            elif noise_type == "loss_model_sig10":
                print("--------loss_model_sig10----------")
                with torch.enable_grad():
                    x_t = cur_mean[:].detach()#.requires_grad_(True)
                    noise_model = noise_model.to("cuda")
                    n_t = y_noisy - c3 * x_t
                    n_t = n_t.requires_grad_(True)
                    mu, sig = noise_model(n_t,"none" )
                    new_sig = calculate_rolling_std_with_means(n_t.reshape(1,-1),mu.reshape(1,-1),window_size=3)
                    
                    loss = noise_model.calc_model_likelihood(mu, new_sig, n_t).to("cuda")#.requires_grad_(True)
                    grad_log_p = -c3 * torch.autograd.grad(loss, n_t)[0]
            elif noise_type == "loss_model_double":
                print("-------- loss_model_double ----------")
                with torch.enable_grad():
                    x_t = cur_mean[:].detach()#.requires_grad_(True)
                    noise_model_low, noise_model_high = noise_model
                    noise_model_low = noise_model_low.to("cuda")
                    noise_model_high = noise_model_high.to("cuda")
                    
                    x_t_1 = x_t[:].requires_grad_(True)
                    x_t_2 = x_t[:].requires_grad_(True)
                    n_t_1 = y_noisy - c3 * x_t_1
                    n_t_2 = y_noisy - c3 * x_t_2
                    # n_t.to("cuda")
                    
                    n_t_low = fir_filter(n_t_1, high=False).to("cuda")
                    n_t_high = fir_filter(n_t_2, high=True).to("cuda") 
                    
                    def calc_grad_log_p(noise_model_, n_t_, x_t_):
                        mu, sig = noise_model_(n_t_,"none" )
                        loss = noise_model_.calc_model_likelihood(mu, sig, n_t_).to("cuda")
                        grad_log_p = torch.autograd.grad(loss, x_t_)[0]
                        return grad_log_p,loss

                    grad_log_p_low,loss_low = calc_grad_log_p(noise_model_low, n_t_low, x_t_1)
                    grad_log_p_high,loss_high = calc_grad_log_p(noise_model_high, n_t_high, x_t_2)
                    l_high = 1 - l_low
                    grad_log_p = l_low*grad_log_p_low + l_high*grad_log_p_high
                    
                    loss = (loss_low, loss_high)
                    
                    # x_t_low = fir_filter(x_t, high=False) #maybe problematic
                    # x_t_high = fir_filter(x_t, high=True) #maybe x_t instead
                    # x_t_low = x_t_low.requires_grad_(True)
                    # x_t_high = x_t_high.requires_grad_(True)
                    # y_noisy_low = fir_filter(x_t, high=False)
                    # y_noisy_high = fir_filter(x_t, high=True)
                    # n_t_low = y_noisy_low - c3 * x_t_low
                    # n_t_high = y_noisy_high - c3 * x_t_high
                    
                    # def calc_grad_log_p(noise_model_, x_t_, n_t_):
                    #     mu, sig = noise_model_(n_t_,"none" )
                    #     loss = noise_model_.calc_model_likelihood(mu, sig, n_t_).to("cuda")
                    #     grad_log_p =  torch.autograd.grad(loss, x_t_)[0]
                    #     return grad_log_p,loss

                    # grad_log_p_low,loss_low = calc_grad_log_p(noise_model_low, x_t_low,n_t_low)
                    # grad_log_p_high,loss_high = calc_grad_log_p(noise_model_high, x_t_high,n_t_high)
                    # l_high = 1 - l_low
                    # grad_log_p = l_low*grad_log_p_low + l_high*grad_log_p_high
                    
                    # loss = (loss_low, loss_high)
                    

                    
                    
                    
                    # import os
                    # # base_root = "/data/ephraim/datasets/known_noise/undiff/exp_ar_5d/"
                    # base_root = "/data/ephraim/datasets/known_noise/undiff/exp_ar_g/"

                    # tarpath = base_root+"b/{}/noise_diffusion/{}.wav".format(guid_s, diffusion_idx)
                    # root1 =  base_root+f"b/{guid_s}/noise_diffusion/"

                    # if not os.path.exists(root1):
                    #     os.mkdir(base_root+f"b/{guid_s}/")
                    #     os.mkdir(root1)
                    # if n_t.max() > 1:
                    #     n_t_normed = n_t/n_t.max()
                    # else:
                    #     n_t_normed = n_t
                    # torchaudio.save(tarpath, n_t_normed.to("cpu").view(1,-1), 16000)
            elif noise_type == "freq_gaussian":
                with torch.enable_grad():
                    n_fft,hop_length,win_length,window = noise_model["params"]
                    window = window.to("cuda")
                    mu, stds = noise_model["stats"]
                    mu = mu.to("cuda")
                    stds = stds.to("cuda")
                    
                    x_t = cur_mean[:]
                    x_t = x_t.requires_grad_(True)
                    n_t = y_noisy - c3 * x_t
                    
                    stft_nt = torch.stft(n_t.view(-1), n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)
                    # print("stft_nt:", stft_nt)
                    # magnitude_nt
                    magnitude_nt = torch.log(torch.abs(stft_nt))
                    # print("magnitude_nt:", magnitude_nt)
                    log_p = -(1/2)*(torch.square(magnitude_nt.T-mu)/torch.square(stds))+torch.log(1/(np.sqrt(2*np.pi)*stds))
                    log_p_sum = torch.sum(log_p)
                    loss = log_p_sum
                    grad_log_p = torch.autograd.grad(log_p_sum, x_t)[0]
            elif noise_type == "freq_gaussian_nolog":
                with torch.enable_grad():
                    n_fft,hop_length,win_length,window = noise_model["params"]
                    window = window.to("cuda")
                    mu, stds = noise_model["stats"]
                    mu = mu.to("cuda")
                    stds = stds.to("cuda")
                    
                    x_t = cur_mean[:]
                    x_t = x_t.requires_grad_(True)
                    n_t = y_noisy - c3 * x_t
                    
                    stft_nt = torch.stft(n_t.view(-1), n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)
                    # print("stft_nt:", stft_nt)
                    # magnitude_nt
                    magnitude_nt = (torch.abs(stft_nt))
                    # print("magnitude_nt:", magnitude_nt)
                    log_p = -(1/2)*(torch.square(magnitude_nt.T-mu)/torch.square(stds))+torch.log(1/(np.sqrt(2*np.pi)*stds))
                    log_p_sum = torch.sum(log_p)
                    loss = log_p_sum
                    grad_log_p = torch.autograd.grad(log_p_sum, x_t)[0]
            elif noise_type == "freq_gaussian_complex":
                with torch.enable_grad():
                    n_fft,hop_length,win_length,window = noise_model["params"]
                    window = window.to("cuda")
                    stds_re, stds_im,means_re,means_im = noise_model["stats"]
                    stds_re = stds_re.to("cuda")
                    stds_im = stds_im.to("cuda")
                    means_re = means_re.to("cuda")
                    means_im = means_im.to("cuda")
                    # print("stds_re[:2] ", stds_re[:2])
                    # print("means_im[:2] ", means_im[:2])
                    x_t = cur_mean[:]
                    x_t = x_t.requires_grad_(True)
                    n_t = y_noisy - c3 * x_t
                    
                    stft_nt = torch.stft(n_t.view(-1), n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)
                    eps = 0.000000000001
                    log_p_re = -(1/2)*(torch.square(stft_nt.T.real-means_re)/(torch.square(2*(stds_re/2)**2+eps)))+torch.log(1/(np.sqrt(2*np.pi)*stds_re/(2**0.5)+eps))
                    log_p_im = -(1/2)*(torch.square(stft_nt.T.imag-means_im)/(torch.square(2*(stds_im/2)**2+eps)))+torch.log(1/(np.sqrt(2*np.pi)*stds_im/(2**0.5)+eps))
                    log_p = log_p_re+log_p_im
                    # print("log_p_im[0,0]: ", log_p_im[0,0])
                    # print("log_p_re[0,0]: ", log_p_re[0,0])
                    log_p_sum = torch.mean(torch.sum(log_p, axis=0))
                    loss = log_p_sum
                    grad_log_p = torch.autograd.grad(log_p_sum, x_t)[0]
            elif noise_type == "freq_nn":
                with torch.enable_grad():
                    noise_model = noise_model.to("cuda")
                    x_t = cur_mean[:]
                    x_t = x_t.requires_grad_(True)
                    n_t = y_noisy - c3 * x_t
                    
                    stft_nt, duration, sample_rate = calc_stft(n_t.view(1,-1))
                    magnitude_nt = (torch.abs(stft_nt)).view(1,1,stft_nt.shape[0],stft_nt.shape[1]) #no log
                    
                    mu, sig = noise_model(magnitude_nt,"none" )
                    loss = noise_model.calc_model_likelihood(mu, sig, magnitude_nt).to("cuda")
                    print("t: ", t)
                    loss = loss*0.0001
                    # else:
                    #     loss = loss*0.001
                    grad_log_p = torch.autograd.grad(loss, x_t)[0]
            else: 
                grad_log_p = c3 * (y_noisy - c3 * cur_mean) / (c4 + cur_noise_var) ** 2
            cur_mean = cur_mean + cur_s * sigma_t * grad_log_p.detach()
            # root2 =  base_root+f"b/{guid_s}/x_t/"
            # if not os.path.exists(root2):
            #     os.mkdir(root2)
            # xtpath = base_root + "b/{}/x_t/{}.wav".format(guid_s,diffusion_idx)
            if True: #torch.isnan( cur_mean.max()):
                print("cur_mean.max(): ", cur_mean.max())
                
                # if cur_mean.max() > 1:
                #     cur_mean_normed = cur_mean/cur_mean.max()
                # else:
                #     cur_mean_normed = cur_mean
                # torchaudio.save(xtpath, cur_mean_normed.to("cpu").view(1,-1), 16000)

                # loss_path = base_root+f"b/{guid_s}/losses.txt"
                
                print("loss: ", loss)
                # print("grad_log_p: ", grad_log_p)
                # print("c3: ", c3)
                # print("cur_s: ", cur_s)
                print("cur_mean: ", cur_mean) #/print("gu`id_s: ", guid_s)
        sample = cur_mean + nonzero_mask * sigma_t * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"],"loss": loss}#,"loss": loss

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        resizers=None,
        range_t=0,
        cond_fn=None,
        measurement=None,
        measurement_cond_fn=None,
        sample_method=None,
        orig_x=None,
        degradation=None,
        use_rg_bwe: bool = False,
        guidance=False,
        guid_s=0,
        cur_noise_var=None,
        y_noisy=None,
        s_schedule=None,
        noise_type=None, 
        noise_model_path=None,
        l_low=0.2,
        network=None,
        mog=0,
    ):
        """
        Generate samples from the model.
        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :param orig_x:  original sample, use only in case of BWE imputation guidance, otherwise not used
                        and can be None or some dummy input
        :param sample_method: task that is solved via posterior sampling
        :param degradation: degradation that is used for given task in 'sample_method' argument
        :param use_rg_bwe:  False by default, but can be enabled
                            if we wish to combine imputation guidance with reconstruction guidance
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            resizers=resizers,
            range_t=range_t,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            measurement=measurement,
            measurement_cond_fn=measurement_cond_fn,
            sample_method=sample_method,
            orig_x=orig_x,
            degradation=degradation,
            use_rg_bwe=use_rg_bwe,
            guidance=guidance,
            guid_s=guid_s,
            cur_noise_var=cur_noise_var,
            y_noisy=y_noisy,
            s_schedule=s_schedule,
            noise_type=noise_type, 
            noise_model_path=noise_model_path,
            l_low=l_low,
            network=network,
            mog=mog,
        ):
            final = sample

        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        resizers=None,
        range_t=0,
        cond_fn=None,
        orig_x=None,
        sample_method=None,
        degradation=None,
        measurement=None,
        measurement_cond_fn=None,
        use_rg_bwe: bool = False,
        guidance=False,
        guid_s=0,
        cur_noise_var=None,
        y_noisy=None,
        s_schedule=None,
        noise_type=None, 
        noise_model_path=None,
        l_low=0.2,
        network=None,
        mog=0,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        
        if device is None:
            device = next(model.parameters()).device
        
        print("calculating y noisy")
        if y_noisy is not None:
            #for analitycal calculation
            # cur_noise_var = float(y_noisy.split("var")[1].split(".wav")[0])
            # print("cur_noise_var: ", cur_noise_var)
            print("shape: ", shape)
            
            y_noisy_, sr = torchaudio.load(y_noisy)
            shape = (1,1,y_noisy_.shape[1])  ###########shape#######
            print("shape: ", shape)
            
            y_noisy = torch.zeros(*shape)
            y_noisy[0,:,:] = y_noisy_
            y_noisy = y_noisy.to(device=device)
            print("y_noisy: ", y_noisy)
         
        
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        img.requires_grad_(False)

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        if resizers is not None:
            down, up = resizers

        if sample_method in {TaskType.BWE, TaskType.VOCODING, TaskType.DECLIPPING}:
            snr = 0.01
            xi = 0.01
        elif sample_method == TaskType.SOURCE_SEPARATION:
            snr = 0.00001
            xi = None
            

        # define corrector (as in predictor-corrector sampling)
        if sample_method == TaskType.UNCONDITIONAL:
            corrector = None
        else:
            corrector = CorrectorVPConditional(
                degradation=degradation,
                snr=snr,
                xi=xi,
                sde=self,
                score_fn=model,
                device=device,
            )

        # rg exps require gradient calculation via obtained score model
        rg_exps = {TaskType.VOCODING, TaskType.DECLIPPING}
        if use_rg_bwe:
            rg_exps.add(TaskType.BWE)


        from train_on_all_noises_wavenet import WaveNet
        
        def load_network_from_scrach(network, mog=None):
            # Dynamically get the class from globals()
            if network in globals():
                if network.endswith("MoG") and mog is not None:
                    return globals()[network](num_mixtures=mog)
                elif network == "WaveNet":
                    return WaveNet(
                        in_channels=1,
                        out_channels=2,
                        residual_channels=32,
                        skip_channels=64,
                        kernel_size=3,
                        dilation_depth=8,
                        num_stacks=3
                    )
                else:
                    return globals()[network]()
            else:
                raise ValueError(f"Unknown network: {network}")
            
        def load_network(network,state_dict, mog=None):
            # Dynamically get the class from globals()
            if network in globals():
                if network.endswith("MoG") and mog is not None:
                    model_ =  globals()[network](num_mixtures=mog)
                    model_.load_state_dict(state_dict)
                    return model_
                elif network == "WaveNet":
                    model_ =  WaveNet(
                        in_channels=1,
                        out_channels=2,
                        residual_channels=32,
                        skip_channels=64,
                        kernel_size=3,
                        dilation_depth=8,
                        num_stacks=3
                    )
                    model_.load_state_dict(state_dict)
                    return model_
                else:
                    model_ =  globals()[network]()
                    model_.load_state_dict(state_dict)
                    return model_
            else:
                raise ValueError(f"Unknown network: {network}")
            
        import io
        single_noise_model=False
        class CPU_Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'torch.storage' and name == '_load_from_bytes':
                    return lambda b: torch.load(io.BytesIO(b), map_location='cuda:0')
                else: return super().find_class(module, name)
        
        if noise_model_path is not None and network is None:
            if noise_type=="loss_model" or noise_type=="y_model" or noise_type=="loss_model_x" or noise_type=="loss_model_mog" or noise_type=="freq_nn" or noise_type=="loss_model_sig10":
                with open(noise_model_path, 'rb') as handle:
                    params_dict =  CPU_Unpickler(handle).load()
                    print("noise_model_path: ", noise_model_path)
                    try:
                        network = params_dict["network"]
                        noise_models = [load_network(network,params_dict["nets"][k], mog=mog) for k in range(len(params_dict["nets"]))] 
                    except:
                        raise Exception("Error: network not found in params_dict")
                        noise_models = params_dict["nets"] 
            elif noise_type=="loss_model_double":
                with open(noise_model_path, 'rb') as handle:
                    params_dict =  CPU_Unpickler(handle).load()
                    print("noise_model_path: ", noise_model_path)
                    try:
                        network = params_dict["network"]
                        # noise_models_low = [load_network(network, mog=mog) for _ in range(len(params_dict["nets_low"]))] 
                        # noise_models_high = [load_network(network, mog=mog) for _ in range(len(params_dict["nets_high"]))] 
                        noise_models_low = [load_network(network,params_dict["nets"][k], mog=mog) for k in range(len(params_dict["nets_low"]))] 
                        noise_models_high = [load_network(network,params_dict["nets"][k], mog=mog) for k in range(len(params_dict["nets_high"]))] 
                    except:
                        noise_models_low = params_dict["nets_low"] 
                        noise_models_high = params_dict["nets_high"] 
                    noise_models = (noise_models_low,noise_models_high)
            elif noise_type in ["freq_gaussian","freq_gaussian_complex","freq_gaussian_nolog"]:
                with open(noise_model_path, 'rb') as handle:
                    params_dict = pickle.load(handle)
                    try:
                        network = params_dict["network"]
                        # noise_models = [load_network(network, mog=mog) for _ in range(len(params_dict["models"]))] 
                        noise_models = [load_network(network,params_dict["nets"][k], mog=mog) for k in range(len(params_dict["models"]))] 
                    except:
                        noise_models = params_dict["models"] 
        elif network is not None:
            # from create_exp_m import NetworkNoise8,NetworkNoise7, NetworkNoise6,NetworkNoise6d,NetworkNoise9,NetworkNoise6c,NetworkNoise6b,NetworkNoise5,NetworkNoise4, NetworkNoise3, WaveNetCausalModel, NetworkNoiseWaveNetMoG, NetworkNoise6MoG, NetworkNoise4MoG

            
            noise_model = load_network_from_scrach(network,mog)

            noise_model.load_state_dict(torch.load(noise_model_path, map_location=device))
            noise_model.eval()
            noise_model.to(device)
            single_noise_model = True
            
        
        # xt_history={}
        loss_array=[]
        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            if network is not None:
                # if "noise_model" in locals():
                if single_noise_model:
                    noise_model = noise_model
                else:
                    noise_model = noise_models[i]
            elif  (noise_model_path is not None and network is None) and noise_type!="loss_model_double":
                noise_model = noise_models[i]
                if noise_type=="loss_model" or noise_type=="y_model" or noise_type=="loss_model_x" or  noise_type=="loss_model_mog" or noise_type=="loss_model_sig10":
                    noise_model = noise_model.to(device)
            else:
                noise_models_low,noise_models_high = noise_models
                noise_model_low = noise_models_low[i].to(device)
                noise_model_high = noise_models_high[i].to(device)
                noise_model = (noise_model_low,noise_model_high)
            
            if sample_method in rg_exps:
                img.requires_grad_(True)
                if i != 199:
                    y = degradation(orig_x)
                    img = corrector.update_fn_adaptive(
                        None, img, t, y, threshold=200, steps=1, source_separation=False
                    )
            
            with th.no_grad():  #######
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    degradation=degradation if sample_method == "BWE" else None,
                    orig_x=orig_x,
                    guidance=guidance,
                    guid_s=guid_s,
                    cur_noise_var=cur_noise_var,
                    y_noisy=y_noisy,
                    s_schedule=s_schedule,
                    noise_type=noise_type, 
                    noise_model=noise_model,
                    diffusion_idx=i,
                    l_low=l_low 
                )
            # xt_history[i] = out["sample"]
            loss_array.append(str(out["loss"]))
            
            if sample_method == TaskType.SOURCE_SEPARATION:
                y = degradation(orig_x)
                img = corrector.update_fn_adaptive(
                    out, img, t, y, threshold=150, steps=2, source_separation=True
                )
                out["sample"] = img

            yield out
            img = out["sample"]
        path_pickle =  noise_model_path.replace("models", ("_"+str(s_schedule)+"_"+str(guid_s)))#+"_"
        print(path_pickle)
        if path_pickle != noise_model_path:
            with open(path_pickle, 'wb') as handle:
                pickle.dump(loss_array, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print("failed dumping loss array, bad path")
        
    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.
        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma**2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.
        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.
        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.
        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.
        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.
        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        # assert (cond is not None if self.p_uncond < 1.0), 'Need condition for classifier-free guidance training'

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            if self.input_sigma_t:
                model_output = model(
                    x_t, _extract_into_tensor(self.betas, t, t.shape), **model_kwargs
                )
            else:
                model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                print(x_t.shape)
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert (
                model_output.shape == target.shape == x_start.shape
            ), f"model_output.shape: {model_output.shape}, target.shape: {target.shape}, x_start.shape: {x_start.shape}"
            terms["mse"] = mean_flat((target - model_output) ** 2)
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]

        elif self.loss_type == LossType.L1:
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)
            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert (
                model_output.shape == target.shape == x_start.shape
            ), f"model_output.shape: {model_output.shape}, target.shape: {target.shape}, x_start.shape: {x_start.shape}"
            terms["l1"] = mean_flat(th.abs(target - model_output))
            terms["loss"] = terms["l1"]

        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.
        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


class CorrectorVPConditional:
    def __init__(self, degradation, xi, sde, snr, score_fn, device):
        self.degradation = degradation
        self.xi = xi
        self.alphas = torch.from_numpy(sde.alphas).to(device)
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.recip_alphas = torch.from_numpy(1 / sde.sqrt_one_minus_alphas_cumprod).to(
            device
        )

    def update_fn_adaptive(
        self, x, x_prev, t, y, threshold=150, steps=1, source_separation=False
    ):
        x, condition = self.update_fn(x, x_prev, t, y, steps, source_separation)

        if t[0] < threshold and t[0] > 0:
            if self.sde.input_sigma_t:
                eps = self.score_fn(
                    x, _extract_into_tensor(self.sde.beta_variance, t, t.shape)
                )
            else:
                eps = self.score_fn(x, self.sde._scale_timesteps(t))

            if condition is None:
                condition = y - (x[:, :, : y.size(-1)] + x[:, :, y.size(-1) :])
            if source_separation:
                x = self.langevin_corrector_sliced(x, t, eps, y, condition)
            else:
                x = self.langevin_corrector(x, t, eps, y, condition)

        return x

    def update_fn(self, x, x_prev, t, y, steps, source_separation):
        if source_separation:
            coefficient = 0.5
            total_log_sum = 0

            for i in range(steps):
                new_samples = []
                log_p_y_x = y - (
                    x_prev[:, :, : y.size(-1)] + x_prev[:, :, y.size(-1) :]
                )
                total_log_sum += log_p_y_x
                start = 0
                end = y.size(-1)
                while end <= x_prev.size(-1):
                    new_sample = (
                        x["sample"][:, :, start:end] + coefficient * total_log_sum
                    )
                    new_samples.append(new_sample)
                    start = end
                    end += y.size(-1)
                x_prev = torch.cat(new_samples, dim=-1)
            condition = None

        else:
            for i in range(steps):
                if self.sde.input_sigma_t:
                    eps = self.score_fn(
                        x_prev, _extract_into_tensor(self.sde.beta_variance, t, t.shape)
                    )
                else:
                    eps = self.score_fn(x_prev, self.sde._scale_timesteps(t))

                x_0 = self.sde._predict_xstart_from_eps(x_prev, t, eps)
                A_x0 = self.degradation(x_0)

                if len(y.shape) < 3 and len(A_x0.shape) < 3:
                    while len(y.shape) != 3:
                        y = y.unsqueeze(0)
                        A_x0 = A_x0.unsqueeze(0) - 1e-3

                rec_norm = torch.linalg.norm(
                    (y - A_x0).view(y.size(0), -1), dim=-1, ord=2
                ).mean()
                condition = torch.autograd.grad(outputs=rec_norm, inputs=x_prev)[0]

                normguide = torch.linalg.norm(condition) / (x_0.size(-1) ** 0.5)
                sigma = torch.sqrt(self.alphas[t])
                s = self.xi / (normguide * sigma + 1e-6)

                x_prev = x_prev - s * condition
        return x_prev.float(), condition

    def langevin_corrector_sliced(self, x, t, eps, y, condition=None):
        alpha = self.alphas[t]
        corrected_samples = []

        start = 0
        end = y.size(-1)
        while end <= x.size(-1):
            score = self.recip_alphas[t] * eps[:, :, start:end]
            noise = torch.randn_like(x[:, :, start:end], device=x.device)
            grad_norm = torch.norm(score.reshape(score.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (self.snr * noise_norm / grad_norm) ** 2 * 2 * alpha

            score_to_use = score + condition if condition is not None else score
            x_new = (
                x[:, :, start:end]
                + step_size * score_to_use
                + torch.sqrt(2 * step_size) * noise
            )
            corrected_samples.append(x_new)
            start = end
            end += y.size(-1)

        return torch.cat(corrected_samples, dim=-1).float()

    def langevin_corrector(self, x, t, eps, y, condition=None):
        alpha = self.alphas[t]

        score = self.recip_alphas[t] * eps
        noise = torch.randn_like(x, device=x.device)
        grad_norm = torch.norm(score.reshape(score.shape[0], -1), dim=-1).mean()
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        step_size = (self.snr * noise_norm / grad_norm) ** 2 * 2 * alpha

        score_to_use = score + condition if condition is not None else score
        x_new = x + step_size * score_to_use + torch.sqrt(2 * step_size) * noise

        return x_new.float()

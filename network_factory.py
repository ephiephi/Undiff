import torch
import torch.nn as nn
import numpy as np
import math






class CausalConv1dClassS(nn.Conv1d):
    def __init__(self,in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        pad = (kernel_size - 1) * dilation
        super().__init__(in_channels, out_channels, kernel_size, padding=pad, dilation=dilation, **kwargs)
    
    def forward(self, inputs):
        output = super().forward(inputs)
        if self.padding[0] != 0:
            output = output[:, :, :-self.padding[0]]
        return output

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Assuming CausalConv1dClassS is defined elsewhere

class NetworkNoise6bMoG(nn.Module):
    def __init__(self, kernel_size=9, num_channels=8, num_mixtures=1, dilation_pattern=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_channels = num_channels
        n_mixtures = num_mixtures
        self.n_mixtures = num_mixtures
        
        if dilation_pattern is None:
            dilation_pattern = [1, 2, 4, 8, 1, 2, 4, 8]
        self.dilation_pattern = dilation_pattern
        
        self.blocks = nn.ModuleList()
        in_channels = 1  # Input channels (single waveform channel)
        for dilation in dilation_pattern:
            block = nn.Sequential(
                CausalConv1dClassS(in_channels, num_channels, kernel_size=kernel_size, dilation=dilation),
                nn.Tanh()
            )
            self.blocks.append(block)
            in_channels = num_channels
        
        # Three heads: one for means, one for log-variance and one for mixture logits
        self.conv_means = nn.Conv1d(num_channels, n_mixtures, kernel_size=1)
        self.conv_log_var = nn.Conv1d(num_channels, n_mixtures, kernel_size=1)
        self.conv_logits = nn.Conv1d(num_channels, n_mixtures, kernel_size=1)
        
        self.receptive_field = self.calculate_receptive_field()

    def forward(self, x, cur_gt=None):
        residuals = x
        skip_connections = 0
        
        for block in self.blocks:
            out = block(residuals)
            skip_connections = skip_connections + out
            residuals = out

        # Predict mixture logits, means, and log variances.
        mixture_logits = self.conv_logits(skip_connections)
        # Use log_softmax so that we get log probabilities that sum to 1 over mixtures.
        mixture_log_probs = F.log_softmax(mixture_logits, dim=1)
        
        means = self.conv_means(skip_connections)
        log_var = self.conv_log_var(skip_connections)
        stds = torch.exp(0.5 * log_var)
        
        # The shapes of mixture_log_probs, means, and stds are (batch, n_mixtures, time)
        return mixture_log_probs, means, stds

    def calc_model_likelihood(self, mixture_log_probs, means, stds, wav_tensor, verbose=False, offset=True):
        # Use offset to ignore initial parts not covered by the receptive field.
        offset_value = self.receptive_field if offset else 0
        
        # Adjust target waveform: assuming wav_tensor shape is (batch, 1, time)
        wav_tensor = wav_tensor.squeeze(1)[:, offset_value+1:]
        means_ = means[:, :, offset_value:-1]
        stds_ = stds[:, :, offset_value:-1]
        mix_log_probs_ = mixture_log_probs[:, :, offset_value:-1]
        
        # Expand wav_tensor to shape (batch, 1, time) for broadcasting.
        wav_tensor = wav_tensor.unsqueeze(1)
        
        # Compute the log probability for each mixture component
        # Log Gaussian density: -0.5*log(2π) - log(std) - 0.5*((x-mean)/std)**2
        log_pdf = (
            -0.5 * math.log(2 * math.pi)
            - torch.log(stds_)
            - 0.5 * ((wav_tensor - means_) / stds_)**2
        )
        # Add the mixture log probabilities.
        log_prob_components = mix_log_probs_ + log_pdf
        
        # Use log-sum-exp over the mixture components dimension (dim=1)
        log_prob_mix = torch.logsumexp(log_prob_components, dim=1)
        
        # Average log likelihood over time and batch
        likelihood = log_prob_mix.mean()
        if verbose:
            print("Mean log likelihood:", likelihood.item())
        return likelihood

    def casual_loss(self, mixture_log_probs, means, stds, wav_tensor, offset=True):
        model_likelihood = self.calc_model_likelihood(mixture_log_probs, means, stds, wav_tensor, offset=offset)
        return -model_likelihood   

    def calculate_receptive_field(self):
        """
        Computes the receptive field for a stack of causal convolution blocks,
        summing (kernel_size - 1) * dilation for each block.
        """
        rf = 1
        k_minus_1 = self.kernel_size - 1
        for d in self.dilation_pattern:
            rf += k_minus_1 * d
        return rf


class NetworkNoise6cMoG(nn.Module):
    def __init__(self, kernel_size=9, num_channels=8, num_mixtures=1, dilation_pattern=None):
        super().__init__()
        n_mixtures = num_mixtures
        self.kernel_size = kernel_size
        self.num_channels = num_channels
        self.n_mixtures = n_mixtures
        
        if dilation_pattern is None:
            dilation_pattern = [1, 2, 4, 8, 1, 2, 4, 8]
        self.dilation_pattern = dilation_pattern
        
        self.blocks = nn.ModuleList()
        in_channels = 1  # Input channels (single waveform channel)
        for dilation in dilation_pattern:
            block = nn.Sequential(
                CausalConv1dClassS(in_channels, num_channels, kernel_size=kernel_size, dilation=dilation),
                nn.Tanh()
            )
            self.blocks.append(block)
            in_channels = num_channels
        
        # Three heads: one for means, one for log-variance, and one for mixture logits.
        self.conv_means = nn.Conv1d(num_channels, n_mixtures, kernel_size=1)
        self.conv_log_var = nn.Conv1d(num_channels, n_mixtures, kernel_size=1)
        self.conv_logits = nn.Conv1d(num_channels, n_mixtures, kernel_size=1)
        
        self.receptive_field = self.calculate_receptive_field()

    def forward(self, x, cur_gt=None):
        residuals = x
        skip_connections = 0
        for block in self.blocks:
            out = block(residuals)
            skip_connections = skip_connections + out
            residuals = out

        # Predict mixture logits, means, and log variances.
        mixture_logits = self.conv_logits(skip_connections)
        # Log softmax to get log probabilities over mixtures.
        mixture_log_probs = F.log_softmax(mixture_logits, dim=1)
        
        means = self.conv_means(skip_connections)
        log_var = self.conv_log_var(skip_connections)
        stds = torch.exp(0.5 * log_var)
        
        # All outputs are of shape (batch, n_mixtures, time)
        return mixture_log_probs, means, stds

    def calc_model_likelihood(self, mixture_log_probs, means, stds, wav_tensor, verbose=False, offset=True):
        # Determine offset to account for the receptive field.
        offset_value = self.receptive_field if offset else 0

        # Adjust target waveform: assume wav_tensor shape is (batch, 1, time)
        # Squeeze channel and slice to match valid predictions.
        wav_tensor = wav_tensor.squeeze(1)[:, offset_value+1:]
        means_ = means[:, :, offset_value:-1]
        stds_ = stds[:, :, offset_value:-1]
        mix_log_probs_ = mixture_log_probs[:, :, offset_value:-1]
        
        # Expand wav_tensor to shape (batch, 1, time) for broadcasting.
        wav_tensor = wav_tensor.unsqueeze(1)
        
        # Compute the log Gaussian density per component:
        # log_pdf = -0.5*log(2*pi) - log(std) - 0.5*((x-mean)/std)^2
        log_pdf = (
            -0.5 * math.log(2 * math.pi)
            - torch.log(stds_)
            - 0.5 * ((wav_tensor - means_) / stds_)**2
        )
        # Combine with mixture log weights.
        log_prob_components = mix_log_probs_ + log_pdf  # Shape: (batch, n_mixtures, time)
        
        # For each time step, marginalize over the mixtures.
        log_prob_mix = torch.logsumexp(log_prob_components, dim=1)  # Shape: (batch, time)
        
        # Sum log likelihoods over time steps, then average over the batch.
        likelihood = log_prob_mix.sum(dim=-1)  # Sum over time.
        if verbose:
            print("Likelihood per sample:", likelihood)
        return likelihood.mean()

    def casual_loss(self, mixture_log_probs, means, stds, wav_tensor, offset=True):
        model_likelihood = self.calc_model_likelihood(mixture_log_probs, means, stds, wav_tensor, offset=offset)
        return -model_likelihood   

    def calculate_receptive_field(self):
        """
        Computes the receptive field for a stack of causal convolution blocks.
        """
        rf = 1
        k_minus_1 = self.kernel_size - 1
        for d in self.dilation_pattern:
            rf += k_minus_1 * d
        return rf



class NetworkNoise6dMoG(nn.Module):
    def __init__(self, kernel_size=9, num_channels=8, num_mixtures=1, dilation_pattern=None):
        """
        When n_mixtures == 1, the network behaves exactly like your original model.
        When n_mixtures > 1, the network outputs parameters for a mixture of Gaussians.
        """
        super().__init__()
        n_mixtures = num_mixtures
        self.kernel_size = kernel_size
        self.num_channels = num_channels
        self.n_mixtures = n_mixtures

        if dilation_pattern is None:
            dilation_pattern = [1, 2, 4, 8, 1, 2, 4, 8]
        self.dilation_pattern = dilation_pattern

        # Build the causal convolution blocks.
        self.blocks = nn.ModuleList()
        in_channels = 1  # single-channel input
        for dilation in self.dilation_pattern:
            block = nn.Sequential(
                CausalConv1dClassS(in_channels, num_channels, kernel_size=kernel_size, dilation=dilation),
                nn.Tanh()
            )
            self.blocks.append(block)
            in_channels = num_channels

        # For both single and mixture cases, define conv_mean and conv_log_var.
        # If n_mixtures == 1, these output 1 channel (matching the original).
        # Otherwise, they output n_mixtures channels.
        self.conv_mean    = nn.Conv1d(num_channels, self.n_mixtures, kernel_size=1)
        self.conv_log_var = nn.Conv1d(num_channels, self.n_mixtures, kernel_size=1)
        
        # Only needed for mixture (n_mixtures > 1).
        if self.n_mixtures > 1:
            self.conv_weight = nn.Conv1d(num_channels, self.n_mixtures, kernel_size=1)
        
        self.receptive_field = self.calculate_receptive_field()

    def forward(self, x, cur_gt=None):
        """
        Input: x of shape [batch, 1, time]
        Returns:
          - mixture_log_probs: [batch, n_mixtures, time]
          - means:             [batch, n_mixtures, time]
          - stds:              [batch, n_mixtures, time]
        For n_mixtures==1, mixture_log_probs is filled with zeros.
        """
        residuals = x
        skip_connections = 0
        for block in self.blocks:
            out = block(residuals)
            skip_connections = skip_connections + out
            residuals = out

        means   = self.conv_mean(skip_connections)      # shape: [B, n_mixtures, time]
        log_var = self.conv_log_var(skip_connections)     # shape: [B, n_mixtures, time]
        stds    = torch.exp(0.5 * log_var)                 # shape: [B, n_mixtures, time]

        if self.n_mixtures > 1:
            weights_logits = self.conv_weight(skip_connections)  # [B, n_mixtures, time]
            mixture_log_probs = F.log_softmax(weights_logits, dim=1)
        else:
            # For a single Gaussian, set mixture_log_probs to zero.
            mixture_log_probs = torch.zeros_like(means)

        return mixture_log_probs, means, stds

    def calc_model_likelihood(self,  mixture_log_probs, means, stds, wav_tensor, verbose=False, offset=True):
        """
        Unified likelihood calculation for both single Gaussian and mixture of Gaussians.
        Uses the formula:
        
          log p(x) = logsumexp_k ( log pi_k + log N(x | mu_k, sigma_k) )
        
        When n_mixtures==1, mixture_log_probs are zeros so this reduces to the original:
        
          log p(x) = log N(x | mu, sigma)
        
        Slices the time dimension to account for the receptive field.
        Assumes wav_tensor is of shape [B, 1, time].
        """
        # mixture_log_probs, means, stds = output  # shapes: [B, n_mixtures, time]
        offset_value = self.receptive_field if offset else 0

        # Slice target and predictions to match valid outputs.
        target = wav_tensor.squeeze(1)[:, offset_value+1:]  # [B, time_valid]
        means = means[:, :, offset_value:-1]                # [B, n_mixtures, time_valid]
        stds  = stds[:, :, offset_value:-1]
        mixture_log_probs = mixture_log_probs[:, :, offset_value:-1]

        # Expand target for broadcasting.
        target_expanded = target.unsqueeze(1)  # [B, 1, time_valid]
        # Compute per-component Gaussian log-density:
        log_pdf = (-0.5 * math.log(2 * np.pi)
                   - torch.log(stds)
                   - 0.5 * ((target_expanded - means) / stds) ** 2)
        # Add the mixture log probabilities.
        log_components = mixture_log_probs + log_pdf  # [B, n_mixtures, time_valid]
        # Marginalize over mixture components.
        log_prob = torch.logsumexp(log_components, dim=1)  # [B, time_valid]
        # Sum over time and average over batch.
        log_prob = log_prob.sum(dim=-1)  # [B]
        if verbose:
            print("Unified log-likelihood per sample:", log_prob)
        return log_prob.mean()

    def casual_loss(self,  mixture_log_probs, means, stds, wav_tensor, offset=True):
        """
        Returns the negative log likelihood (averaged over the batch).
        """
        ll = self.calc_model_likelihood( mixture_log_probs, means, stds, wav_tensor, offset=offset)
        return -ll

    def calculate_receptive_field(self):
        """
        Computes the receptive field from the causal convolution blocks.
        """
        rf = 1
        k_minus_1 = self.kernel_size - 1
        for d in self.dilation_pattern:
            rf += k_minus_1 * d
        return rf


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
    


class NetworkNoise3_1(nn.Module):
    def __init__(self, kernel_size=9):
        super().__init__()
        self.kernel_size=kernel_size
        self.conv1 = CausalConv1dClassS(1, 2, kernel_size=kernel_size, dilation=1)
        self.tanh1 = nn.Tanh()
        self.conv2 = CausalConv1dClassS(2, 2, kernel_size=kernel_size, dilation=2)
        self.tanh2 = nn.Tanh()
        self.conv3 = CausalConv1dClassS(2, 2, kernel_size=kernel_size, dilation=4)
        # self.b = nn.Parameter(torch.tensor(0.5))  # Initial value of 'b'


    def forward(self, x, cur_gt):

        x1 = self.conv1(x)
        x = self.tanh1(x1)
        x = self.conv2(x)
        x = self.tanh2(x)
        x = self.conv3(x)
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
    
    
    
    








class NetworkNoise3_7MoG(nn.Module):
    def __init__(self, kernel_size=9, num_mixtures=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.n_mixtures = num_mixtures

        # Convolutional Blocks with Gated Activation
        self.conv1 = nn.utils.parametrizations.weight_norm(
            CausalConv1dClassS(1, 2, kernel_size=kernel_size, dilation=1)
        )
        self.gate1 = nn.Conv1d(2, 2, kernel_size=1)

        self.conv2 = nn.utils.parametrizations.weight_norm(
            CausalConv1dClassS(2, 2, kernel_size=kernel_size, dilation=2)
        )
        self.gate2 = nn.Conv1d(2, 2, kernel_size=1)

        self.conv3 = nn.utils.parametrizations.weight_norm(
            CausalConv1dClassS(2, 2, kernel_size=kernel_size, dilation=4)
        )
        self.gate3 = nn.Conv1d(2, 2, kernel_size=1)

        self.conv4 = nn.utils.parametrizations.weight_norm(
            CausalConv1dClassS(2, 2, kernel_size=kernel_size, dilation=8)
        )
        self.gate4 = nn.Conv1d(2, 2, kernel_size=1)

        # Heads for Gaussian parameters
        self.conv_mean = nn.Conv1d(2, num_mixtures, kernel_size=1)
        self.conv_log_var = nn.Conv1d(2, num_mixtures, kernel_size=1)
        self.conv_log_probs = nn.Conv1d(2, num_mixtures, kernel_size=1)

    def gated_activation(self, x, gate):
        return torch.tanh(x) * torch.sigmoid(gate)

    def forward(self, x, cur_gt):
        x1 = self.conv1(x)
        x1 = self.gated_activation(x1, self.gate1(x1))

        x = self.conv2(x1)
        x = self.gated_activation(x, self.gate2(x))

        x = self.conv3(x)
        x = self.gated_activation(x, self.gate3(x))

        x = self.conv4(x)
        x = self.gated_activation(x, self.gate4(x))

        x = x + x1  # skip connection

        if self.n_mixtures == 1:
            # For single Gaussian, use channels: 0 for mean, 1 for log variance.
            means = self.conv_mean(x).squeeze(1)      # [B, T]
            log_var = self.conv_log_var(x).squeeze(1)   # [B, T]
            stds = torch.exp(0.5 * log_var)
            mixture_log_probs = torch.zeros_like(means)  # [B, T]
        else:
            means = self.conv_mean(x)        # [B, n_mixtures, T]
            log_var = self.conv_log_var(x)   # [B, n_mixtures, T]
            stds = torch.exp(0.5 * log_var)
            weights_logits = self.conv_log_probs(x)  # [B, n_mixtures, T]
            mixture_log_probs = F.log_softmax(weights_logits, dim=1)

        return mixture_log_probs, means, stds

    def calc_model_likelihood(self, mixture_log_probs, means, stds, wav_tensor, verbose=False):
        """
        Computes the log likelihood using a unified formula:

            log p(x) = logsumexp_k ( log(pi_k) + log N(x|mu_k,sigma_k) )

        When n_mixtures==1, mixture_log_probs are zeros so this reduces to the original single-Gaussian likelihood.

        Slices the time dimension similarly to your original NetworkNoise3:
          - target: from index (kernel_size+1) onward.
          - predictions: from index kernel_size to -1.

        Assumes wav_tensor shape [B, 1, T].
        """
        # Slice along the last dimension using ellipsis.
        target = wav_tensor.squeeze(1)[..., self.kernel_size+1:]   # [B, T_valid]
        means = means[..., self.kernel_size:-1]                     # now slices the time dimension
        stds  = stds[..., self.kernel_size:-1]
        mixture_log_probs = mixture_log_probs[..., self.kernel_size:-1]

        # Expand target for broadcasting: [B, 1, T_valid]
        target_exp = target.unsqueeze(1)
        
        # Compute per-component Gaussian log-density.
        log_pdf = (-0.5 * math.log(2 * np.pi)
                   - torch.log(stds)
                   - 0.5 * ((target_exp - means) / stds) ** 2)
        # Add mixture log weights.
        log_components = mixture_log_probs + log_pdf  # [B, n_mixtures, T_valid] or [B, T_valid] if single Gaussian
        # Marginalize over mixture components if applicable.
        if self.n_mixtures == 1:
            log_prob = log_components  # Already the correct log-density.
        else:
            log_prob = torch.logsumexp(log_components, dim=1)  # [B, T_valid]
        
        # Sum over time steps and average over batch.
        log_prob = log_prob.sum(dim=-1)  # [B]
        if verbose:
            print("Unified log-likelihood per sample:", log_prob)
        return log_prob.mean()

    def casual_loss(self, mixture_log_probs, expected_means, expected_stds, wav_tensor):
        """
        Returns the negative log likelihood loss averaged over the batch.
        """
        ll = self.calc_model_likelihood(mixture_log_probs, expected_means, expected_stds, wav_tensor)
        return -ll



class NetworkNoise3MoG(nn.Module):
    def __init__(self, kernel_size=9, num_mixtures=1):
        """
        When n_mixtures == 1, the network reproduces your original NetworkNoise3.
        When n_mixtures > 1, the network outputs parameters for a mixture of Gaussians.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.n_mixtures = num_mixtures
        
        # Original network layers.
        self.conv1 = CausalConv1dClassS(1, 3, kernel_size=kernel_size, dilation=1)
        self.tanh1 = nn.Tanh()
        self.conv2 = CausalConv1dClassS(3, 3, kernel_size=kernel_size, dilation=2)
        self.tanh2 = nn.Tanh()
        self.conv3 = CausalConv1dClassS(3, 3, kernel_size=kernel_size, dilation=4)
        self.tanh3 = nn.Tanh()
        self.conv4 = CausalConv1dClassS(3, 3, kernel_size=kernel_size, dilation=8)
        # In the original network, after conv4 the output is added to the output of conv1.
        
        # # For the mixture case, we add separate 1x1 heads on the final hidden representation.
        # if self.n_mixtures > 1:
        #     # These heads take the hidden representation of size 2 (channels) and produce n_mixtures outputs.
        #     self.head_mean     = nn.Conv1d(2, n_mixtures, kernel_size=1)
        #     self.head_log_var  = nn.Conv1d(2, n_mixtures, kernel_size=1)
        #     self.head_weight   = nn.Conv1d(2, n_mixtures, kernel_size=1)
        
        # For consistency in likelihood slicing.
        # (The original network used indices [kernel_size+1:] for target and [kernel_size:-1] for predictions.)
        self.valid_start = self.kernel_size  # start index for predictions (as in original)
    
    def forward(self, x, cur_gt=None):
        """
        Input:
          - x: [B, 1, T]
        Returns a tuple of three tensors:
          - mixture_log_probs: [B, n_mixtures, T]
          - means:             [B, n_mixtures, T]
          - stds:              [B, n_mixtures, T]
        For n_mixtures==1, mixture_log_probs is zeros and means, stds are obtained by splitting the two channels.
        """
        # Pass through original layers.
        x1 = self.conv1(x)         # shape: [B, 2, T]
        h = self.tanh1(x1)
        h = self.conv2(h)          # shape: [B, 2, T]
        h = self.tanh2(h)
        h = self.conv3(h)          # shape: [B, 2, T]
        h = self.tanh3(h)
        h = self.conv4(h)          # shape: [B, 2, T]
        h = h + x1                 # skip connection, as in original
        
        
        means   = h[:, 0:1, :]           # shape: [B, 1, T]
        log_var = h[:, 1:2, :]           # shape: [B, 1, T]
        stds    = torch.exp(0.5 * log_var)
        if self.n_mixtures == 1:
            mixture_log_probs = torch.zeros_like(means)  # [B, 1, T]
        else:
            weights_logits = h[:, 2:3, :]  # shape: [B, n_mixtures, T]
            mixture_log_probs = F.log_softmax(weights_logits, dim=1)
        
        return mixture_log_probs, means, stds

    def calc_model_likelihood(self, mixture_log_probs,means, stds, wav_tensor, verbose=False):
        """
        Computes the log likelihood using a unified formula:
        
            log p(x) = logsumexp_k ( log(pi_k) + log N(x|mu_k,sigma_k) )
        
        When n_mixtures==1, mixture_log_probs are zeros so this reduces to the original single-Gaussian likelihood.
        
        Slices the time dimension similarly to your original NetworkNoise3:
          - target: from index (kernel_size+1) onward.
          - predictions: from index kernel_size to -1.
        
        Assumes wav_tensor shape [B, 1, T].
        """
        # mixture_log_probs, means, stds = output  # shapes: [B, n_mixtures, T]
        
        # Slicing indices as in your original network.
        # Original: wav_tensor = wav_tensor.squeeze(axis=1)[:, self.kernel_size+1:]
        #           means = expected_means.squeeze(axis=1)[:, self.kernel_size:-1]
        target = wav_tensor.squeeze(1)[:, self.kernel_size+1:]  # [B, T_valid]
        means = means[:, :, self.kernel_size:-1]                # [B, n_mixtures, T_valid]
        stds  = stds[:, :, self.kernel_size:-1]
        mixture_log_probs = mixture_log_probs[:, :, self.kernel_size:-1]
        
        # Expand target for broadcasting: [B, 1, T_valid]
        target_exp = target.unsqueeze(1)
        
        # Compute per-component Gaussian log-density:
        log_pdf = (-0.5 * math.log(2 * np.pi)
                   - torch.log(stds)
                   - 0.5 * ((target_exp - means) / stds) ** 2)
        # Add mixture log weights.
        log_components = mixture_log_probs + log_pdf  # [B, n_mixtures, T_valid]
        # Marginalize over mixture components.
        log_prob = torch.logsumexp(log_components, dim=1)  # [B, T_valid]
        # Sum over time steps.
        log_prob = log_prob.sum(dim=-1)  # [B]
        if verbose:
            print("Unified log-likelihood per sample:", log_prob)
        return log_prob.mean()
    
    def casual_loss(self, mixture_log_probs,expected_means, expected_stds, wav_tensor):
        """
        Returns the negative log likelihood loss averaged over the batch.
        """
        ll = self.calc_model_likelihood(mixture_log_probs,expected_means, expected_stds, wav_tensor)
    
        return -ll


class NetworkNoise3_2(nn.Module):
    def __init__(self, kernel_size=9):
        super().__init__()
        self.kernel_size=kernel_size
        self.conv1 = CausalConv1dClassS(1, 2, kernel_size=kernel_size, dilation=1)
        self.tanh1 = nn.Tanh()
        self.conv2 = CausalConv1dClassS(2, 2, kernel_size=kernel_size, dilation=2)
        self.tanh2 = nn.Tanh()
        self.conv3 = CausalConv1dClassS(2, 2, kernel_size=kernel_size, dilation=2)
        self.tanh3 = nn.Tanh()
        self.conv4 = CausalConv1dClassS(2, 2, kernel_size=kernel_size, dilation=2)
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



class NetworkNoise3_4(nn.Module):
    def __init__(self, kernel_size=9):
        super().__init__()
        self.kernel_size=kernel_size
        self.conv1 = CausalConv1dClassS(1, 2, kernel_size=kernel_size, dilation=1)
        self.tanh1 = nn.Tanh()
        self.conv2 = CausalConv1dClassS(2, 4, kernel_size=kernel_size, dilation=2)
        self.tanh2 = nn.Tanh()
        self.conv3 = CausalConv1dClassS(4, 4, kernel_size=kernel_size, dilation=4)
        self.tanh3 = nn.Tanh()
        self.conv4 = CausalConv1dClassS(4, 2, kernel_size=kernel_size, dilation=8)
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


class NetworkNoise3_3(nn.Module):
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
        self.receptive_field = self.compute_receptive_field()


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
        wav_tensor = wav_tensor.squeeze(axis=1)[:,self.receptive_field+1:]
        means_=expected_means.squeeze(axis=1)[:,self.receptive_field:-1]
        stds_ = expected_stds.squeeze(axis=1)[:,self.receptive_field:-1]

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
    
    def compute_receptive_field(model):
        R = 1  # Start by counting the current time-step itself
        for layer in model.modules():
            if isinstance(layer, nn.Conv1d):
                # layer.kernel_size and layer.dilation are tuples (for 1D conv, they have length=1)
                k = layer.kernel_size[0]
                d = layer.dilation[0]
                R += (k - 1) * d
        return R


class NetworkNoise3_5(nn.Module):
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
        return likelihood.mean()/float(wav_tensor.shape[-1])
    
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
        self.dilation_pattern = [1, 2, 4,8, 8, 4, 2,1]
        self.receptive_field = self.calculate_receptive_field()
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
    
    
    def calc_model_likelihood(self, expected_means, expected_stds, wav_tensor, verbose=False, offset=True):
        offset_value = self.kernel_size
        if offset == True:
            offset_value = self.receptive_field
        wav_tensor = wav_tensor.squeeze(axis=1)[:,offset_value+1:]
        means_=expected_means.squeeze(axis=1)[:,offset_value:-1]
        stds_ = expected_stds.squeeze(axis=1)[:,offset_value:-1]
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
    
    def casual_loss(self, expected_means, expected_stds, wav_tensor, offset=True):
        model_likelihood = self.calc_model_likelihood(expected_means, expected_stds, wav_tensor,offset=offset)
        return -model_likelihood   
    
    def calculate_receptive_field(self):
        """
        Computes the sequential-sum receptive field for a stack of causal convolution blocks,
        each with the given kernel size and dilation specified in the 'dilations' list.
        """
        dilations=self.dilation_pattern
        kernel_size=self.kernel_size
        rf = 1
        k_minus_1 = kernel_size - 1

        # Add up (k-1)*d_i for each block
        for d in dilations:
            rf += k_minus_1 * d

        return rf
    


    


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class NetworkNoise4MoG(nn.Module):
    def __init__(self, kernel_size=9, num_mixtures=5, clamp_log_sig=(-7.0, 5.0)):
        """
        A Mixture-of-Gaussians version of NetworkNoise4.

        :param kernel_size: size of the causal conv kernels
        :param num_mixtures: K, number of mixture components
        :param clamp_log_sig: (min_val, max_val) to clamp log_sigma
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.num_mixtures = num_mixtures
        self.clamp_log_sig = clamp_log_sig  # e.g. (-7, 5)

        # same structure, but final out-channels = 3*K
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
        # Instead of outputting 2 channels, output 3*K channels
        self.conv8 = CausalConv1dClassS(2, 3 * num_mixtures, kernel_size=kernel_size, dilation=1)

        self.dilation_pattern = [1, 2, 4, 8, 8, 4, 2, 1]
        self.receptive_field = self.calculate_receptive_field()

    def forward(self, x, cur_gt=None):
        """
        x: shape (B, 1, T)
        Returns (logits, means, log_sig) each shape (B, K, T).
        We'll clamp log_sig in [self.clamp_log_sig].
        """
        # same chain as your original
        x1 = self.conv1(x)
        x = self.tanh1(x1)
        x = self.conv2(x)
        x = self.tanh2(x)
        x = self.conv3(x)
        x = self.tanh3(x)
        x2 = self.conv4(x)
        x = x2 + x1  # skip
        x = self.tanh4(x)
        x = self.conv5(x)
        x = self.tanh5(x)
        x = self.conv6(x)
        x = self.tanh6(x)
        x = self.conv7(x)
        x = self.tanh7(x)

        # Add skip from x1 & x2 as done originally:
        x = x + x1 + x2  # shape: (B, 3K, T)
        # final conv => produce 3K channels
        x = self.conv8(x)
        

        B, ch, T = x.shape
        K = self.num_mixtures  # for clarity

        # reshape => (B, 3, K, T)
        x = x.view(B, 3, K, T)
        logits  = x[:, 0, :, :]  # (B, K, T)
        means   = x[:, 1, :, :]  # (B, K, T)
        log_sig = x[:, 2, :, :]  # (B, K, T)

        # clamp log_sig
        min_val, max_val = self.clamp_log_sig
        log_sig = torch.clamp(log_sig, min_val, max_val)

        return logits, means, log_sig

    def calc_model_likelihood(self, logits, means, log_sig, wav_tensor, verbose=False, offset=True):
        """
        Mixture-of-Gaussians log-likelihood.
        We'll mimic your original offset indexing:
          - skip the first `self.receptive_field+1` in wav_tensor
          - skip self.receptive_field:-1 in the predictions
        """
        if offset:
            offset_value = self.receptive_field
        else:
            offset_value = 0

        # wav_tensor: shape (B, 1, T)
        wav_tensor = wav_tensor.squeeze(1)[:, offset_value+1:]   # skip first (receptive_field+1)
        logits_ = logits[:, :, offset_value:-1]  # skip [offset_value:-1]
        means_  = means[:, :, offset_value:-1]
        log_sig_= log_sig[:, :, offset_value:-1]

        if verbose:
            print("wav_tensor shape:", wav_tensor.shape)
            print("logits_ shape:", logits_.shape)
            print("means_ shape:", means_.shape)
            print("log_sig_ shape:", log_sig_.shape)

        # mixture weights alpha_k = softmax over mixture dim => dim=1
        alpha = F.softmax(logits_, dim=1)  # (B, K, T)

        # expand wav_tensor => shape (B, 1, T)
        x = wav_tensor.unsqueeze(1)  # (B, 1, T)
        sig = torch.exp(log_sig_)    # => std = exp(log_sig)
        var = sig**2

        # log p_k(x) = -0.5*((x - mu)^2 / sig^2) - 0.5*log(2*pi) - log_sig
        diff_sq = (x - means_)**2
        log_prob_k = -0.5 * diff_sq / var - 0.5 * math.log(2*math.pi) - log_sig_

        # Combine mixture weights
        # log p(x) = log sum_k [ alpha_k * exp(log_prob_k ) ]
        # = logsumexp( log(alpha_k) + log_prob_k , dim=1 )
        log_alpha = torch.log(alpha + 1e-12)
        combo = log_alpha + log_prob_k  # (B, K, T)
        log_prob = torch.logsumexp(combo, dim=1)  # -> (B, T)

        # sum over time
        log_prob_time = torch.sum(log_prob, dim=-1)  # (B,)
        # mean over batch
        likelihood = torch.mean(log_prob_time)
        return likelihood

    def casual_loss(self, logits, means, log_sig, wav_tensor, offset=True):
        """
        Negative log-likelihood for the mixture distribution.
        """
        ll = self.calc_model_likelihood(logits, means, log_sig, wav_tensor, offset=offset)
        return -ll

    def calculate_receptive_field(self):
            """
            Computes the sequential-sum receptive field for a stack of causal convolution blocks,
            each with the given kernel size and dilation specified in the 'dilations' list.
            """
            dilations=self.dilation_pattern
            kernel_size=self.kernel_size
            rf = 1
            k_minus_1 = kernel_size - 1

            # Add up (k-1)*d_i for each block
            for d in dilations:
                rf += k_minus_1 * d

            return rf




class NetworkNoise5(nn.Module):
    def __init__(self, kernel_size=9, num_channels=8, dilation_pattern=None):
        super().__init__()
        self.kernel_size=kernel_size
        self.num_channels = num_channels
        if dilation_pattern is None:
            dilation_pattern = [1, 2, 2,2, 4, 8, 16]
        self.blocks = nn.ModuleList()
        in_channels = 1  # Input channels (single waveform channel)
        for dilation in dilation_pattern:
            block = nn.Sequential(
                CausalConv1dClassS(in_channels, num_channels, kernel_size=kernel_size, dilation=dilation),
                nn.BatchNorm1d([num_channels]),
                nn.Tanh()
            )
            self.blocks.append(block)
            in_channels = num_channels
        # out_channels = 2
        # last_block = nn.Sequential(
        #         CausalConv1dClassS(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation),
        #         nn.LayerNorm([num_channels]),
        #         nn.Tanh()
        #     )
        # self.blocks.append(last_block)
        self.conv_mean = nn.Conv1d(num_channels, 1, kernel_size=1)
        self.conv_log_var = nn.Conv1d(num_channels, 1, kernel_size=1)
        


    def forward(self, x, cur_gt):
        residuals = x
        skip_connections = 0
        
        for block in self.blocks:
            out = block(residuals)
            skip_connections = skip_connections + out
            residuals = out


        means = self.conv_mean(skip_connections).squeeze(1)
        log_var = self.conv_log_var(skip_connections).squeeze(1)
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




class NetworkNoise6(nn.Module):
    def __init__(self, kernel_size=9, num_channels=8, dilation_pattern=None):
        super().__init__()
        self.kernel_size=kernel_size
        self.num_channels = num_channels
        
        if dilation_pattern is None:
            dilation_pattern = [1, 2, 4,8, 1, 2, 4, 8]
        self.dilation_pattern=dilation_pattern ###########attention
        self.blocks = nn.ModuleList()
        in_channels = 1  # Input channels (single waveform channel)
        for dilation in dilation_pattern:
            block = nn.Sequential(
                CausalConv1dClassS(in_channels, num_channels, kernel_size=kernel_size, dilation=dilation),
                # nn.BatchNorm1d([num_channels]),
                nn.Tanh()
            )
            self.blocks.append(block)
            in_channels = num_channels

        self.conv_mean = nn.Conv1d(num_channels, 1, kernel_size=1)
        self.conv_log_var = nn.Conv1d(num_channels, 1, kernel_size=1)
        
        self.receptive_field = self.calculate_receptive_field()  ###########attention
        

    def forward(self, x, cur_gt):
        residuals = x
        skip_connections = 0
        
        for block in self.blocks:
            out = block(residuals)
            skip_connections = skip_connections + out
            residuals = out


        means = self.conv_mean(skip_connections).squeeze(1)
        log_var = self.conv_log_var(skip_connections).squeeze(1)
        stds = torch.exp(0.5 *log_var)
        # stds = torch.ones_like(means)*self.b
        return means, stds
    
    def calc_model_likelihood(self, expected_means, expected_stds, wav_tensor, verbose=False, offset=True):
        offset_value = 0
        if offset == True:
            offset_value = self.receptive_field
        
        wav_tensor = wav_tensor.squeeze(axis=1)[:,offset_value+1:]
        means_=expected_means.squeeze(axis=1)[:,offset_value:-1]
        stds_ = expected_stds.squeeze(axis=1)[:,offset_value:-1]
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
    
    def casual_loss(self, expected_means, expected_stds, wav_tensor, offset=True):
        model_likelihood = self.calc_model_likelihood(expected_means, expected_stds, wav_tensor,offset=offset)
        return -model_likelihood   
    
    def calculate_receptive_field(self):
        """
        Computes the sequential-sum receptive field for a stack of causal convolution blocks,
        each with the given kernel size and dilation specified in the 'dilations' list.
        """
        dilations=self.dilation_pattern
        kernel_size=self.kernel_size
        rf = 1
        k_minus_1 = kernel_size - 1

        # Add up (k-1)*d_i for each block
        for d in dilations:
            rf += k_minus_1 * d

        return rf



class NetworkNoise6_5(nn.Module):
    def __init__(self, kernel_size=9, num_channels=8, dilation_pattern=None):
        super().__init__()
        self.kernel_size=kernel_size
        self.num_channels = num_channels
        
        if dilation_pattern is None:
            dilation_pattern = [1, 2, 4,8, 1, 2, 4, 8]
        self.dilation_pattern=dilation_pattern ###########attention
        self.blocks = nn.ModuleList()
        in_channels = 1  # Input channels (single waveform channel)
        for dilation in dilation_pattern:
            block = nn.Sequential(
                CausalConv1dClassS(in_channels, num_channels, kernel_size=kernel_size, dilation=dilation),
                # nn.BatchNorm1d([num_channels]),
                nn.Tanh()
            )
            self.blocks.append(block)
            in_channels = num_channels
        # out_channels = 2
        # last_block = nn.Sequential(
        #         CausalConv1dClassS(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation),
        #         nn.LayerNorm([num_channels]),
        #         nn.Tanh()
        #     )
        # self.blocks.append(last_block)
        self.conv_mean = nn.Conv1d(num_channels, 1, kernel_size=1)
        self.conv_log_var = nn.Conv1d(num_channels, 1, kernel_size=1)
        
        self.receptive_field = self.calculate_receptive_field()  ###########attention
        

    def forward(self, x, cur_gt):
        residuals = x
        skip_connections = 0
        
        for block in self.blocks:
            out = block(residuals)
            skip_connections = skip_connections + out
            residuals = out


        means = self.conv_mean(skip_connections).squeeze(1)
        log_var = self.conv_log_var(skip_connections).squeeze(1)
        stds = torch.exp(0.5 *log_var)
        # stds = torch.ones_like(means)*self.b
        return means, stds
    
    def calc_model_likelihood(self, expected_means, expected_stds, wav_tensor, verbose=False, offset=True):
        offset_value = 0
        if offset == True:
            offset_value = self.receptive_field
        
        wav_tensor = wav_tensor.squeeze(axis=1)[:,offset_value+1:]
        means_=expected_means.squeeze(axis=1)[:,offset_value:-1]
        stds_ = expected_stds.squeeze(axis=1)[:,offset_value:-1]
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
        return likelihood.mean()/float(wav_tensor.shape[-1])
    
    def casual_loss(self, expected_means, expected_stds, wav_tensor, offset=True):
        model_likelihood = self.calc_model_likelihood(expected_means, expected_stds, wav_tensor,offset=offset)
        return -model_likelihood   
    
    def calculate_receptive_field(self):
        """
        Computes the sequential-sum receptive field for a stack of causal convolution blocks,
        each with the given kernel size and dilation specified in the 'dilations' list.
        """
        dilations=self.dilation_pattern
        kernel_size=self.kernel_size
        rf = 1
        k_minus_1 = kernel_size - 1

        # Add up (k-1)*d_i for each block
        for d in dilations:
            rf += k_minus_1 * d

        return rf



class NetworkNoise6b(nn.Module):
    def __init__(self, kernel_size=9, num_channels=8, dilation_pattern=None):
        super().__init__()
        self.kernel_size=kernel_size
        self.num_channels = num_channels
        
        if dilation_pattern is None:
            dilation_pattern = [1, 2, 4,8, 1, 2, 4, 8]
        self.dilation_pattern=dilation_pattern ###########attention
        self.blocks = nn.ModuleList()
        in_channels = 1  # Input channels (single waveform channel)
        for dilation in dilation_pattern:
            block = nn.Sequential(
                CausalConv1dClassS(in_channels, num_channels, kernel_size=kernel_size, dilation=dilation),
                # nn.BatchNorm1d([num_channels]),
                nn.Tanh()
            )
            self.blocks.append(block)
            in_channels = num_channels
        # out_channels = 2
        # last_block = nn.Sequential(
        #         CausalConv1dClassS(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation),
        #         nn.LayerNorm([num_channels]),
        #         nn.Tanh()
        #     )
        # self.blocks.append(last_block)
        self.conv_mean = nn.Conv1d(num_channels, 1, kernel_size=1)
        self.conv_log_var = nn.Conv1d(num_channels, 1, kernel_size=1)
        
        self.receptive_field = self.calculate_receptive_field()  ###########attention
        

    def forward(self, x, cur_gt):
        residuals = x
        skip_connections = 0
        
        for block in self.blocks:
            out = block(residuals)
            skip_connections = skip_connections + out
            residuals = out


        means = self.conv_mean(skip_connections).squeeze(1)
        log_var = self.conv_log_var(skip_connections).squeeze(1)
        stds = torch.exp(0.5 *log_var)
        # stds = torch.ones_like(means)*self.b
        return means, stds
    
    def calc_model_likelihood(self, expected_means, expected_stds, wav_tensor, verbose=False, offset=True):
        offset_value = 1
        if offset == True:
            offset_value = self.receptive_field
        
        wav_tensor = wav_tensor.squeeze(axis=1)[:,offset_value:]
        means_=expected_means.squeeze(axis=1)[:,offset_value:]
        stds_ = expected_stds.squeeze(axis=1)[:,offset_value:]
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
    
    def casual_loss(self, expected_means, expected_stds, wav_tensor, offset=True):
        model_likelihood = self.calc_model_likelihood(expected_means, expected_stds, wav_tensor,offset=offset)
        return -model_likelihood   
    
    def calculate_receptive_field(self):
        """
        Computes the sequential-sum receptive field for a stack of causal convolution blocks,
        each with the given kernel size and dilation specified in the 'dilations' list.
        """
        dilations=self.dilation_pattern
        kernel_size=self.kernel_size
        rf = 1
        k_minus_1 = kernel_size - 1

        # Add up (k-1)*d_i for each block
        for d in dilations:
            rf += k_minus_1 * d

        return rf


class NetworkNoise6c(nn.Module):
    def __init__(self, kernel_size=9, num_channels=8, dilation_pattern=None):
        super().__init__()
        self.kernel_size=kernel_size
        self.num_channels = num_channels
        
        if dilation_pattern is None:
            dilation_pattern = [1, 2, 4, 8, 8, 8, 16, 16]
        self.dilation_pattern=dilation_pattern ###########attention
        self.blocks = nn.ModuleList()
        in_channels = 1  # Input channels (single waveform channel)
        for dilation in dilation_pattern:
            block = nn.Sequential(
                CausalConv1dClassS(in_channels, num_channels, kernel_size=kernel_size, dilation=dilation),
                # nn.BatchNorm1d([num_channels]),
                nn.Tanh()
            )
            self.blocks.append(block)
            in_channels = num_channels
        # out_channels = 2
        # last_block = nn.Sequential(
        #         CausalConv1dClassS(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation),
        #         nn.LayerNorm([num_channels]),
        #         nn.Tanh()
        #     )
        # self.blocks.append(last_block)
        self.conv_mean = nn.Conv1d(num_channels, 1, kernel_size=1)
        self.conv_log_var = nn.Conv1d(num_channels, 1, kernel_size=1)
        
        self.receptive_field = self.calculate_receptive_field()  ###########attention
        

    def forward(self, x, cur_gt):
        residuals = x
        skip_connections = 0
        
        for block in self.blocks:
            out = block(residuals)
            skip_connections = skip_connections + out
            residuals = out


        means = self.conv_mean(skip_connections).squeeze(1)
        log_var = self.conv_log_var(skip_connections).squeeze(1)
        stds = torch.exp(0.5 *log_var)
        # stds = torch.ones_like(means)*self.b
        return means, stds
    
    def calc_model_likelihood(self, expected_means, expected_stds, wav_tensor, verbose=False, offset=True):
        offset_value = 0
        if offset == True:
            offset_value = self.receptive_field
        
        wav_tensor = wav_tensor.squeeze(axis=1)[:,offset_value+1:]
        means_=expected_means.squeeze(axis=1)[:,offset_value:-1]
        stds_ = expected_stds.squeeze(axis=1)[:,offset_value:-1]
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
    
    def casual_loss(self, expected_means, expected_stds, wav_tensor, offset=True):
        model_likelihood = self.calc_model_likelihood(expected_means, expected_stds, wav_tensor,offset=offset)
        return -model_likelihood   
    
    def calculate_receptive_field(self):
        """
        Computes the sequential-sum receptive field for a stack of causal convolution blocks,
        each with the given kernel size and dilation specified in the 'dilations' list.
        """
        dilations=self.dilation_pattern
        kernel_size=self.kernel_size
        rf = 1
        k_minus_1 = kernel_size - 1

        # Add up (k-1)*d_i for each block
        for d in dilations:
            rf += k_minus_1 * d

        return rf


class NetworkNoise6d(nn.Module):
    def __init__(self, kernel_size=2, num_channels=8, dilation_pattern=None):
        super().__init__()
        self.kernel_size=kernel_size
        self.num_channels = num_channels
        
        if dilation_pattern is None:
            dilation_pattern = [1, 1, 1, 1, 1, 1, 1, 1]
        self.dilation_pattern=dilation_pattern ###########attention
        self.blocks = nn.ModuleList()
        in_channels = 1  # Input channels (single waveform channel)
        for dilation in dilation_pattern:
            block = nn.Sequential(
                CausalConv1dClassS(in_channels, num_channels, kernel_size=kernel_size, dilation=dilation),
                # nn.BatchNorm1d([num_channels]),
                nn.Tanh()
            )
            self.blocks.append(block)
            in_channels = num_channels
        # out_channels = 2
        # last_block = nn.Sequential(
        #         CausalConv1dClassS(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation),
        #         nn.LayerNorm([num_channels]),
        #         nn.Tanh()
        #     )
        # self.blocks.append(last_block)
        self.conv_mean = nn.Conv1d(num_channels, 1, kernel_size=1)
        self.conv_log_var = nn.Conv1d(num_channels, 1, kernel_size=1)
        
        self.receptive_field = self.calculate_receptive_field()  ###########attention
        

    def forward(self, x, cur_gt):
        residuals = x
        skip_connections = 0
        
        for block in self.blocks:
            out = block(residuals)
            skip_connections = skip_connections + out
            residuals = out


        means = self.conv_mean(skip_connections).squeeze(1)
        log_var = self.conv_log_var(skip_connections).squeeze(1)
        stds = torch.exp(0.5 *log_var)
        # stds = torch.ones_like(means)*self.b
        return means, stds
    
    def calc_model_likelihood(self, expected_means, expected_stds, wav_tensor, verbose=False, offset=True):
        offset_value = 0
        if offset == True:
            offset_value = self.receptive_field
        
        wav_tensor = wav_tensor.squeeze(axis=1)[:,offset_value+1:]
        means_=expected_means.squeeze(axis=1)[:,offset_value:-1]
        stds_ = expected_stds.squeeze(axis=1)[:,offset_value:-1]
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
    
    def casual_loss(self, expected_means, expected_stds, wav_tensor, offset=True):
        model_likelihood = self.calc_model_likelihood(expected_means, expected_stds, wav_tensor,offset=offset)
        return -model_likelihood   
    
    def calculate_receptive_field(self):
        """
        Computes the sequential-sum receptive field for a stack of causal convolution blocks,
        each with the given kernel size and dilation specified in the 'dilations' list.
        """
        dilations=self.dilation_pattern
        kernel_size=self.kernel_size
        rf = 1
        k_minus_1 = kernel_size - 1

        # Add up (k-1)*d_i for each block
        for d in dilations:
            rf += k_minus_1 * d

        return rf
    

class NetworkNoise6e(nn.Module):
    def __init__(self, kernel_size=9, num_channels=8, dilation_pattern=None):
        super().__init__()
        self.kernel_size=kernel_size
        self.num_channels = num_channels
        
        if dilation_pattern is None:
            dilation_pattern = [1, 1, 1, 1, 1, 1, 1, 1]
        self.dilation_pattern=dilation_pattern ###########attention
        self.blocks = nn.ModuleList()
        in_channels = 1  # Input channels (single waveform channel)
        for dilation in dilation_pattern:
            block = nn.Sequential(
                CausalConv1dClassS(in_channels, num_channels, kernel_size=kernel_size, dilation=dilation),
                # nn.BatchNorm1d([num_channels]),
                nn.Tanh()
            )
            self.blocks.append(block)
            in_channels = num_channels
        # out_channels = 2
        # last_block = nn.Sequential(
        #         CausalConv1dClassS(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation),
        #         nn.LayerNorm([num_channels]),
        #         nn.Tanh()
        #     )
        # self.blocks.append(last_block)
        self.conv_mean = nn.Conv1d(num_channels, 1, kernel_size=1)
        self.conv_log_var = nn.Conv1d(num_channels, 1, kernel_size=1)
        
        self.receptive_field = self.calculate_receptive_field()  ###########attention
        

    def forward(self, x, cur_gt):
        residuals = x
        skip_connections = 0
        
        for block in self.blocks:
            out = block(residuals)
            skip_connections = skip_connections + out
            residuals = out


        means = self.conv_mean(skip_connections).squeeze(1)
        log_var = self.conv_log_var(skip_connections).squeeze(1)
        stds = torch.exp(0.5 *log_var)
        # stds = torch.ones_like(means)*self.b
        return means, stds
    
    def calc_model_likelihood(self, expected_means, expected_stds, wav_tensor, verbose=False, offset=True):
        offset_value = 0
        if offset == True:
            offset_value = self.receptive_field
        
        wav_tensor = wav_tensor.squeeze(axis=1)[:,offset_value+1:]
        means_=expected_means.squeeze(axis=1)[:,offset_value:-1]
        stds_ = expected_stds.squeeze(axis=1)[:,offset_value:-1]
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
    
    def casual_loss(self, expected_means, expected_stds, wav_tensor, offset=True):
        model_likelihood = self.calc_model_likelihood(expected_means, expected_stds, wav_tensor,offset=offset)
        return -model_likelihood   
    
    def calculate_receptive_field(self):
        """
        Computes the sequential-sum receptive field for a stack of causal convolution blocks,
        each with the given kernel size and dilation specified in the 'dilations' list.
        """
        dilations=self.dilation_pattern
        kernel_size=self.kernel_size
        rf = 1
        k_minus_1 = kernel_size - 1

        # Add up (k-1)*d_i for each block
        for d in dilations:
            rf += k_minus_1 * d

        return rf
    

class NetworkNoise9(nn.Module):
    def __init__(self, kernel_size=11, num_channels=8, dilation_pattern=None):
        super().__init__()
        self.kernel_size=kernel_size
        self.num_channels = num_channels
        
        if dilation_pattern is None:
            dilation_pattern = [1, 2, 4,8, 16, 4, 8, 16,32,64]
        self.dilation_pattern=dilation_pattern ###########attention
        self.blocks = nn.ModuleList()
        in_channels = 1  # Input channels (single waveform channel)
        for dilation in dilation_pattern:
            block = nn.Sequential(
                CausalConv1dClassS(in_channels, num_channels, kernel_size=kernel_size, dilation=dilation),
                # nn.BatchNorm1d([num_channels]),
                nn.Tanh()
            )
            self.blocks.append(block)
            in_channels = num_channels
        # out_channels = 2
        # last_block = nn.Sequential(
        #         CausalConv1dClassS(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation),
        #         nn.LayerNorm([num_channels]),
        #         nn.Tanh()
        #     )
        # self.blocks.append(last_block)
        self.conv_mean = nn.Conv1d(num_channels, 1, kernel_size=1)
        self.conv_log_var = nn.Conv1d(num_channels, 1, kernel_size=1)
        
        self.receptive_field = self.calculate_receptive_field()  ###########attention
        

    def forward(self, x, cur_gt):
        residuals = x
        skip_connections = 0
        
        for block in self.blocks:
            out = block(residuals)
            skip_connections = skip_connections + out
            residuals = out


        means = self.conv_mean(skip_connections).squeeze(1)
        log_var = self.conv_log_var(skip_connections).squeeze(1)
        stds = torch.exp(0.5 *log_var)
        # stds = torch.ones_like(means)*self.b
        return means, stds
    
    def calc_model_likelihood(self, expected_means, expected_stds, wav_tensor, verbose=False, offset=True):
        offset_value = 0
        if offset == True:
            offset_value = self.receptive_field
        
        wav_tensor = wav_tensor.squeeze(axis=1)[:,offset_value+1:]
        means_=expected_means.squeeze(axis=1)[:,offset_value:-1]
        stds_ = expected_stds.squeeze(axis=1)[:,offset_value:-1]
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
    
    def casual_loss(self, expected_means, expected_stds, wav_tensor, offset=True):
        model_likelihood = self.calc_model_likelihood(expected_means, expected_stds, wav_tensor,offset=offset)
        return -model_likelihood   
    
    def calculate_receptive_field(self):
        """
        Computes the sequential-sum receptive field for a stack of causal convolution blocks,
        each with the given kernel size and dilation specified in the 'dilations' list.
        """
        dilations=self.dilation_pattern
        kernel_size=self.kernel_size
        rf = 1
        k_minus_1 = kernel_size - 1

        # Add up (k-1)*d_i for each block
        for d in dilations:
            rf += k_minus_1 * d

        return rf
    
    
class NetworkNoise10(nn.Module):
    def __init__(self, kernel_size=13, num_channels=12, dilation_pattern=None):
        super().__init__()
        self.kernel_size=kernel_size
        self.num_channels = num_channels
        
        if dilation_pattern is None:
            dilation_pattern = [1, 2, 4,8, 2, 4, 8, 2, 4, 8, 2, 4, 8, 2, 4, 8, 2, 4, 8,2, 4, 8,2, 4, 8, 4, 8, 16, 32]
        self.dilation_pattern=dilation_pattern ###########attention
        self.blocks = nn.ModuleList()
        in_channels = 1  # Input channels (single waveform channel)
        for dilation in dilation_pattern:
            block = nn.Sequential(
                CausalConv1dClassS(in_channels, num_channels, kernel_size=kernel_size, dilation=dilation),
                # nn.BatchNorm1d([num_channels]),
                nn.Tanh()
            )
            self.blocks.append(block)
            in_channels = num_channels
        # out_channels = 2
        # last_block = nn.Sequential(
        #         CausalConv1dClassS(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation),
        #         nn.LayerNorm([num_channels]),
        #         nn.Tanh()
        #     )
        # self.blocks.append(last_block)
        self.conv_mean = nn.Conv1d(num_channels, 1, kernel_size=1)
        self.conv_log_var = nn.Conv1d(num_channels, 1, kernel_size=1)
        
        self.receptive_field = self.calculate_receptive_field()  ###########attention
        

    def forward(self, x, cur_gt):
        residuals = x
        skip_connections = 0
        
        for block in self.blocks:
            out = block(residuals)
            skip_connections = skip_connections + out
            residuals = out


        means = self.conv_mean(skip_connections).squeeze(1)
        log_var = self.conv_log_var(skip_connections).squeeze(1)
        stds = torch.exp(0.5 *log_var)
        # stds = torch.ones_like(means)*self.b
        return means, stds
    
    def calc_model_likelihood(self, expected_means, expected_stds, wav_tensor, verbose=False, offset=True):
        offset_value = 0
        if offset == True:
            offset_value = self.receptive_field
        
        wav_tensor = wav_tensor.squeeze(axis=1)[:,offset_value+1:]
        means_=expected_means.squeeze(axis=1)[:,offset_value:-1]
        stds_ = expected_stds.squeeze(axis=1)[:,offset_value:-1]
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
    
    def casual_loss(self, expected_means, expected_stds, wav_tensor, offset=True):
        model_likelihood = self.calc_model_likelihood(expected_means, expected_stds, wav_tensor,offset=offset)
        return -model_likelihood   
    
    def calculate_receptive_field(self):
        """
        Computes the sequential-sum receptive field for a stack of causal convolution blocks,
        each with the given kernel size and dilation specified in the 'dilations' list.
        """
        dilations=self.dilation_pattern
        kernel_size=self.kernel_size
        rf = 1
        k_minus_1 = kernel_size - 1

        # Add up (k-1)*d_i for each block
        for d in dilations:
            rf += k_minus_1 * d

        return rf
    
    

class NetworkNoise7(nn.Module):
    def __init__(self, kernel_size=9, num_channels=32, dilation_pattern=None):
        super().__init__()
        self.kernel_size=kernel_size
        self.num_channels = num_channels
        if dilation_pattern is None:
            dilation_pattern = [1, 1, 2,2, 1, 2, 2, 1,1, 2, 2, 1,4]
        self.blocks = nn.ModuleList()
        in_channels = 1  # Input channels (single waveform channel)
        for dilation in dilation_pattern:
            block = nn.Sequential(
                CausalConv1dClassS(in_channels, num_channels, kernel_size=kernel_size, dilation=dilation),
                # nn.BatchNorm1d([num_channels]),
                nn.Tanh()
            )
            self.blocks.append(block)
            in_channels = num_channels

        self.conv_mean = nn.Conv1d(num_channels, 1, kernel_size=1)
        self.conv_log_var = nn.Conv1d(num_channels, 1, kernel_size=1)
        


    def forward(self, x, cur_gt):
        residuals = x
        skip_connections = 0
        
        for block in self.blocks:
            out = block(residuals)
            skip_connections = skip_connections + out
            residuals = out


        means = self.conv_mean(skip_connections).squeeze(1)
        log_var = self.conv_log_var(skip_connections).squeeze(1)
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



class NetworkNoise6MoG(nn.Module):
    """
    A version of your NetworkNoise6 that predicts a mixture of Gaussians
    (logits, means, log_sig) instead of a single mean/std.

    - Maintains the same block structure & skip-connections.
    - We produce 3*K output channels: [logits, means, log_sigma].
    - Includes log_sigma clamping to avoid extreme variances.
    - The mixture-likelihood logic mirrors your original indexing for skipping.
    """

    def __init__(
        self,
        kernel_size=9,
        num_channels=8,
        dilation_pattern=None,
        num_mixtures=50,
        clamp_log_sig=(-7.0, 5.0)
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_channels = num_channels
        self.num_mixtures = num_mixtures
        self.clamp_log_sig = clamp_log_sig  # (min_val, max_val) for log_sigma
        

        if dilation_pattern is None:
            dilation_pattern = [1, 2, 4, 8, 1, 2, 4, 8]
        self.dilation_pattern = dilation_pattern

        self.blocks = nn.ModuleList()
        in_channels = 1  # single waveform channel

        for dilation in dilation_pattern:
            block = nn.Sequential(
                CausalConv1dClassS(in_channels, num_channels,
                                   kernel_size=kernel_size,
                                   dilation=dilation),
                nn.Tanh()
            )
            self.blocks.append(block)
            in_channels = num_channels

        # Instead of conv_mean and conv_log_var, we have a single conv
        # that outputs 3*K channels: (logits, means, log_sigma)
        self.conv_out = nn.Conv1d(num_channels, 3 * num_mixtures, kernel_size=1)
        self.receptive_field = self.calculate_receptive_field()

    def forward(self, x, cur_gt=None):
        """
        :param x: shape (B, 1, T)
        :param cur_gt: unused in your original code, so we'll keep it for signature compatibility
        :return:
            logits:  (B, K, T)
            means:   (B, K, T)
            log_sig: (B, K, T)  [clamped]
        """
        residuals = x
        skip_connections = 0

        for block in self.blocks:
            out = block(residuals)
            skip_connections = skip_connections + out
            residuals = out

        # Produce 3*K channels
        out = self.conv_out(skip_connections)  # shape: (B, 3*K, T)

        B, ch, T = out.shape
        # Reshape to (B, 3, K, T)
        out = out.view(B, 3, self.num_mixtures, T)

        logits  = out[:, 0, :, :]  # (B, K, T)
        means   = out[:, 1, :, :]  # (B, K, T)
        log_sig = out[:, 2, :, :]  # (B, K, T)

        # Clamp log_sigma to avoid extreme values
        min_log, max_log = self.clamp_log_sig
        log_sig = torch.clamp(log_sig, min_log, max_log)

        return logits, means, log_sig

    def calc_model_likelihood(self, logits, means, log_sig, wav_tensor,verbose=False, offset=True):
        """
        Compute log-likelihood of the mixture of Gaussians at each time step,
        then sum over time and average over batch.
        
        Following your original indexing scheme:
          - skip the first `kernel_size+1` samples from wav_tensor
          - skip [kernel_size:-1] in the predictions
        """
        offset_value = 0
        if offset == True:
            offset_value = self.receptive_field
        
        # wav_tensor: (B, 1, T)
        wav_tensor = wav_tensor.squeeze(1)[:, offset_value+1:]  # skip first kernel_size+1
        # The model outputs: (B, K, T)
        # We'll skip [self.kernel_size:-1] in time dimension for these
        logits_ = logits[:, :, offset_value:-1]
        means_  = means[:,  :, offset_value:-1]
        log_sig_= log_sig[:, :, offset_value:-1]

        if verbose:
            print("wav_tensor shape:", wav_tensor.shape)
            print("logits_ shape:", logits_.shape)
            print("means_ shape:", means_.shape)
            print("log_sig_ shape:", log_sig_.shape)

        # alpha_k = softmax over mixture dimension
        alpha = torch.nn.functional.softmax(logits_, dim=1)  # (B, K, T)

        # Expand wav_tensor to broadcast with (B, K, T)
        x = wav_tensor.unsqueeze(1)  # (B, 1, T)
        sig = torch.exp(log_sig_)    # (B, K, T)
        var = sig ** 2

        # log p_k(x) = -0.5*((x - mu)^2 / sig^2) - 0.5*log(2*pi) - log_sig
        diff_sq = (x - means_) ** 2
        log_prob_k = -0.5 * diff_sq / var - 0.5 * math.log(2 * math.pi) - log_sig_

        # Combine mixture weights: log p(x) = log sum_k [ alpha_k * exp(log_prob_k ) ]
        # => logsumexp( log_alpha + log_prob_k )
        log_alpha = torch.log(alpha + 1e-12)
        combo = log_alpha + log_prob_k  # (B, K, T)
        log_prob = torch.logsumexp(combo, dim=1)  # sum over mixture dimension -> (B, T)

        # Sum over time dimension
        log_prob_time = torch.sum(log_prob, dim=-1)  # (B,)

        # Mean over batch
        likelihood = torch.mean(log_prob_time)
        return likelihood

    def casual_loss(self, logits, means, log_sig, wav_tensor, offset=True):
        """
        Negative log-likelihood (just like your original code but with mixture).
        """
        model_likelihood = self.calc_model_likelihood(logits, means, log_sig, wav_tensor, offset=offset)
        return -model_likelihood
    
    def calculate_receptive_field(self):
        """
        Computes the sequential-sum receptive field for a stack of causal convolution blocks,
        each with the given kernel size and dilation specified in the 'dilations' list.
        """
        dilations=self.dilation_pattern
        kernel_size=self.kernel_size
        rf = 1
        k_minus_1 = kernel_size - 1

        # Add up (k-1)*d_i for each block
        for d in dilations:
            rf += k_minus_1 * d

        return rf



import torch
import torch.nn as nn

class NetworkNoise8(nn.Module):
    def __init__(self, kernel_size=9, initial_channels=16, dilation_patterns=None):
        super().__init__()
        self.kernel_size = kernel_size
        
        # Default dilation patterns for each stage
        if dilation_patterns is None:
            dilation_patterns = [
                [1, 1, 2, 2],   # Stage 1
                [1, 2, 2, 1],   # Stage 2
                [1, 4, 4, 1],   # Stage 3
                [1, 8, 8, 1]    # Stage 4
            ]
        
        self.stages = nn.ModuleList()
        in_channels = 1  # Input channels (single waveform channel)
        channels = initial_channels

        # Create stages with progressively increasing channels
        for stage_dilations in dilation_patterns:
            stage_blocks = nn.ModuleList()
            for dilation in stage_dilations:
                block = nn.Sequential(
                    CausalConv1dClassS(in_channels, channels, kernel_size=kernel_size, dilation=dilation),
                    nn.Tanh()
                )
                stage_blocks.append(block)
                in_channels = channels
            self.stages.append(stage_blocks)
            channels= channels*2  # Double the number of channels at the end of each stage

        # Final output layers
        self.conv_mean = nn.Conv1d(in_channels, 1, kernel_size=1)
        self.conv_log_var = nn.Conv1d(in_channels, 1, kernel_size=1)

    def forward(self, x, cur_gt):
        device = x.device  # Ensure all layers are on the same device as the input
        residuals = x
        skip_connections = None

        # Pass through each stage
        for stage_blocks in self.stages:
            for block in stage_blocks:
                out = block(residuals.to(device))
                if skip_connections is None:
                    skip_connections = out
                else:
                    # Align skip connections before adding
                    if skip_connections.size(1) != out.size(1):
                        skip_proj = nn.Conv1d(skip_connections.size(1), out.size(1), kernel_size=1).to(device)
                        skip_connections = skip_proj(skip_connections)
                    skip_connections=skip_connections + out
                residuals = out

        means = self.conv_mean(skip_connections).squeeze(1).to(device)
        log_var = self.conv_log_var(skip_connections).squeeze(1).to(device)
        stds = torch.exp(0.5 * log_var)
        return means, stds

    def calc_model_likelihood(self, expected_means, expected_stds, wav_tensor, verbose=False):
        wav_tensor = wav_tensor.squeeze(axis=1)[:, self.kernel_size + 1:]
        means_ = expected_means.squeeze(axis=1)[:, self.kernel_size:-1]
        stds_ = expected_stds.squeeze(axis=1)[:, self.kernel_size:-1]

        exp_all = -(1 / 2) * ((torch.square(wav_tensor - means_) / torch.square(stds_)))
        param_all = 1 / (np.sqrt(2 * np.pi) * stds_)
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
        residual_channels=8,
        skip_channels=4,
        dilation_depth=8,
        num_stacks=4
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




        # in_channels=1,
        # out_channels=2,
        # residual_channels=32,
        # skip_channels=64,
        # kernel_size=3,
        # dilation_depth=8,
        # num_stacks=3


import torch.nn.utils as nn_utils

class GatedResidualBlockWeightNorm(nn.Module):
    """
    WaveNet-style gated residual block with weight normalization.
    """
    def __init__(self, in_channels, residual_channels, skip_channels, kernel_size, dilation):
        super().__init__()

        self.conv_filter = nn_utils.weight_norm(
            CausalConv1dClassS(in_channels, residual_channels, kernel_size, dilation=dilation),
            name='weight'
        )
        self.conv_gate = nn_utils.weight_norm(
            CausalConv1dClassS(in_channels, residual_channels, kernel_size, dilation=dilation),
            name='weight'
        )

        self.conv_res = nn_utils.weight_norm(
            nn.Conv1d(residual_channels, in_channels, kernel_size=1),
            name='weight'
        )
        self.conv_skip = nn_utils.weight_norm(
            nn.Conv1d(residual_channels, skip_channels, kernel_size=1),
            name='weight'
        )

    def forward(self, x):
        """
        x: shape (B, in_channels, T)
        returns:
          - residual_out: (B, in_channels, T)
          - skip_out:     (B, skip_channels, T)
        """
        filter_out = torch.tanh(self.conv_filter(x))
        gate_out   = torch.sigmoid(self.conv_gate(x))

        out = filter_out * gate_out  # (B, residual_channels, T)

        residual_out = self.conv_res(out)
        residual_out = residual_out + x  # skip connection

        skip_out = self.conv_skip(out)

        return residual_out, skip_out
    
class NetworkNoiseWaveNetMoG(nn.Module):
    def __init__(
        self,
        in_channels=1,
        residual_channels=32,
        skip_channels=64,
        kernel_size=3,
        dilation_cycle=[1,2,4,8,16,32,64,128],
        num_cycles=3,
        num_mixtures=5,
        clamp_log_sig=(-7.0, 5.0)
    ):
        """
        :param residual_channels: # of channels within each residual block
        :param skip_channels:     # of channels used in skip output
        :param kernel_size:       size of the convolution kernel
        :param dilation_cycle:    pattern of dilations (e.g. [1,2,4,8])
        :param num_cycles:        how many times to repeat that pattern
        :param num_mixtures:      K in the mixture of Gaussians
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.num_mixtures = num_mixtures
        self.clamp_log_sig = clamp_log_sig

        # Compute receptive field
        self.receptive_field = 1
        for _ in range(num_cycles):
            for d in dilation_cycle:
                self.receptive_field += (kernel_size - 1) * d

        # Pre-processing: 1x1 CausalConv1d -> WeightNorm
        self.causal_in = nn_utils.weight_norm(
            CausalConv1dClassS(in_channels, residual_channels, kernel_size=1, dilation=1),
            name='weight'
        )

        # Build GatedResidualBlocks
        self.blocks = nn.ModuleList()
        for _ in range(num_cycles):
            for d in dilation_cycle:
                block = GatedResidualBlock(
                    in_channels=residual_channels,
                    residual_channels=residual_channels,
                    skip_channels=skip_channels,
                    kernel_size=kernel_size,
                    dilation=d
                )
                self.blocks.append(block)

        # Post-processing:
        # final_out channels = 3*K (for mixture: logits, means, log_sig)
        self.skip_post = nn.Sequential(
            nn.ReLU(),
            nn_utils.weight_norm(nn.Conv1d(skip_channels, skip_channels, kernel_size=1), name='weight'),
            nn.ReLU(),
            nn_utils.weight_norm(nn.Conv1d(skip_channels, 3 * num_mixtures, kernel_size=1), name='weight')
        )

    def forward(self, x, cur_gt):
        """
        :param x: shape (B, 1, T)
        :return:
          logits (B, K, T),
          means  (B, K, T),
          log_sig (B, K, T)    # clamped
        """
        x_proc = self.causal_in(x)  # (B, residual_channels, T)

        skip_sum = 0
        residual = x_proc
        for block in self.blocks:
            residual, skip = block(residual)
            skip_sum = skip_sum + skip

        out = self.skip_post(skip_sum)  # shape (B, 3*K, T)

        # Reshape to (B, 3, K, T)
        B, ch, T = out.shape
        out = out.view(B, 3, self.num_mixtures, T)

        logits = out[:, 0, :, :]  # (B, K, T)
        means  = out[:, 1, :, :]  # (B, K, T)
        log_sig= out[:, 2, :, :]  # (B, K, T)

        # Clamp log_sigma
        log_sig = torch.clamp(log_sig, min=self.clamp_log_sig[0], max=self.clamp_log_sig[1])

        return logits, means, log_sig

#expected_means, expected_stds, wav_tensor
    def calc_model_likelihood(self, logits, means, log_sig, wav_tensor, offset=None):
        """
        Mixture-of-Gaussians log-likelihood, summed over time, then averaged over batch.
        :param logits: (B, K, T)
        :param means:  (B, K, T)
        :param log_sig:(B, K, T)
        :param wav_tensor: (B, 1, T) original waveform
        :param offset: how many initial samples to skip due to receptive field
        :return: average log-likelihood (scalar)
        """
        if offset is None:
            offset = self.receptive_field #1531

        # wav_tensor = wav_tensor.squeeze(axis=1)[:,self.kernel_size+1:]
        # means_=expected_means.squeeze(axis=1)[:,self.kernel_size:-1]
        # stds_ = expected_stds.squeeze(axis=1)[:,self.kernel_size:-1]
        # Squeeze wave
        x = wav_tensor.squeeze(1)  # shape (B, T)
        # We'll compare x[:, t] with the MoG parameters at [:, :, t]
        # so we skip the first 'offset' samples
        x = x[:, offset+1:]                    # (B, T-offset)
        logits = logits[:, :, offset:-1]       # (B, K, T-offset)
        means  = means[:, :, offset:-1]        # (B, K, T-offset)
        log_sig= log_sig[:, :, offset:-1]      # (B, K, T-offset)

        B, K, T = logits.shape
        # Mixture weights alpha_k
        # shape: (B, K, T)
        alpha = torch.softmax(logits, dim=1)

        # Expand x to broadcast: (B, 1, T)
        x_expanded = x.unsqueeze(1)  # (B, 1, T)
        # Compute log-likelihood for each mixture component
        # p_k(x) = N(x | mu_k, sig_k^2)
        # log p_k(x) = -0.5 * ((x - mu_k)^2 / sig_k^2) - 0.5 log(2 pi) - log sig_k
        # We'll do all in shape (B, K, T).

        sig = torch.exp(log_sig)  # (B, K, T)

        # (x - means)^2
        diff_sq = (x_expanded - means) ** 2  # (B, K, T)
        var = sig**2
        log_prob_k = -0.5 * diff_sq / var - 0.5 * math.log(2 * math.pi) - log_sig

        # Now combine with alpha_k:
        # log p(x) = log sum_k alpha_k * exp(log_prob_k)
        # We do that in a numerically stable way, e.g.:
        # max_log = log_prob_k.max(dim=1, keepdim=True).values  # or we handle differently
        # But here let's do a direct approach with torch.logsumexp.

        # We'll do: log p(x) = logsumexp( log(alpha_k) + log_prob_k , dim=1 )
        log_alpha = torch.log(alpha + 1e-12)  # avoid log(0)

        # shape (B, K, T)
        combo = log_alpha + log_prob_k
        # sum over mixture dimension K
        log_prob = torch.logsumexp(combo, dim=1)  # shape (B, T)

        # sum over time
        log_prob_time = torch.sum(log_prob, dim=1)  # shape (B,)

        # average over batch
        avg_log_prob = torch.mean(log_prob_time)  # scalar
        return avg_log_prob

    def casual_loss(self, logits, means, log_sig, wav_tensor):
        # Negative log-likelihood
        ll = self.calc_model_likelihood(logits, means, log_sig, wav_tensor)
        return -ll


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



class NetworkNoiseWaveNetMoG2(nn.Module):
    def __init__(
        self,
        in_channels=1,
        residual_channels=32,
        skip_channels=64,
        kernel_size=9,
        dilation_cycle=[1,2,4,4,8,16,32,64,128],
        num_cycles=3,
        num_mixtures=50,
        clamp_log_sig=(-7.0, 5.0)
    ):
        """
        :param residual_channels: # of channels within each residual block
        :param skip_channels:     # of channels used in skip output
        :param kernel_size:       size of the convolution kernel
        :param dilation_cycle:    pattern of dilations (e.g. [1,2,4,8])
        :param num_cycles:        how many times to repeat that pattern
        :param num_mixtures:      K in the mixture of Gaussians
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.num_mixtures = num_mixtures
        self.clamp_log_sig = clamp_log_sig

        # Compute receptive field
        self.receptive_field = 1
        for _ in range(num_cycles):
            for d in dilation_cycle:
                self.receptive_field += (kernel_size - 1) * d

        # Pre-processing: 1x1 CausalConv1d -> WeightNorm
        self.causal_in = nn_utils.weight_norm(
            CausalConv1dClassS(in_channels, residual_channels, kernel_size=1, dilation=1),
            name='weight'
        )

        # Build GatedResidualBlocks
        self.blocks = nn.ModuleList()
        for _ in range(num_cycles):
            for d in dilation_cycle:
                block = GatedResidualBlock(
                    in_channels=residual_channels,
                    residual_channels=residual_channels,
                    skip_channels=skip_channels,
                    kernel_size=kernel_size,
                    dilation=d
                )
                self.blocks.append(block)

        # Post-processing:
        # final_out channels = 3*K (for mixture: logits, means, log_sig)
        self.skip_post = nn.Sequential(
            nn.ReLU(),
            nn_utils.weight_norm(nn.Conv1d(skip_channels, skip_channels, kernel_size=1), name='weight'),
            nn.ReLU(),
            nn_utils.weight_norm(nn.Conv1d(skip_channels, 3 * num_mixtures, kernel_size=1), name='weight')
        )

    def forward(self, x, cur_gt):
        """
        :param x: shape (B, 1, T)
        :return:
          logits (B, K, T),
          means  (B, K, T),
          log_sig (B, K, T)    # clamped
        """
        x_proc = self.causal_in(x)  # (B, residual_channels, T)

        skip_sum = 0
        residual = x_proc
        for block in self.blocks:
            residual, skip = block(residual)
            skip_sum = skip_sum + skip

        out = self.skip_post(skip_sum)  # shape (B, 3*K, T)

        # Reshape to (B, 3, K, T)
        B, ch, T = out.shape
        out = out.view(B, 3, self.num_mixtures, T)

        logits = out[:, 0, :, :]  # (B, K, T)
        means  = out[:, 1, :, :]  # (B, K, T)
        log_sig= out[:, 2, :, :]  # (B, K, T)

        # Clamp log_sigma
        log_sig = torch.clamp(log_sig, min=self.clamp_log_sig[0], max=self.clamp_log_sig[1])

        return logits, means, log_sig

#expected_means, expected_stds, wav_tensor
    def calc_model_likelihood(self, logits, means, log_sig, wav_tensor, offset=None):
        """
        Mixture-of-Gaussians log-likelihood, summed over time, then averaged over batch.
        :param logits: (B, K, T)
        :param means:  (B, K, T)
        :param log_sig:(B, K, T)
        :param wav_tensor: (B, 1, T) original waveform
        :param offset: how many initial samples to skip due to receptive field
        :return: average log-likelihood (scalar)
        """
        if offset is None:
            offset = self.receptive_field #1531

        # wav_tensor = wav_tensor.squeeze(axis=1)[:,self.kernel_size+1:]
        # means_=expected_means.squeeze(axis=1)[:,self.kernel_size:-1]
        # stds_ = expected_stds.squeeze(axis=1)[:,self.kernel_size:-1]
        # Squeeze wave
        x = wav_tensor.squeeze(1)  # shape (B, T)
        # We'll compare x[:, t] with the MoG parameters at [:, :, t]
        # so we skip the first 'offset' samples
        x = x[:, offset+1:]                    # (B, T-offset)
        logits = logits[:, :, offset:-1]       # (B, K, T-offset)
        means  = means[:, :, offset:-1]        # (B, K, T-offset)
        log_sig= log_sig[:, :, offset:-1]      # (B, K, T-offset)

        B, K, T = logits.shape
        # Mixture weights alpha_k
        # shape: (B, K, T)
        alpha = torch.softmax(logits, dim=1)

        # Expand x to broadcast: (B, 1, T)
        x_expanded = x.unsqueeze(1)  # (B, 1, T)
        # Compute log-likelihood for each mixture component
        # p_k(x) = N(x | mu_k, sig_k^2)
        # log p_k(x) = -0.5 * ((x - mu_k)^2 / sig_k^2) - 0.5 log(2 pi) - log sig_k
        # We'll do all in shape (B, K, T).

        sig = torch.exp(log_sig)  # (B, K, T)

        # (x - means)^2
        diff_sq = (x_expanded - means) ** 2  # (B, K, T)
        var = sig**2
        log_prob_k = -0.5 * diff_sq / var - 0.5 * math.log(2 * math.pi) - log_sig

        # Now combine with alpha_k:
        # log p(x) = log sum_k alpha_k * exp(log_prob_k)
        # We do that in a numerically stable way, e.g.:
        # max_log = log_prob_k.max(dim=1, keepdim=True).values  # or we handle differently
        # But here let's do a direct approach with torch.logsumexp.

        # We'll do: log p(x) = logsumexp( log(alpha_k) + log_prob_k , dim=1 )
        log_alpha = torch.log(alpha + 1e-12)  # avoid log(0)

        # shape (B, K, T)
        combo = log_alpha + log_prob_k
        # sum over mixture dimension K
        log_prob = torch.logsumexp(combo, dim=1)  # shape (B, T)

        # sum over time
        log_prob_time = torch.sum(log_prob, dim=1)  # shape (B,)

        # average over batch
        avg_log_prob = torch.mean(log_prob_time)  # scalar
        return avg_log_prob

    def casual_loss(self, logits, means, log_sig, wav_tensor):
        # Negative log-likelihood
        ll = self.calc_model_likelihood(logits, means, log_sig, wav_tensor)
        return -ll
    
    
    
class Network2DNoise6(nn.Module):
    def __init__(self, kernel_size=9, num_channels=8, dilation_pattern=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_channels = num_channels
        
        if dilation_pattern is None:
            dilation_pattern = [1, 2, 4, 8, 1, 2, 4, 8]
        self.dilation_pattern = dilation_pattern
        
        # 1) We now have 2 input channels: the waveform + the noise-level.
        in_channels = 2   # <--- changed from 1 to 2

        self.blocks = nn.ModuleList()
        for dilation in dilation_pattern:
            block = nn.Sequential(
                CausalConv1dClassS(
                    in_channels, 
                    num_channels,
                    kernel_size=kernel_size,
                    dilation=dilation
                ),
                nn.Tanh()
            )
            self.blocks.append(block)
            in_channels = num_channels  # output of block is 'num_channels'

        # Final heads for mean & log_var (unchanged)
        self.conv_mean = nn.Conv1d(num_channels, 1, kernel_size=1)
        self.conv_log_var = nn.Conv1d(num_channels, 1, kernel_size=1)
        
        self.receptive_field = self.calculate_receptive_field()

    def forward(self, x, cur_gt):
        """
        x:       (B, 1, T) - original waveform
        cur_gt:  (B, 1, T) - noise-level or any other condition

        We'll concatenate them on channel dimension => (B, 2, T).
        """
        # 2) Concatenate waveform & noise level
        net_input = torch.cat([x, cur_gt], dim=1)  # shape (B, 2, T)

        # The rest is the same as your original
        residuals = net_input
        skip_connections = 0
        
        for block in self.blocks:
            out = block(residuals)
            skip_connections = skip_connections + out
            residuals = out

        means = self.conv_mean(skip_connections).squeeze(1)
        log_var = self.conv_log_var(skip_connections).squeeze(1)
        stds = torch.exp(0.5 * log_var)
        return means, stds
    
    def calc_model_likelihood(self, expected_means, expected_stds, wav_tensor, verbose=False, offset=True):
        offset_value = 0
        if offset:
            offset_value = self.receptive_field
        
        wav_tensor = wav_tensor.squeeze(1)[:, offset_value+1:]
        means_ = expected_means.squeeze(1)[:, offset_value:-1]
        stds_  = expected_stds.squeeze(1)[:, offset_value:-1]

        # standard Gaussian log-likelihood
        exp_all = -(0.5)*( (wav_tensor - means_)**2 / (stds_**2) )
        param_all = 1/( np.sqrt(2*np.pi)*stds_ )
        model_likelihood1 = torch.sum(torch.log(param_all), axis=-1)
        model_likelihood2 = torch.sum(exp_all, axis=-1)
        likelihood = model_likelihood1 + model_likelihood2

        if verbose:
            print("model_likelihood1: ", model_likelihood1)
            print("model_likelihood2: ", model_likelihood2)

        return likelihood.mean()
    
    def casual_loss(self, expected_means, expected_stds, wav_tensor, offset=True):
        model_likelihood = self.calc_model_likelihood(expected_means, expected_stds, wav_tensor, offset=offset)
        return -model_likelihood

    def calculate_receptive_field(self):
        dilations = self.dilation_pattern
        kernel_size = self.kernel_size
        rf = 1
        k_minus_1 = kernel_size - 1
        for d in dilations:
            rf += k_minus_1 * d
        return rf

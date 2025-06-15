import torch
import torch.nn as nn
import numpy as np
import math
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod





class CausalConv1dClassS(nn.Conv1d):
    def __init__(self,in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        pad = (kernel_size - 1) * dilation
        super().__init__(in_channels, out_channels, kernel_size, padding=pad, dilation=dilation, **kwargs)
    
    def forward(self, inputs):
        output = super().forward(inputs)
        if self.padding[0] != 0:
            output = output[:, :, :-self.padding[0]]
        return output
    


class NetworkNoiseClass(nn.Module,ABC):
    def __init__(self, kernel_size=9):
        super().__init__()
        
    @abstractmethod
    def forward(self, x, cur_gt):
        pass
    
    @abstractmethod
    def calculate_receptive_field(self, x, cur_gt):
        pass

        
    def calc_model_likelihood(self, expected_means, expected_stds, wav_tensor,offset=True,normalize=False):
        offset_value = self.kernel_size
        if offset:
            offset_value = self.receptive_field
        wav_tensor = wav_tensor.squeeze(axis=1)[:,offset_value+1:]
        means_=expected_means.squeeze(axis=1)[:,offset_value:-1]
        stds_ = expected_stds.squeeze(axis=1)[:,offset_value:-1]

        exp_all = -(1/2)*((torch.square(wav_tensor-means_)/torch.square(stds_)))
        param_all = 1/(np.sqrt(2*np.pi)*stds_)
        model_likelihood1 = torch.sum(torch.log(param_all), axis=-1) 
        model_likelihood2 = torch.sum(exp_all, axis=-1) 
        likelihood = model_likelihood1 + model_likelihood2
        likelihood = likelihood.mean()
        if normalize:
            return likelihood/wav_tensor.shape[-1]
        else:
            return likelihood
    
    def casual_loss(self, expected_means, expected_stds, wav_tensor, offset=True,normalize=False):
        model_likelihood = self.calc_model_likelihood(expected_means, expected_stds, wav_tensor,offset=offset,normalize=normalize)
        return -model_likelihood   


class NetworkNoise3_6(NetworkNoiseClass):
    def __init__(self, kernel_size=9):
        super().__init__()
        self.kernel_size = kernel_size

        # Convolutional Blocks with Gated Activation
        self.conv1 = nn.utils.weight_norm(CausalConv1dClassS(1, 2, kernel_size=kernel_size, dilation=1))
        self.gate1 = nn.Conv1d(2, 2, kernel_size=1)
        
        self.conv2 = nn.utils.weight_norm(CausalConv1dClassS(2, 2, kernel_size=kernel_size, dilation=2))
        self.gate2 = nn.Conv1d(2, 2, kernel_size=1)
        
        self.conv3 = nn.utils.weight_norm(CausalConv1dClassS(2, 2, kernel_size=kernel_size, dilation=4))
        self.gate3 = nn.Conv1d(2, 2, kernel_size=1)
        
        self.conv4 = nn.utils.weight_norm(CausalConv1dClassS(2, 2, kernel_size=kernel_size, dilation=8))
        self.gate4 = nn.Conv1d(2, 2, kernel_size=1)

        self.conv_mean = nn.Conv1d(2, 1, kernel_size=1)
        self.conv_log_var = nn.Conv1d(2, 1, kernel_size=1)
        self.receptive_field = self.calculate_receptive_field()
    
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
        
        x = x + x1
        
        means = self.conv_mean(x).squeeze(1)
        log_var = self.conv_log_var(x).squeeze(1)
        stds = torch.exp(0.5 * log_var)
        
        return means, stds
    
    
    def calculate_receptive_field(self):
        total_rf = 1
        conv_layers = [self.conv1, self.conv2, self.conv3, self.conv4]
        for conv in conv_layers:
            # weight_norm wraps your CausalConv1dClassS, but the base object 
            # still exposes .kernel_size and .dilation as (tuple,) 
            # so we take [0] for 1D
            k = conv.kernel_size[0]
            d = conv.dilation[0]
            total_rf += (k - 1) * d
        return total_rf






import torch
import torch.nn as nn
import torch.nn.functional as F



class ResidualBlock(nn.Module):
    """
    One "residual" layer:
     1) A weight-normed causal conv
     2) A 1x1 gate conv
     3) Gated activation function
     4) Residual skip: out += x
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv = nn.utils.weight_norm(
            CausalConv1dClassS(in_channels, out_channels,
                               kernel_size=kernel_size, dilation=dilation)
        )
        self.gate = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def gated_activation(self, x):
        """
        Gated activation:  tanh(x) * sigmoid(x).
        (We apply self.gate(...) outside this function.)
        """
        return torch.tanh(x[0]) * torch.sigmoid(x[1])

    def forward(self, x):
        """
        x: [B, in_channels, T]
        Output has shape [B, out_channels, T].
        """
        # Save the incoming x for the skip connection
        residual = x

        # 1) Causal convolution
        out = self.conv(x)

        # 2) Gate conv
        gated = self.gate(out)  # shape [B, out_channels, T]

        # 3) Gated activation
        out = self.gated_activation((out, gated))

        # 4) Add skip connection
        out = out + residual
        return out


class NetworkNoise18(NetworkNoiseClass):
    """
    18-layer network. Each layer:
      - CausalConv1d + gating
      - skip connection with input to that layer
    Finally, we produce a 1×1 conv for means and log-variance.

    The `dilations` here are an example sequence. Adapt as needed.
    """
    def __init__(self, kernel_size=50, channels=8):
        super().__init__()
        self.kernel_size = kernel_size

        # If you want powers-of-two dilations, do something like:
        # dilations = [2**i for i in range(18)]
        # Or if you want the repeated pattern [1,2,4,8], repeated more times, define it here.
        # For example, 18 layers repeating [1,2,4,8]:
        # repeated = [1,2,4,8] * 5 (makes 20 layers, so you'd slice to 18)
        # We'll just do powers-of-two for this example:
        self.dilations = [1,1,1,1,1,1,1,2,2,2,2,4,4,4,4,8,8,8]

        # We'll keep everything at 2 channels in/out, as in your original code.
        # However, the first block has in_channels=1, the rest have in_channels=2.
        self.blocks = nn.ModuleList()
        
        # The very first block goes from 1->2 channels
        first_block = ResidualBlock(in_channels=1, out_channels=channels,
                                    kernel_size=kernel_size, dilation=self.dilations[0])
        self.blocks.append(first_block)
        # The next 17 blocks will be 2->2
        for d in self.dilations[1:]:
            block = ResidualBlock(in_channels=channels, out_channels=channels,
                                  kernel_size=kernel_size, dilation=d)
            self.blocks.append(block)

        # After all blocks, we output mean and log-var from 2 channels -> 1
        self.conv_mean = nn.Conv1d(channels, 1, kernel_size=1)
        self.conv_log_var = nn.Conv1d(channels, 1, kernel_size=1)

        # Pre-calculate the receptive field
        self.receptive_field = self.calculate_receptive_field()

    def forward(self, x, cur_gt=None):
        """
        x: shape [B, 1, T], for example
        We won't use 'cur_gt' inside the net, but keep the signature if needed.
        """
        out = x
        for block in self.blocks:
            out = block(out)  # each block has a skip connection inside

        # Finally produce means & standard deviations
        means = self.conv_mean(out).squeeze(1)
        log_var = self.conv_log_var(out).squeeze(1)
        stds = torch.exp(0.5 * log_var)
        return means, stds

    def calculate_receptive_field(self):
        """
        Summation of (kernel_size - 1)*dilation for all 18 layers, plus 1 for
        the current time step.
        """
        total_rf = 1
        for block in self.blocks:
            # Access the underlying weight-normed conv
            conv = block.conv
            k = conv.kernel_size[0]
            d = conv.dilation[0]
            total_rf += (k - 1) * d
        return total_rf






class NetworkNoise16b(NetworkNoiseClass):
    """
    18-layer network. Each layer:
      - CausalConv1d + gating
      - skip connection with input to that layer
    Finally, we produce a 1×1 conv for means and log-variance.

    The `dilations` here are an example sequence. Adapt as needed.
    """
    def __init__(self, kernel_size=50, channels=16):
        super().__init__()
        self.kernel_size = kernel_size

        # If you want powers-of-two dilations, do something like:
        # dilations = [2**i for i in range(18)]
        # Or if you want the repeated pattern [1,2,4,8], repeated more times, define it here.
        # For example, 18 layers repeating [1,2,4,8]:
        # repeated = [1,2,4,8] * 5 (makes 20 layers, so you'd slice to 18)
        # We'll just do powers-of-two for this example:
        self.dilations = [1,1,1,1,1,1,1,1]

        # We'll keep everything at 2 channels in/out, as in your original code.
        # However, the first block has in_channels=1, the rest have in_channels=2.
        self.blocks = nn.ModuleList()
        
        # The very first block goes from 1->2 channels
        first_block = ResidualBlock(in_channels=1, out_channels=channels,
                                    kernel_size=kernel_size, dilation=self.dilations[0])
        self.blocks.append(first_block)
        # The next 17 blocks will be 2->2
        for d in self.dilations[1:]:
            block = ResidualBlock(in_channels=channels, out_channels=channels,
                                  kernel_size=kernel_size, dilation=d)
            self.blocks.append(block)

        # After all blocks, we output mean and log-var from 2 channels -> 1
        self.conv_mean = nn.Conv1d(channels, 1, kernel_size=1)
        self.conv_log_var = nn.Conv1d(channels, 1, kernel_size=1)

        # Pre-calculate the receptive field
        self.receptive_field = self.calculate_receptive_field()

    def forward(self, x, cur_gt=None):
        """
        x: shape [B, 1, T], for example
        We won't use 'cur_gt' inside the net, but keep the signature if needed.
        """
        out = x
        for block in self.blocks:
            out = block(out)  # each block has a skip connection inside

        # Finally produce means & standard deviations
        means = self.conv_mean(out).squeeze(1)
        log_var = self.conv_log_var(out).squeeze(1)
        stds = torch.exp(0.5 * log_var)
        return means, stds

    def calculate_receptive_field(self):
        """
        Summation of (kernel_size - 1)*dilation for all 18 layers, plus 1 for
        the current time step.
        """
        total_rf = 1
        for block in self.blocks:
            # Access the underlying weight-normed conv
            conv = block.conv
            k = conv.kernel_size[0]
            d = conv.dilation[0]
            total_rf += (k - 1) * d
        return total_rf



class NetworkNoise16(NetworkNoiseClass):
    """
    18-layer network. Each layer:
      - CausalConv1d + gating
      - skip connection with input to that layer
    Finally, we produce a 1×1 conv for means and log-variance.

    The `dilations` here are an example sequence. Adapt as needed.
    """
    def __init__(self, kernel_size=50, channels=16):
        super().__init__()
        self.kernel_size = kernel_size

        # If you want powers-of-two dilations, do something like:
        # dilations = [2**i for i in range(18)]
        # Or if you want the repeated pattern [1,2,4,8], repeated more times, define it here.
        # For example, 18 layers repeating [1,2,4,8]:
        # repeated = [1,2,4,8] * 5 (makes 20 layers, so you'd slice to 18)
        # We'll just do powers-of-two for this example:
        self.dilations = [1,1,2,2,2,4,4,8]

        # We'll keep everything at 2 channels in/out, as in your original code.
        # However, the first block has in_channels=1, the rest have in_channels=2.
        self.blocks = nn.ModuleList()
        
        # The very first block goes from 1->2 channels
        first_block = ResidualBlock(in_channels=1, out_channels=channels,
                                    kernel_size=kernel_size, dilation=self.dilations[0])
        self.blocks.append(first_block)
        # The next 17 blocks will be 2->2
        for d in self.dilations[1:]:
            block = ResidualBlock(in_channels=channels, out_channels=channels,
                                  kernel_size=kernel_size, dilation=d)
            self.blocks.append(block)

        # After all blocks, we output mean and log-var from 2 channels -> 1
        self.conv_mean = nn.Conv1d(channels, 1, kernel_size=1)
        self.conv_log_var = nn.Conv1d(channels, 1, kernel_size=1)

        # Pre-calculate the receptive field
        self.receptive_field = self.calculate_receptive_field()

    def forward(self, x, cur_gt=None):
        """
        x: shape [B, 1, T], for example
        We won't use 'cur_gt' inside the net, but keep the signature if needed.
        """
        out = x
        for block in self.blocks:
            out = block(out)  # each block has a skip connection inside

        # Finally produce means & standard deviations
        means = self.conv_mean(out).squeeze(1)
        log_var = self.conv_log_var(out).squeeze(1)
        stds = torch.exp(0.5 * log_var)
        return means, stds

    def calculate_receptive_field(self):
        """
        Summation of (kernel_size - 1)*dilation for all 18 layers, plus 1 for
        the current time step.
        """
        total_rf = 1
        for block in self.blocks:
            # Access the underlying weight-normed conv
            conv = block.conv
            k = conv.kernel_size[0]
            d = conv.dilation[0]
            total_rf += (k - 1) * d
        return total_rf





class NetworkNoise17(NetworkNoiseClass):
    """
    18-layer network. Each layer:
      - CausalConv1d + gating
      - skip connection with input to that layer
    Finally, we produce a 1×1 conv for means and log-variance.

    The `dilations` here are an example sequence. Adapt as needed.
    """
    def __init__(self, kernel_size=50, channels=8):
        super().__init__()
        self.kernel_size = kernel_size

        # If you want powers-of-two dilations, do something like:
        # dilations = [2**i for i in range(18)]
        # Or if you want the repeated pattern [1,2,4,8], repeated more times, define it here.
        # For example, 18 layers repeating [1,2,4,8]:
        # repeated = [1,2,4,8] * 5 (makes 20 layers, so you'd slice to 18)
        # We'll just do powers-of-two for this example:
        self.dilations = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,4,4]

        # We'll keep everything at 2 channels in/out, as in your original code.
        # However, the first block has in_channels=1, the rest have in_channels=2.
        self.blocks = nn.ModuleList()
        
        # The very first block goes from 1->2 channels
        first_block = ResidualBlock(in_channels=1, out_channels=channels,
                                    kernel_size=kernel_size, dilation=self.dilations[0])
        self.blocks.append(first_block)
        # The next 17 blocks will be 2->2
        for d in self.dilations[1:]:
            block = ResidualBlock(in_channels=channels, out_channels=channels,
                                  kernel_size=kernel_size, dilation=d)
            self.blocks.append(block)

        # After all blocks, we output mean and log-var from 2 channels -> 1
        self.conv_mean = nn.Conv1d(channels, 1, kernel_size=1)
        self.conv_log_var = nn.Conv1d(channels, 1, kernel_size=1)

        # Pre-calculate the receptive field
        self.receptive_field = self.calculate_receptive_field()

    def forward(self, x, cur_gt=None):
        """
        x: shape [B, 1, T], for example
        We won't use 'cur_gt' inside the net, but keep the signature if needed.
        """
        out = x
        for block in self.blocks:
            out = block(out)  # each block has a skip connection inside

        # Finally produce means & standard deviations
        means = self.conv_mean(out).squeeze(1)
        log_var = self.conv_log_var(out).squeeze(1)
        stds = torch.exp(0.5 * log_var)
        return means, stds

    def calculate_receptive_field(self):
        """
        Summation of (kernel_size - 1)*dilation for all 18 layers, plus 1 for
        the current time step.
        """
        total_rf = 1
        for block in self.blocks:
            # Access the underlying weight-normed conv
            conv = block.conv
            k = conv.kernel_size[0]
            d = conv.dilation[0]
            total_rf += (k - 1) * d
        return total_rf





class NetworkNoise28(NetworkNoiseClass):
    """
    18-layer network. Each layer:
      - CausalConv1d + gating
      - skip connection with input to that layer
    Finally, we produce a 1×1 conv for means and log-variance.

    The `dilations` here are an example sequence. Adapt as needed.
    """
    def __init__(self, kernel_size=50, channels=8):
        super().__init__()
        self.kernel_size = kernel_size

        # If you want powers-of-two dilations, do something like:
        # dilations = [2**i for i in range(18)]
        # Or if you want the repeated pattern [1,2,4,8], repeated more times, define it here.
        # For example, 18 layers repeating [1,2,4,8]:
        # repeated = [1,2,4,8] * 5 (makes 20 layers, so you'd slice to 18)
        # We'll just do powers-of-two for this example:
        self.dilations = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2]

        # We'll keep everything at 2 channels in/out, as in your original code.
        # However, the first block has in_channels=1, the rest have in_channels=2.
        self.blocks = nn.ModuleList()
        
        # The very first block goes from 1->2 channels
        first_block = ResidualBlock(in_channels=1, out_channels=channels,
                                    kernel_size=kernel_size, dilation=self.dilations[0])
        self.blocks.append(first_block)
        # The next 17 blocks will be 2->2
        for d in self.dilations[1:]:
            block = ResidualBlock(in_channels=channels, out_channels=channels,
                                  kernel_size=kernel_size, dilation=d)
            self.blocks.append(block)

        # After all blocks, we output mean and log-var from 2 channels -> 1
        self.conv_mean = nn.Conv1d(channels, 1, kernel_size=1)
        self.conv_log_var = nn.Conv1d(channels, 1, kernel_size=1)

        # Pre-calculate the receptive field
        self.receptive_field = self.calculate_receptive_field()

    def forward(self, x, cur_gt=None):
        """
        x: shape [B, 1, T], for example
        We won't use 'cur_gt' inside the net, but keep the signature if needed.
        """
        out = x
        for block in self.blocks:
            out = block(out)  # each block has a skip connection inside

        # Finally produce means & standard deviations
        means = self.conv_mean(out).squeeze(1)
        log_var = self.conv_log_var(out).squeeze(1)
        stds = torch.exp(0.5 * log_var)
        return means, stds

    def calculate_receptive_field(self):
        """
        Summation of (kernel_size - 1)*dilation for all 18 layers, plus 1 for
        the current time step.
        """
        total_rf = 1
        for block in self.blocks:
            # Access the underlying weight-normed conv
            conv = block.conv
            k = conv.kernel_size[0]
            d = conv.dilation[0]
            total_rf += (k - 1) * d
        return total_rf







class NetworkNoise3_6b(NetworkNoiseClass):
    def __init__(self, kernel_size=9):
        super().__init__()
        self.kernel_size = kernel_size

        # Convolutional Blocks with Gated Activation
        self.conv1 = nn.utils.weight_norm(CausalConv1dClassS(1, 2, kernel_size=kernel_size, dilation=1))
        self.gate1 = nn.Conv1d(2, 2, kernel_size=1)
        
        self.conv2 = nn.utils.weight_norm(CausalConv1dClassS(2, 2, kernel_size=kernel_size, dilation=2))
        self.gate2 = nn.Conv1d(2, 2, kernel_size=1)
        
        self.conv3 = nn.utils.weight_norm(CausalConv1dClassS(2, 2, kernel_size=kernel_size, dilation=4))
        self.gate3 = nn.Conv1d(2, 2, kernel_size=1)
        
        self.conv4 = nn.utils.weight_norm(CausalConv1dClassS(2, 2, kernel_size=kernel_size, dilation=8))
        self.gate4 = nn.Conv1d(2, 2, kernel_size=1)

        self.conv_mean = nn.Conv1d(2, 1, kernel_size=1)
        self.conv_log_var = nn.Conv1d(2, 1, kernel_size=1)
        self.receptive_field = self.calculate_receptive_field()
    
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
        
        x = x + x1
        
        means = self.conv_mean(x).squeeze(1)
        log_var = self.conv_log_var(x).squeeze(1)
        stds = torch.exp(0.5 * log_var)
        
        return means, stds
    
    
    def calculate_receptive_field(self):
        total_rf = 1
        conv_layers = [self.conv1, self.conv2, self.conv3, self.conv4]
        for conv in conv_layers:
            # weight_norm wraps your CausalConv1dClassS, but the base object 
            # still exposes .kernel_size and .dilation as (tuple,) 
            # so we take [0] for 1D
            k = conv.kernel_size[0]
            d = conv.dilation[0]
            total_rf += (k - 1) * d
        return total_rf
    
    def calc_model_likelihood(self, expected_means, expected_stds, wav_tensor,offset=False,normalize=False):
        offset_value = self.kernel_size
        if offset:
            offset_value = self.receptive_field
        wav_tensor = wav_tensor.squeeze(axis=1)[:,offset_value+1:]
        means_=expected_means.squeeze(axis=1)[:,offset_value:-1]
        stds_ = expected_stds.squeeze(axis=1)[:,offset_value:-1]

        exp_all = -(1/2)*((torch.square(wav_tensor-means_)/torch.square(stds_)))
        param_all = 1/(np.sqrt(2*np.pi)*stds_)
        model_likelihood1 = torch.sum(torch.log(param_all), axis=-1) 
        model_likelihood2 = torch.sum(exp_all, axis=-1) 
        likelihood = model_likelihood1 + model_likelihood2
        likelihood = likelihood.mean()
        if normalize:
            return likelihood/wav_tensor.shape[-1]
        else:
            return likelihood
    
    def casual_loss(self, expected_means, expected_stds, wav_tensor, offset=False,normalize=False):
        model_likelihood = self.calc_model_likelihood(expected_means, expected_stds, wav_tensor,offset=offset,normalize=normalize)
        return -model_likelihood   
    
    

class NetworkNoise3_6MoG(nn.Module):
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
        self.receptive_field = self.calculate_receptive_field()

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

    def calc_model_likelihood(self, mixture_log_probs, means, stds, wav_tensor, offset=True,verbose=False):
        """
        Computes the log likelihood using a unified formula:

            log p(x) = logsumexp_k ( log(pi_k) + log N(x|mu_k,sigma_k) )

        When n_mixtures==1, mixture_log_probs are zeros so this reduces to the original single-Gaussian likelihood.

        Slices the time dimension similarly to your original NetworkNoise3:
          - target: from index (kernel_size+1) onward.
          - predictions: from index kernel_size to -1.

        Assumes wav_tensor shape [B, 1, T].
        """
        offset_value = self.kernel_size
        if offset:
            offset_value = self.receptive_field
        # Slice along the last dimension using ellipsis.
        target = wav_tensor.squeeze(1)[..., offset_value+1:]   # [B, T_valid]
        means = means[..., offset_value:-1]                     # now slices the time dimension
        stds  = stds[..., offset_value:-1]
        mixture_log_probs = mixture_log_probs[..., offset_value:-1]

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
        return log_prob.mean() #/ target.shape[-1]

    def casual_loss(self, mixture_log_probs, expected_means, expected_stds, wav_tensor,offset=True):
        """
        Returns the negative log likelihood loss averaged over the batch.
        """
        ll = self.calc_model_likelihood(mixture_log_probs, expected_means, expected_stds, wav_tensor, offset=offset)
        return -ll
    
    def calculate_receptive_field(self):
        """
        Compute the total receptive field from the 4 causal conv layers.
        Each layer has kernel_size = 9 and dilations = 1,2,4,8.
        The formula for each layer is (k - 1)*d, and we start from 1
        to represent the current timestep.
        """
        total_rf = 1
        conv_layers = [self.conv1, self.conv2, self.conv3, self.conv4]
        for conv in conv_layers:
            # weight_norm wraps 'CausalConv1dClassS', but we can still access
            # .kernel_size and .dilation directly.
            k = conv.kernel_size[0]
            d = conv.dilation[0]
            total_rf += (k - 1) * d

        return total_rf
    
    
    
    #--------------------------------------------------------------
    
class ResBlock(nn.Module):
    def __init__(self, channels, kernel, dil):
        super().__init__()
        self.conv = CausalConv1dClassS(channels, channels, kernel, dilation=dil)
    def forward(self, x):
        return x + torch.relu(self.conv(torch.relu(x)))

class CausalStack(nn.Module):
    def __init__(self, channels=64, kernel=3, n_layers=4):
        super().__init__()
        self.inp = CausalConv1dClassS(1, channels, 1)
        self.body = nn.Sequential(*[
            ResBlock(channels, kernel, 2 ** i) for i in range(n_layers)
        ])
        self.out = CausalConv1dClassS(channels, 1, 1)
    def forward(self, x):                 # x: (B,1,L)
        h = self.body(torch.relu(self.inp(x)))
        return self.out(torch.relu(h))    # (B,1,L)
    
class GaussianARStepModel(nn.Module):
    """
    Implements one diffusion-step model
        μ_i = b · x_i + f_μ( y_{i-1} − a x_{i-1}, … )
        σ_i = exp( f_logσ( … ) )

    *a*, *b*, f_μ and f_logσ are unique to this instance.
    (B,1,L) tensor format is kept throughout the forward pass.
    """
    def __init__(self, channels=16, kernel=13, n_layers=4):
        super().__init__()
        self.raw_a = nn.Parameter(torch.tensor(-1.0))   # sigmoid → a∈(0,1)
        self.raw_b = nn.Parameter(torch.tensor( 1.0))   # sigmoid → b∈(0,1)
        self.f_mu     = CausalStack(channels, kernel, n_layers)
        self.f_logsig = CausalStack(channels, kernel, n_layers)

    # learned scalars
    def a(self): return torch.sigmoid(self.raw_a)
    def b(self): return torch.sigmoid(self.raw_b)

    # ------------------------------------------------------------------ #
    # forward keeps shapes (B,1,L) — *no* squeeze                           #
    # ------------------------------------------------------------------ #
    def forward(self, x_t, y):
        """
        Args
        ----
        x_t : (B,1,L)  noisy input at step t
        y   : (B,1,L)  clean / target waveform
        Returns
        -------
        mu, log_sigma : each (B,1,L)
        """
        a, b = self.a(), self.b()                 # scalars

        diff = y - a * x_t                        # (B,1,L)
        # shift one step left so μ_i can’t peek at y_i
        diff_shift = F.pad(diff, (1, 0))[:, :, :-1]     # (B,1,L)

        mu_adj    = self.f_mu(diff_shift)               # (B,1,L)
        log_sigma = self.f_logsig(diff_shift)           # (B,1,L)
        mu = b * x_t + mu_adj
        return mu, log_sigma

    # unchanged NLL helpers (keep “sum” & sign convention exactly as you had)
    def calc_model_likelihood(self, mu, log_sigma, y):
        inv_var = torch.exp(-2 * log_sigma)
        nll = -(log_sigma + 0.5 * math.log(2 * math.pi)
                + 0.5 * inv_var * (y - mu) ** 2).sum()
        return nll

    def casual_loss(self, mu, log_sigma, y):
        return -self.calc_model_likelihood(mu, log_sigma, y)
    
    
    
    #------------------------------------------------------------
    
    
    
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


# -----------------------------------------------------------
# 1.  Gated residual block with weight-norm
# -----------------------------------------------------------
class GatedResBlock(nn.Module):
    """
    Input / output shape: (B, C, L)
    A causal conv produces feature map h,
    a 1×1 conv produces gate g,
    output = x + tanh(h) * sigmoid(g)
    """
    def __init__(self, channels: int, kernel: int, dil: int):
        super().__init__()
        self.h = weight_norm(
            CausalConv1dClassS(channels, channels, kernel, dilation=dil)
        )
        self.g = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        h = self.h(x)
        g = self.g(h)                      # gate uses the same receptive field
        return x + torch.tanh(h) * torch.sigmoid(g)


# -----------------------------------------------------------
# 2.  Causal stack built from GatedResBlock
# -----------------------------------------------------------
class CausalStack2(nn.Module):
    """
    in:  (B, 1, L)
    out: (B, 1, L)
    """
    def __init__(self, channels=64, kernel=3, n_layers=4):
        super().__init__()
        # 1×1 causal “embedding’’ layer
        self.inp = weight_norm(CausalConv1dClassS(1, channels, kernel_size=1))
        # dilations = 1,2,4,8,…
        self.body = nn.Sequential(*[
            GatedResBlock(channels, kernel, 2 ** i) for i in range(n_layers)
        ])
        # 1×1 “fully-connected’’ output layer (per-time-step FC)
        self.out = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, x):                  # x: (B,1,L)
        h = self.body(torch.tanh(self.inp(x)))
        return self.out(torch.tanh(h))     # (B,1,L)


# -----------------------------------------------------------
# 3.  Gaussian AR step model that uses the new stack
# -----------------------------------------------------------
class GaussianARStepModel2(nn.Module):
    """
    μ_i = b · x_i + f_μ( y_{i−1} − a x_{i−1}, … )
    σ_i = exp( f_logσ( … ) )
    """
    def __init__(self, channels=64, kernel=3, n_layers=4):
        super().__init__()
        self.raw_a = nn.Parameter(torch.tensor(10.0))  # sigmoid ≈ 1.0 # sigmoid → (0,1)
        self.raw_b = nn.Parameter(torch.tensor(-4.595))  # sigmoid ≈ 0.01  # sigmoid → (0,1)

        self.f_mu     = CausalStack2(channels, kernel, n_layers)
        self.f_logsig = CausalStack2(channels, kernel, n_layers)

    # learned scalars
    def a(self): return torch.sigmoid(self.raw_a)
    def b(self): return torch.sigmoid(self.raw_b)

    # ------------------------------------------------------------------ #
    # forward keeps shapes (B,1,L) — *no* squeeze                         #
    # ------------------------------------------------------------------ #
    def forward(self, x_t, y):
        """
        x_t : (B,1,L)  noisy input at step t
        y   : (B,1,L)  clean / target waveform
        """
        a, b = self.a(), self.b()          # scalars

        diff = y - a * x_t
        diff_shift = F.pad(diff, (1, 0))[:, :, :-1]       # causal shift

        mu_adj    = self.f_mu(diff_shift)
        log_sigma = self.f_logsig(diff_shift)

        mu = b * x_t + mu_adj
        return mu, log_sigma

    # unchanged negative-log-likelihood helpers -------------------------
    def calc_model_likelihood(self, mu, log_sigma, y):
        inv_var = torch.exp(-2 * log_sigma)
        nll = -(log_sigma + 0.5 * math.log(2 * math.pi)
                + 0.5 * inv_var * (y - mu) ** 2).sum()
        return nll

    def casual_loss(self, mu, log_sigma, y):
        return -self.calc_model_likelihood(mu, log_sigma, y)




class GaussianARStepModel2b(nn.Module):
    """
    μ_i = b · x_i + f_μ( y_{i−1} − a x_{i−1}, … )
    σ_i = exp( f_logσ( … ) )
    """
    def __init__(self, channels=32, kernel=23, n_layers=4):
        super().__init__()
        self.raw_a = nn.Parameter(torch.tensor(10.0))  # sigmoid ≈ 1.0 # sigmoid → (0,1)
        self.raw_b = nn.Parameter(torch.tensor(-4.595))  # sigmoid ≈ 0.01  # sigmoid → (0,1)

        self.f_mu     = CausalStack2(channels, kernel, n_layers)
        self.f_logsig = CausalStack2(channels, kernel, n_layers)

    # learned scalars
    def a(self): return torch.sigmoid(self.raw_a)
    def b(self): return torch.sigmoid(self.raw_b)

    # ------------------------------------------------------------------ #
    # forward keeps shapes (B,1,L) — *no* squeeze                         #
    # ------------------------------------------------------------------ #
    def forward(self, x_t, y):
        """
        x_t : (B,1,L)  noisy input at step t
        y   : (B,1,L)  clean / target waveform
        """
        a, b = self.a(), self.b()          # scalars

        diff = y - a * x_t
        diff_shift = F.pad(diff, (1, 0))[:, :, :-1]       # causal shift

        mu_adj    = self.f_mu(diff_shift)
        log_sigma = self.f_logsig(diff_shift)

        mu = b * x_t + mu_adj
        return mu, log_sigma

    # unchanged negative-log-likelihood helpers -------------------------
    def calc_model_likelihood(self, mu, log_sigma, y):
        inv_var = torch.exp(-2 * log_sigma)
        nll = -(log_sigma + 0.5 * math.log(2 * math.pi)
                + 0.5 * inv_var * (y - mu) ** 2).sum()
        return nll

    def casual_loss(self, mu, log_sigma, y):
        return -self.calc_model_likelihood(mu, log_sigma, y)



class GaussianARStepModel3(nn.Module):
    """
    μ_i = 1·x_i + f_μ( y_{i−1} − 1·x_{i−1}, … )
    σ_i = exp( f_logσ( … ) )
    """
    def __init__(self, channels=64, kernel=3, n_layers=4):
        super().__init__()
        # Non-learnable constants
        self.register_buffer('a_const', torch.tensor(1.0))
        self.register_buffer('b_const', torch.tensor(1.0))

        self.f_mu     = CausalStack2(channels, kernel, n_layers)
        self.f_logsig = CausalStack2(channels, kernel, n_layers)

    # convenient accessors
    def a(self): return self.a_const
    def b(self): return self.b_const

    # ------------------------------------------------------------------
    def forward(self, x_t, y):
        """
        x_t : (B,1,L)  noisy input at step t
        y   : (B,1,L)  clean / target waveform
        """
        diff = y - self.a() * x_t
        diff_shift = F.pad(diff, (1, 0))[:, :, :-1]

        mu_adj    = self.f_mu(diff_shift)
        log_sigma = self.f_logsig(diff_shift)

        mu = self.b() * x_t + mu_adj
        return mu, log_sigma

    # unchanged helpers -------------------------------------------------
    def calc_model_likelihood(self, mu, log_sigma, y):
        inv_var = torch.exp(-2 * log_sigma)
        nll = -(log_sigma + 0.5 * math.log(2 * math.pi)
                + 0.5 * inv_var * (y - mu) ** 2).sum()
        return nll

    def casual_loss(self, mu, log_sigma, y):
        return -self.calc_model_likelihood(mu, log_sigma, y)
    
    
    
class GaussianARStepModel4(nn.Module):
    """
    μ_i = 1·x_i + f_μ( y_{i−1} − 1·x_{i−1}, … )
    σ_i = exp( f_logσ( … ) )
    """
    def __init__(self, channels=8, kernel=30, n_layers=16):
        super().__init__()
        # Non-learnable constants
        self.register_buffer('a_const', torch.tensor(1.0))
        self.register_buffer('b_const', torch.tensor(1.0))

        self.f_mu     = CausalStack2(channels, kernel, n_layers)
        self.f_logsig = CausalStack2(channels, kernel, n_layers)

    # convenient accessors
    def a(self): return self.a_const
    def b(self): return self.b_const

    # ------------------------------------------------------------------
    def forward(self, x_t, y):
        """
        x_t : (B,1,L)  noisy input at step t
        y   : (B,1,L)  clean / target waveform
        """
        diff = y - self.a() * x_t
        diff_shift = F.pad(diff, (1, 0))[:, :, :-1]

        mu_adj    = self.f_mu(diff_shift)
        log_sigma = self.f_logsig(diff_shift)

        mu = self.b() * x_t + mu_adj
        return mu, log_sigma

    # unchanged helpers -------------------------------------------------
    def calc_model_likelihood(self, mu, log_sigma, y):
        inv_var = torch.exp(-2 * log_sigma)
        nll = -(log_sigma + 0.5 * math.log(2 * math.pi)
                + 0.5 * inv_var * (y - mu) ** 2).sum()
        return nll

    def casual_loss(self, mu, log_sigma, y):
        return -self.calc_model_likelihood(mu, log_sigma, y)
    
    
class GaussianARStepModel5(nn.Module):
    """
    μ_i = 1·x_i + f_μ( y_{i−1} − 1·x_{i−1}, … )
    σ_i = exp( f_logσ( … ) )
    """
    def __init__(self, channels=64, kernel=50, n_layers=16):
        super().__init__()
        # Non-learnable constants
        self.register_buffer('a_const', torch.tensor(1.0))
        self.register_buffer('b_const', torch.tensor(1.0))

        self.f_mu     = CausalStack2(channels, kernel, n_layers)
        self.f_logsig = CausalStack2(channels, kernel, n_layers)

    # convenient accessors
    def a(self): return self.a_const
    def b(self): return self.b_const

    # ------------------------------------------------------------------
    def forward(self, x_t, y):
        """
        x_t : (B,1,L)  noisy input at step t
        y   : (B,1,L)  clean / target waveform
        """
        diff = y - self.a() * x_t
        diff_shift = F.pad(diff, (1, 0))[:, :, :-1]

        mu_adj    = self.f_mu(diff_shift)
        log_sigma = self.f_logsig(diff_shift)

        mu = self.b() * x_t + mu_adj
        return mu, log_sigma

    # unchanged helpers -------------------------------------------------
    def calc_model_likelihood(self, mu, log_sigma, y):
        inv_var = torch.exp(-2 * log_sigma)
        nll = -(log_sigma + 0.5 * math.log(2 * math.pi)
                + 0.5 * inv_var * (y - mu) ** 2).sum()
        return nll

    def casual_loss(self, mu, log_sigma, y):
        return -self.calc_model_likelihood(mu, log_sigma, y)
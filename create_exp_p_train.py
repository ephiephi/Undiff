
# model_per_step.py
# Requires: PyTorch ≥ 2.1
import math, numpy as np, itertools, torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 0.  Scheduler  (unchanged – your exact linear default)
# ---------------------------------------------------------------------------
def get_named_beta_schedule(name: str, num_steps: int) -> np.ndarray:
    if name == "linear":
        return np.linspace(1e-4, 2e-2, num_steps, dtype=np.float64)
    raise NotImplementedError(name)

# ---------------------------------------------------------------------------
# 1.  Your causal-conv layer
# ---------------------------------------------------------------------------
class CausalConv1dClassS(nn.Conv1d):
    def __init__(self, in_c, out_c, k, dilation=1, **kw):
        pad = (k - 1) * dilation
        super().__init__(in_c, out_c, k, padding=pad, dilation=dilation, **kw)
    def forward(self, x):
        y = super().forward(x)
        if self.padding[0]:
            y = y[:, :, :-self.padding[0]]
        return y

# ---------------------------------------------------------------------------
# 2.  AR(1) noise sampler  (train-time data generation demo)
# ---------------------------------------------------------------------------
def sample_ar_noise(batch, length, phi=0.9, sigma_eps=1.0, device="cpu"):
    eps = sigma_eps * torch.randn(batch, length, device=device)
    w = torch.zeros_like(eps)
    w[:, 0] = eps[:, 0]
    for t in range(1, length):
        w[:, t] = phi * w[:, t - 1] + eps[:, t]
    return w

# ---------------------------------------------------------------------------
# 3.  Tiny causal CNN stack  (used for f_μ and f_logσ)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 4.  **One model = one diffusion step**
# ---------------------------------------------------------------------------
class GaussianARStepModel(nn.Module):
    """
    Implements one timestep:
        μ_i = b * x_i + f_μ( y_{i-1} - a x_{i-1}, … )
        σ_i = exp( f_logσ( … ) )

    Parameters a, b, and both CNNs are *unique* to this instance.
    """
    def __init__(self, channels=16, kernel=13, n_layers=4):
        super().__init__()
        self.raw_a = nn.Parameter(torch.tensor(-1.0)) # sigmoid → a in (0,1)
        self.raw_b = nn.Parameter(torch.tensor( 1.0)) # sigmoid → b in (0,1)
        self.f_mu     = CausalStack(channels, kernel, n_layers)
        self.f_logsig = CausalStack(channels, kernel, n_layers)

    def a(self): return torch.sigmoid(self.raw_a)
    def b(self): return torch.sigmoid(self.raw_b)

    def forward(self, x_t, y):
        """
        x_t, y : shape (B,L)
        Returns μ, logσ : (B,L)
        """
        a, b = self.a(), self.b()
        diff = y - a * x_t                        # (B,L)
        diff_shift = F.pad(diff.unsqueeze(1), (1,0))[:, :, :-1]  # (B,1,L)
        mu_adj    = self.f_mu(diff_shift).squeeze(1)
        log_sigma = self.f_logsig(diff_shift).squeeze(1)
        mu = b * x_t + mu_adj
        return mu, log_sigma
    
    def calc_model_likelihood(self, mu, log_sigma, y):
        # wav_tensor = wav_tensor.squeeze(axis=1)[:,self.receptive_field+1:]
        # means_=expected_means.squeeze(axis=1)[:,self.receptive_field:-1]
        # stds_ = expected_stds.squeeze(axis=1)[:,self.receptive_field:-1]

        inv_var = torch.exp(-2 * log_sigma)
        nll = -(log_sigma + 0.5 * math.log(2 * math.pi)
                + 0.5 * inv_var * (y - mu) ** 2).sum()
        return nll
    
    def casual_loss(self, mu, log_sigma, y):
        model_likelihood = self.calc_model_likelihood( mu, log_sigma, y)
        return -model_likelihood

# ---------------------------------------------------------------------------
# 5.  Build *all* step-models and train them together
# ---------------------------------------------------------------------------

def calculate_scaling_factor(clean_audio, noise_audio, target_snr):
    """Calculate the scaling factor to adjust noise to the target SNR level."""
    target_snr = float(target_snr)
    clean_power = torch.mean(clean_audio**2)
    noise_power = torch.mean(noise_audio**2)
    desired_noise_power = clean_power / (10 ** (target_snr / 10))
    scaling_factor = torch.sqrt(desired_noise_power / noise_power)
    return scaling_factor


import torchaudio
def train(dataloader, model, optimizer, criterion, alpha_bars, num_epochs, device):
    """Train on a dataset of pairs (x0, y), where y = x0 + AR_noise."""
    model.train()
    for epoch in range(num_epochs):
        for x0, y in dataloader:
            # Ensure x0, y are 3D tensors: [B, C, L]
            if x0.dim() == 2: 
                x0 = x0.unsqueeze(1)    # add channel dim if missing (for e.g. [B,L] -> [B,1,L])
            if y.dim() == 2:
                y = y.unsqueeze(1)
            x0, y = x0.to(device), y.to(device)
            B = x0.size(0)  # batch size
            # Sample random timesteps for each sample in the batch
            t = torch.randint(0, len(alpha_bars), (B,), device=device)  # shape [B]
            # Gather ᾱ_t for each sample and reshape for broadcasting
            a_bar_t = alpha_bars[t]                   # shape [B]
            a_bar_t = a_bar_t[:, None, None]          # shape [B,1,1] for broadcasting over [B,C,L]
            # Sample Gaussian noise for each sample
            eps = torch.randn_like(x0)                # same shape as x0 [B,C,L]
            # **Compute x_t from clean x0** for each sample
            x_t = torch.sqrt(a_bar_t) * x0 + torch.sqrt(1 - a_bar_t) * eps
            # Model predicts y (noisy signal) from x_t
            y_pred = model(x_t, t)
            loss = criterion(y_pred, y)               # compare to noisy target y
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ... (any logging or epoch-end operations)




def train_ar(x0, y, model, optimizer, criterion, alpha_bars, num_epochs, device, chunk_size=None):
    """Train on a single sequence (clean signal x0 and noisy signal y = x0 + AR_noise)."""
    model.train()
    # Ensure x0 and y are 3D tensors: [batch, channels, length]
    if x0.dim() == 1:
        x0 = x0.unsqueeze(0)            # add batch dimension
    if x0.dim() == 2:
        x0 = x0.unsqueeze(1)            # add channel dimension
    if y.dim() == 1:
        y = y.unsqueeze(0)
    if y.dim() == 2:
        y = y.unsqueeze(1)
    x0, y = x0.to(device), y.to(device)

    for epoch in range(num_epochs):
        # If chunking is desired to preserve causality over long sequences
        seq_length = x0.size(-1)
        iter_ranges = [ (0, seq_length) ] if chunk_size is None else \
                      [ (i, min(i+chunk_size, seq_length)) for i in range(0, seq_length, chunk_size) ]
        for (start, end) in iter_ranges:
            x0_segment = x0[..., start:end]   # segment of the clean signal
            y_segment  = y[..., start:end]    # corresponding segment of noisy signal
            # Sample a random timestep t (as a tensor of shape [1] since batch=1)
            t = torch.randint(0, len(alpha_bars), (1,), device=device)
            # Gather ᾱ_t (alpha_bar at time t) and reshape for broadcasting
            a_bar_t = alpha_bars[t]                 # shape [1]
            a_bar_t = a_bar_t.view(1, 1, 1)         # shape [1,1,1] for [B,C,L] broadcasting
            # Sample Gaussian noise ε with same shape as x0_segment
            eps = torch.randn_like(x0_segment)
            # **Compute x_t from clean x0_segment (not from y)**:
            x_t = torch.sqrt(a_bar_t) * x0_segment + torch.sqrt(1 - a_bar_t) * eps
            # Model prediction for y from x_t
            y_pred = model(x_t, t)
            # Compute loss against the noisy target y_segment
            loss = criterion(y_pred, y_segment)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ... (any logging or epoch-related code)


# ---------------------------------------------------------------------------
# 6.  Example run
# ---------------------------------------------------------------------------
import pickle
if __name__ == "__main__":
    torch.manual_seed(0)
    trained_models = train_ar(device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nReturned {len(trained_models)} independent step-models.")
    params_dict = {"nets": [net.state_dict() for net in trained_models],"network": "GaussianARStepModel",}
    pickle_path = "/data/ephraim/datasets/known_noise/undiff_exps2/exp_p_try/try2"+"_models.pickle"

    with open(pickle_path, 'wb') as handle:
        pickle.dump(params_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)            





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
def train(clean_data, num_steps=200, epochs=50, scheduler="linear",
               device="cuda"):

    clean = clean_data.to(device)
    ar_noise = sample_ar_noise(clean.shape[0], clean.shape[1], 0.9, .1, device)
    factor = calculate_scaling_factor(clean, ar_noise, 5) #todo: for batch
    y = clean + factor*ar_noise

    # # --- add AR noise -----------------------------------------------------------
    # y = clean + sample_ar_noise(batch, seq_len, 0.9, .1, device)

    # --- diffusion schedule (forward process) -----------------------------------
    betas = torch.tensor(get_named_beta_schedule(scheduler, num_steps),
                         dtype=torch.float32, device=device)
    alphas = 1 - betas
    a_bar = torch.cumprod(alphas, 0)
    sqrt_a_bar     = torch.sqrt(a_bar)           # (T,)
    sqrt_one_minus = torch.sqrt(1 - a_bar)

    # --- build list of independent step-models ----------------------------------
    models = [GaussianARStepModel(channels=16, kernel=13, n_layers=4).to(device)
              for _ in range(num_steps)]

    # One optimiser for **all** parameters
    opt = torch.optim.Adam(itertools.chain(*(m.parameters() for m in models)),
                           lr=3e-4)

    # --- training loop ----------------------------------------------------------
    for epoch in range(1, epochs + 1):
        total = 0.
        for t in range(num_steps):          # 0 ...T-1 
            net = models[t]
            eps = torch.randn_like(y)
            x_t = sqrt_a_bar[t] * y + sqrt_one_minus[t] * eps

            mu, log_sigma = net(x_t, y)              # forward
            # inv_var = torch.exp(-2 * log_sigma)
            # nll = (log_sigma + 0.5 * math.log(2 * math.pi)
            #        + 0.5 * inv_var * (y - mu) ** 2).sum()
            nll = net.casual_loss(mu, log_sigma, y)  
            opt.zero_grad()
            nll.backward()
            opt.step()
            total += nll.item()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | avg-NLL {total/num_steps:.4f} "
                  f"| a0={models[0].a():.3f} b0={models[0].b():.3f}"
                  f"| a1={models[1].a():.3f} b1={models[1].b():.3f}"
                    f"| a2={models[2].a():.3f} b2={models[2].b():.3f}")

    return models   # ← list  [model_t for t=0..T-1]



def train_ar(num_steps=200, epochs=500, scheduler="linear",
               device="cuda"):
    
    import torchaudio
    clean ,sr=  torchaudio.load(r"/data/ephraim/datasets/known_noise/undiff_exps2/exp_p_try/clean_wav/clean_fileid_0.wav")  
    print("clean.shape:", clean.shape)  
    clean = clean.to(device)
    ar_noise = sample_ar_noise(clean.shape[0], clean.shape[1], 0.9, .1, device)
    factor = calculate_scaling_factor(clean, ar_noise, 5) #todo: for batch
    y = clean + factor*ar_noise

    # # --- add AR noise -----------------------------------------------------------
    # y = clean + sample_ar_noise(batch, seq_len, 0.9, .1, device)

    # --- diffusion schedule (forward process) -----------------------------------
    betas = torch.tensor(get_named_beta_schedule(scheduler, num_steps),
                         dtype=torch.float32, device=device)
    alphas = 1 - betas
    a_bar = torch.cumprod(alphas, 0)
    sqrt_a_bar     = torch.sqrt(a_bar)           # (T,)
    sqrt_one_minus = torch.sqrt(1 - a_bar)

    # --- build list of independent step-models ----------------------------------
    models = [GaussianARStepModel(channels=64, kernel=3, n_layers=4).to(device)
              for _ in range(num_steps)]

    # One optimiser for **all** parameters
    opt = torch.optim.Adam(itertools.chain(*(m.parameters() for m in models)),
                           lr=3e-4)

    # --- training loop ----------------------------------------------------------
    for epoch in range(1, epochs + 1):
        total = 0.
        for t in range(num_steps):          # 0 ...T-1 
            net = models[t]
            eps = torch.randn_like(y)
            x_t = sqrt_a_bar[t] * y + sqrt_one_minus[t] * eps

            mu, log_sigma = net(x_t, y)              # forward
            # inv_var = torch.exp(-2 * log_sigma)
            # nll = (log_sigma + 0.5 * math.log(2 * math.pi)
            #        + 0.5 * inv_var * (y - mu) ** 2).sum()
            nll = net.casual_loss(mu, log_sigma, y)  
            opt.zero_grad()
            nll.backward()
            opt.step()
            total += nll.item()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | avg-NLL {total/num_steps:.4f} "
                  f"| a0={models[0].a():.3f} b0={models[0].b():.3f}"
                  f"| a1={models[1].a():.3f} b1={models[1].b():.3f}"
                    f"| a2={models[2].a():.3f} b2={models[2].b():.3f}")

    return models   # ← list  [model_t for t=0..T-1]

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




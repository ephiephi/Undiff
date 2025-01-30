import os
import glob
import argparse
import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

###############################################################################
# 0) Minimal CausalConv1dClassS Implementation (Replace if you already have it)
###############################################################################
class CausalConv1dClassS(nn.Conv1d):
    """
    A simple causal convolution layer. 
    Expects input shape (B, C_in, T).
    'causal' means we left-pad by (kernel_size - 1) * dilation 
    and then trim the extra on the right.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=True):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation,
            bias=bias
        )

    def forward(self, x):
        # Perform the convolution
        out = super().forward(x)
        # Remove the trailing padding on the right so that it is causal
        if self.padding[0] > 0:
            out = out[:, :, :-self.padding[0]]
        return out


###############################################################################
# 1) Noise Schedule Helpers
###############################################################################
def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas, dtype=np.float64)

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    By default, "linear" is used here.
    """
    if schedule_name == "linear":
        beta_start = 0.0001
        beta_end   = 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"Unknown beta schedule: {schedule_name}")


###############################################################################
# 2) The NetworkNoise3 Architecture (Instead of WaveNet)
###############################################################################
class NetworkNoise3(nn.Module):
    def __init__(self, kernel_size=9):
        super().__init__()
        self.kernel_size = kernel_size

        self.conv1 = CausalConv1dClassS(1, 2, kernel_size=kernel_size, dilation=1)
        self.tanh1 = nn.Tanh()
        self.conv2 = CausalConv1dClassS(2, 2, kernel_size=kernel_size, dilation=2)
        self.tanh2 = nn.Tanh()
        self.conv3 = CausalConv1dClassS(2, 2, kernel_size=kernel_size, dilation=4)
        self.tanh3 = nn.Tanh()
        self.conv4 = CausalConv1dClassS(2, 2, kernel_size=kernel_size, dilation=8)

    def forward(self, x, cur_gt=None):
        """
        x: (B, 1, T)
        cur_gt: not used in this minimal example, but included for API consistency
        returns:
            means: (B, T)
            stds:  (B, T)
        """
        # 1) pass through first conv
        x1 = self.conv1(x)
        x_ = self.tanh1(x1)

        # 2) pass through second conv
        x_ = self.conv2(x_)
        x_ = self.tanh2(x_)

        # 3) pass through third conv
        x_ = self.conv3(x_)
        x_ = self.tanh3(x_)

        # 4) pass through fourth conv, add skip from x1
        x_ = self.conv4(x_)
        x_ = x_ + x1

        means = x_[:, 0, :]
        log_var = x_[:, 1, :]
        stds = torch.exp(0.5 * log_var)
        return means, stds

    def calc_model_likelihood(self, expected_means, expected_stds, wav_tensor, verbose=False):
        """
        Next-sample NLL, with a +1 shift for the target 
        (so output[t] predicts target[t+1]).
        """
        # (B,1,T)
        wav_tensor = wav_tensor.squeeze(1)[:, self.kernel_size+1:]
        # means, stds shapes: (B, T)
        means_ = expected_means.squeeze(1)[:, self.kernel_size:-1]
        stds_  = expected_stds.squeeze(1)[:, self.kernel_size:-1]

        # log-likelihood for Gaussian
        # log N(x|mu,sigma) = -log(sigma sqrt(2pi)) - 0.5((x-mu)/sigma)^2
        # We'll do it in parts:
        exp_all = -0.5 * ( (wav_tensor - means_)**2 / (stds_**2) )
        param_all = 1.0 / ( np.sqrt(2*np.pi) * stds_ )

        model_likelihood1 = torch.sum(torch.log(param_all), dim=-1)  # sum over time
        model_likelihood2 = torch.sum(exp_all, dim=-1)

        if verbose:
            print("model_likelihood1:", model_likelihood1)
            print("model_likelihood2:", model_likelihood2)

        likelihood = model_likelihood1 + model_likelihood2
        return likelihood.mean()

    def casual_loss(self, expected_means, expected_stds, wav_tensor):
        """
        Negative of the average model likelihood.
        """
        model_likelihood = self.calc_model_likelihood(expected_means, expected_stds, wav_tensor)
        return -model_likelihood


###############################################################################
# 3) AudioDataset with Diffusion-Based Noise Augmentation
###############################################################################
class AudioDataset(Dataset):
    def __init__(self, file_list, g_t, transform=None):
        """
        file_list: list of paths to .wav files
        g_t: a 1D tensor containing values for the diffusion noise scaling
        transform: optional transform (e.g. normalization)
        """
        super().__init__()
        self.file_list = file_list
        self.transform = transform
        self.g_t = g_t  # shape: (num_diffusion_timesteps,)

        # Max length to avoid excessive RAM
        self.max_length = 320000

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        waveform, sr = torchaudio.load(file_path)  # shape: (channels, T)

        # If stereo or multi-channel, mix down to mono
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Truncate if longer than 320k
        if waveform.shape[-1] > self.max_length:
            waveform = waveform[..., :self.max_length]

        # Optional: apply transform
        if self.transform:
            waveform = self.transform(waveform, sr)

        # Diffusion Noise Augmentation
        i = random.randint(0, len(self.g_t) - 1)
        noise = torch.randn_like(waveform)
        waveform = waveform + noise * self.g_t[i]

        return waveform  # shape: (1, T)

def collate_fn(batch):
    """
    Collate a list of waveforms (each shape: (1, T_i)) into 
    a single tensor (B, 1, T_max), zero-padding shorter waveforms if needed.
    """
    max_len = max(waveform.shape[-1] for waveform in batch)
    batch_size = len(batch)

    batched_waveforms = torch.zeros(batch_size, 1, max_len, dtype=batch[0].dtype)
    for i, waveform in enumerate(batch):
        length = waveform.shape[-1]
        batched_waveforms[i, :, :length] = waveform

    return batched_waveforms


###############################################################################
# 4) Training & Evaluation Functions
###############################################################################
def train_one_epoch(model, dataloader, optimizer, device, step_counter, save_path=None):
    model.train()
    total_loss = 0.0
    num_samples = 0

    for batch_waveforms in tqdm(dataloader):
        batch_waveforms = batch_waveforms.to(device)  # (B, 1, T)

        # Forward pass
        means, stds = model(batch_waveforms, cur_gt=None)

        # Compute negative log-likelihood
        loss = model.casual_loss(means, stds, batch_waveforms)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = batch_waveforms.size(0)
        total_loss += loss.item() * batch_size
        num_samples += batch_size

        # Update global step
        step_counter[0] += 1

        # Save every 50 steps
        if save_path is not None and (step_counter[0] % 50) == 0:
            current_step_path = save_path.replace("model", f"model_step{step_counter[0]}")
            print(f"[Step {step_counter[0]}] Saving model to {current_step_path}")
            torch.save(model.state_dict(), current_step_path)

    return total_loss / num_samples

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    num_samples = 0

    for batch_waveforms in dataloader:
        batch_waveforms = batch_waveforms.to(device)
        means, stds = model(batch_waveforms, cur_gt=None)
        loss = model.casual_loss(means, stds, batch_waveforms)

        batch_size = batch_waveforms.size(0)
        total_loss += loss.item() * batch_size
        num_samples += batch_size

    return total_loss / num_samples


###############################################################################
# 5) Main
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="NetworkNoise3 Training with Diffusion Noise Augmentation")
    parser.add_argument('--data_dir', type=str, default="/data/ephraim/datasets/DNS-Challenge_old/datasets/noise",
                        help="Path to directory containing .wav files.")
    parser.add_argument('--test_files', type=str, nargs='*', default=[],
                        help="List of filenames that must go to the test set (space-separated).")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_model_path', type=str, default="/data/ephraim/datasets/known_noise/undiff_exps/training_all/all_noises_net3/model.pth",
                        help="Path to save model state dict (e.g. 'model.pth').")
    parser.add_argument('--save_test_list', type=str, default="/data/ephraim/datasets/known_noise/undiff_exps/training_all/all_noises_net3/testset.txt",
                        help="If provided, save the test file paths to this text file.")
    parser.add_argument('--diffusion_steps', type=int, default=200,
                        help="Number of diffusion timesteps for the noise schedule.")
    parser.add_argument('--noise_schedule', type=str, default="linear",
                        help="Type of beta schedule: 'linear' or 'cosine'.")
    args = parser.parse_args()

    # 1) Gather all .wav files
    all_wav_paths = sorted(glob.glob(os.path.join(args.data_dir, '*.wav')))

    # 2) Split train/test
    test_paths = []
    train_paths = []
    test_basenames = set(args.test_files)

    for path in all_wav_paths:
        basename = os.path.basename(path)
        if basename in test_basenames:
            test_paths.append(path)
        else:
            train_paths.append(path)

    print(f"Found {len(train_paths)} training files, {len(test_paths)} test files.")

    # Optionally save the test list
    if args.save_test_list:
        with open(args.save_test_list, 'w', encoding='utf-8') as f:
            for tpath in test_paths:
                f.write(tpath + "\n")

    # 3) Create the diffusion schedule
    betas = get_named_beta_schedule(args.noise_schedule, args.diffusion_steps)  # (diffusion_steps,)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_t = torch.from_numpy(alphas_cumprod).float()
    g_t = torch.sqrt((1.0 - alphas_cumprod_t) / alphas_cumprod_t)

    # 4) Create Datasets
    train_dataset = AudioDataset(train_paths, g_t=g_t)
    test_dataset  = AudioDataset(test_paths, g_t=g_t)

    # 5) Create DataLoaders
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              collate_fn=collate_fn)

    # 6) Instantiate NetworkNoise3
    model = NetworkNoise3(kernel_size=9)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Track global steps for saving
    global_step = [0]

    # 7) Training loop
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            step_counter=global_step,
            save_path=args.save_model_path
        )
        if len(test_paths) > 0:
            test_loss = evaluate(model, test_loader, device)
            print(f"Epoch {epoch+1}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | No test set provided.")

    # Final test eval (optional)
    if len(test_paths) > 0:
        final_test_loss = evaluate(model, test_loader, device)
        print(f"Final Test Loss: {final_test_loss:.4f}")

    # 8) Save final model
    if args.save_model_path:
        print(f"Saving final model to {args.save_model_path}")
        torch.save(model.state_dict(), args.save_model_path)


if __name__ == "__main__":
    main()

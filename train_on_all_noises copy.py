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
        # Example of a "cosine" schedule:
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"Unknown beta schedule: {schedule_name}")


###############################################################################
# 2) WaveNet Implementation
###############################################################################
class CausalConv1d(nn.Conv1d):
    """
    A simple causal convolution layer.
    Expects input shape (B, C_in, T).
    We achieve 'causal' behavior by left-padding and cutting off the right.
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
        out = super().forward(x)
        # Remove the trailing padding on the right
        if self.padding[0] > 0:
            out = out[:, :, :-self.padding[0]]
        return out


class GatedResidualBlock(nn.Module):
    """
    A WaveNet-like residual block with:
      - 1x1 conv to expand channels
      - gated activation (tanh * sigmoid)
      - skip connection to an external 'skip' path
      - residual connection back to input
    """
    def __init__(self, 
                 in_channels, 
                 residual_channels, 
                 skip_channels, 
                 kernel_size, 
                 dilation):
        super().__init__()

        # 1x1 to transform from in_channels -> residual_channels
        self.conv_in = nn.Conv1d(in_channels, residual_channels, kernel_size=1)

        # Gated convolution layers
        self.conv_filter = CausalConv1d(
            residual_channels,
            residual_channels,
            kernel_size=kernel_size,
            dilation=dilation
        )
        self.conv_gate   = CausalConv1d(
            residual_channels,
            residual_channels,
            kernel_size=kernel_size,
            dilation=dilation
        )

        # 1x1 conv for skip connection
        self.conv_skip = nn.Conv1d(residual_channels, skip_channels, kernel_size=1)

        # 1x1 conv for residual
        self.conv_out = nn.Conv1d(residual_channels, in_channels, kernel_size=1)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: (B, in_channels, T)
        returns: (residual_out, skip_out)
        """
        # Project up to residual_channels
        residual = self.conv_in(x)  # (B, residual_channels, T)

        # Gated activation
        filter_out = self.tanh(self.conv_filter(residual))
        gate_out   = self.sigmoid(self.conv_gate(residual))
        gated = filter_out * gate_out  # (B, residual_channels, T)

        # Skip connection
        skip_out = self.conv_skip(gated)  # (B, skip_channels, T)

        # Residual connection (project back to in_channels)
        residual_out = self.conv_out(gated) + x  # (B, in_channels, T)

        return residual_out, skip_out


class WaveNet(nn.Module):
    """
    A WaveNet-like model that predicts the mean and std (via log_var) for
    each audio sample.
    """
    def __init__(self,
                 in_channels=1,    # 1D audio
                 out_channels=2,   # predict (mean, log_var)
                 residual_channels=32,
                 skip_channels=64,
                 kernel_size=3,
                 dilation_depth=8,  # number of layers in a stack
                 num_stacks=1):
        super().__init__()
        self.kernel_size = kernel_size

        self.blocks = nn.ModuleList()

        # Build the stacks of dilated conv blocks
        for _ in range(num_stacks):
            for i in range(dilation_depth):
                dilation = 2 ** i
                block = GatedResidualBlock(
                    in_channels=in_channels,
                    residual_channels=residual_channels,
                    skip_channels=skip_channels,
                    kernel_size=kernel_size,
                    dilation=dilation
                )
                self.blocks.append(block)

        # Final layers from skip-connection
        self.skip_conv1 = nn.Conv1d(skip_channels, skip_channels, kernel_size=1)
        self.skip_conv2 = nn.Conv1d(skip_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        x: (B, 1, T)
        returns: means: (B, T), stds: (B, T)
        """
        skip_accumulator = 0
        out = x  # shape: (B, 1, T)

        for block in self.blocks:
            out, skip = block(out)
            skip_accumulator = skip_accumulator + skip

        skip = self.relu(self.skip_conv1(skip_accumulator))
        output = self.skip_conv2(skip)  # (B, 2, T)

        means   = output[:, 0, :]
        log_var = output[:, 1, :]
        stds    = torch.exp(0.5 * log_var)

        return means, stds

    def gaussian_nll_loss(self, means, stds, target):
        """
        Negative log-likelihood under N(means, stds).
        target: (B, 1, T)
        means/stds: (B, T)
        """
        # Squeeze channel dimension if shape is (B,1,T)
        target = target.squeeze(1)  # (B, T)

        # Minimal trimming by kernel_size to account for initial context
        trim = self.kernel_size
        target = target[:, trim+1:]
        means_ = means[:, trim:-1]
        stds_  = stds[:, trim:-1]

        # log N(x|mu,sigma) = - log(sigma*sqrt(2pi)) - 0.5*((x-mu)/sigma)^2
        log_coef = -torch.log(stds_ * np.sqrt(2*np.pi))
        log_exp  = -0.5 * ((target - means_) ** 2) / (stds_ ** 2)

        log_prob_per_timestep = log_coef + log_exp  # (B, T_after_trim)
        log_prob_per_seq = torch.sum(log_prob_per_timestep, dim=1)  # sum over time
        avg_log_prob = torch.mean(log_prob_per_seq, dim=0)          # mean over batch

        return -avg_log_prob  # negative log-likelihood


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

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        waveform, sr = torchaudio.load(file_path)  # shape: (channels, T)

        # If stereo or multi-channel, mix down to mono
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Optional: apply transform (e.g., normalization)
        if self.transform:
            waveform = self.transform(waveform, sr)

        # ---- Diffusion Noise Augmentation ----
        # Pick a random index i from g_t
        i = random.randint(0, len(self.g_t) - 1)
        # Create Gaussian noise of the same shape as waveform
        cur_white_noise_diffusion = torch.randn_like(waveform)
        # Add the noise scaled by g_t[i]
        waveform = waveform + cur_white_noise_diffusion * self.g_t[i]

        return waveform  # shape: (1, T)

def collate_fn(batch):
    """
    Collate a list of waveforms (each shape: (1, T_i)) into 
    a single tensor (B, 1, T_max), zero-padding shorter waveforms if needed.
    """
    # Find max length in this batch
    max_len = max(waveform.shape[-1] for waveform in batch)

    # Initialize a tensor of shape (B, 1, max_len) with zeros
    batch_size = len(batch)
    batched_waveforms = torch.zeros(batch_size, 1, max_len, dtype=batch[0].dtype)

    for i, waveform in enumerate(batch):
        length = waveform.shape[-1]
        batched_waveforms[i, :, :length] = waveform

    return batched_waveforms

###############################################################################
# 4) Training & Evaluation Functions
###############################################################################
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    num_samples = 0

    for batch_waveforms in tqdm(dataloader):
        batch_waveforms = batch_waveforms.to(device)  # (B, 1, T)
        means, stds = model(batch_waveforms)

        loss = model.gaussian_nll_loss(means, stds, batch_waveforms)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = batch_waveforms.size(0)
        total_loss += loss.item() * batch_size
        num_samples += batch_size

    return total_loss / num_samples

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    num_samples = 0

    for batch_waveforms in dataloader:
        batch_waveforms = batch_waveforms.to(device)
        means, stds = model(batch_waveforms)
        loss = model.gaussian_nll_loss(means, stds, batch_waveforms)

        batch_size = batch_waveforms.size(0)
        total_loss += loss.item() * batch_size
        num_samples += batch_size

    return total_loss / num_samples

###############################################################################
# 5) Main
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="WaveNet Training for Mean/Std Prediction with Diffusion Noise Augmentation")
    parser.add_argument('--data_dir', type=str,  default="/data/ephraim/datasets/DNS-Challenge_old/datasets/noise",
                        help="Path to directory containing .wav files.")
    parser.add_argument('--test_files', type=str, nargs='*', default=['wntLte49djU.wav', 'mPlJSgPoiAw.wav', '1DUIzBDv17s.wav', 'ycHlCbP3Gvc.wav', 'uQl3_7PRgiU.wav', 'door_Freesound_validated_458454_3.wav', 'fan_Freesound_validated_361372_19.wav', 'door_Freesound_validated_439434_0.wav', 'door_Freesound_validated_385420_2.wav', 'breath_spit_Freesound_validated_26803_1.wav', 'zRhCXaEYN6I.wav', 'yH4huWPvzfM.wav', 'door_Freesound_validated_179351_3.wav', '2ErbvVnLS3Q.wav', 'PwnYHHLddCM.wav', 'FPKLZ3tHdkU.wav', 'door_Freesound_validated_323558_0.wav', 'iBXl2PXRb-8.wav', 'c257oj8370c.wav', 'fan_Freesound_validated_329714_0.wav', 'R4J9yOJFkb8.wav', '8TI_QD0vvQ4.wav', '1BonlocdKno.wav', 'OFVzrakJhbw.wav', 'XcIpvyl4es0.wav', 'NeXK6-kYUzA.wav', 'typing_Freesound_validated_390343_7.wav', 'QMYTtaizBCI.wav', 'LoiPr_bDqow.wav', 'cp-cFndaRcM.wav', '3ezEit7AyZo.wav', 'eMVevP1mwt8.wav', 's_dSo-zSGDg.wav', 'fCe9bJVte3k.wav', 'LohqmNzxccQ.wav'],
                        help="List of filenames that must go to the test set (space-separated).")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--dilation_depth', type=int, default=8)
    parser.add_argument('--num_stacks', type=int, default=3)
    parser.add_argument('--residual_channels', type=int, default=32)
    parser.add_argument('--skip_channels', type=int, default=64)
    parser.add_argument('--save_model_path', type=str, default="/data/ephraim/datasets/known_noise/undiff_exps/training_all/all_noises/model.pth",
                        help="If provided, save the trained model state dict here (e.g. 'model.pth').")
    parser.add_argument('--save_test_list', type=str, default="/data/ephraim/datasets/known_noise/undiff_exps/training_all/all_noises/testset.txt",
                        help="If provided, save the test file paths to this text file.")
    parser.add_argument('--diffusion_steps', type=int, default=200,
                        help="Number of diffusion timesteps for the noise schedule.")
    parser.add_argument('--noise_schedule', type=str, default="linear",
                        help="Type of beta schedule: 'linear' or 'cosine'.")
    args = parser.parse_args()

    # 1) Gather all .wav files from data_dir
    all_wav_paths = sorted(glob.glob(os.path.join(args.data_dir, '*.wav')))

    # 2) Separate test files (exact filename match) from training
    test_paths = []
    train_paths = []
    test_basenames = set(args.test_files)  # e.g. {"fileA.wav", "fileB.wav"}

    for path in all_wav_paths:
        basename = os.path.basename(path)
        if basename in test_basenames:
            test_paths.append(path)
        else:
            train_paths.append(path)

    print(f"Found {len(train_paths)} training files, {len(test_paths)} test files.")

    # (Optional) Save the test list to a file
    if args.save_test_list is not None:
        print(f"Saving test file list to {args.save_test_list}")
        with open(args.save_test_list, 'w', encoding='utf-8') as f:
            for test_file in test_paths:
                f.write(test_file + "\n")

    # 3) Create the diffusion schedule
    betas = get_named_beta_schedule(args.noise_schedule, args.diffusion_steps)  # shape: (diffusion_steps,)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)  # shape: (diffusion_steps,)
    alphas_cumprod_t = torch.from_numpy(alphas_cumprod).float()  # convert to torch
    # g_t = sqrt( (1 - alpha_cumprod) / alpha_cumprod )
    g_t = torch.sqrt((1.0 - alphas_cumprod_t) / alphas_cumprod_t)

    # 4) Create Datasets
    train_dataset = AudioDataset(train_paths, g_t=g_t)
    test_dataset  = AudioDataset(test_paths,  g_t=g_t)

    # 5) Create DataLoaders
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              collate_fn=collate_fn)

    # 6) Instantiate the model
    model = WaveNet(
        in_channels=1,
        out_channels=2,
        residual_channels=args.residual_channels,
        skip_channels=args.skip_channels,
        kernel_size=args.kernel_size,
        dilation_depth=args.dilation_depth,
        num_stacks=args.num_stacks
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 7) Training loop
    if args.save_model_path is not None:
        print(f"Saving model state dict to {args.save_model_path}")
        torch.save(model.state_dict(), args.save_model_path.replace("model",f"model_epoch-1"))
    for epoch in tqdm(range(args.epochs)):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        if len(test_paths) > 0:
            test_loss = evaluate(model, test_loader, device)
            print(f"Epoch {epoch+1}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | No test set provided.")
        if args.save_model_path is not None:
            print(f"Saving model state dict to {args.save_model_path}")
            torch.save(model.state_dict(), args.save_model_path.replace("model",f"model_epoch{epoch}"))

    # Final test evaluation (optional)
    if len(test_paths) > 0:
        final_test_loss = evaluate(model, test_loader, device)
        print(f"Final Test Loss: {final_test_loss:.4f}")

    # 8) (Optional) Save model
    if args.save_model_path is not None:
        print(f"Saving model state dict to {args.save_model_path}")
        torch.save(model.state_dict(), args.save_model_path)


if __name__ == "__main__":
    main()

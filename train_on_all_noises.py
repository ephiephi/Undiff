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
import logging

def setup_logging(model_path):
    """Set up logging to save logs in the same directory as the model."""
    model_dir = os.path.dirname(model_path)
    os.makedirs(model_dir, exist_ok=True)  # Ensure the directory exists

    log_file = os.path.join(model_dir, "training_log.txt")

    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),  # Save logs to file
            logging.StreamHandler()  # Print logs to console
        ]
    )

    return log_file

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


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load model and optimizer state from checkpoint if available."""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        step_counter = checkpoint['step_counter']
        logging.info(f"Resumed training from checkpoint: {checkpoint_path}, Step {step_counter}")
        return step_counter
    else:
        logging.info("No checkpoint found, starting training from scratch.")
        return 0  # Start from step 0 if no checkpoint

def save_checkpoint(model, optimizer, checkpoint_path, step_counter):
    """Save model, optimizer, and step counter to checkpoint."""
    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'step_counter': step_counter
    }
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"Checkpoint saved at step {step_counter}: {checkpoint_path}")



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
import torch
import torch.nn as nn



###############################################################################
# 3) AudioDataset with Diffusion-Based Noise Augmentation
###############################################################################
class AudioDataset(Dataset):
    def __init__(self, file_list, g_t, transform=None, add_diffusion_noise=False):
        """
        file_list: list of paths to .wav files
        g_t: a 1D tensor containing values for the diffusion noise scaling
        transform: optional transform (e.g. normalization)
        """
        super().__init__()
        self.file_list = file_list
        self.transform = transform
        self.g_t = g_t  # shape: (num_diffusion_timesteps,)
        self.add_diffusion_noise = add_diffusion_noise

        # Max length to avoid excessive RAM
        self.max_length = 320000
        # self.max_length = 160000

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

        if self.add_diffusion_noise:
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
def train_one_epoch(model, dataloader, optimizer, device, step_counter, save_path=None, test_loader=None,mog=None):
    model.train()
    total_loss = 0.0
    num_samples = 0

    for batch_waveforms in tqdm(dataloader):
        batch_waveforms = batch_waveforms.to(device)  # (B, 1, T)

        
        if mog:
            logits, means, log_sig = model(batch_waveforms, cur_gt=None)
            loss = model.casual_loss(logits, means, log_sig, batch_waveforms)
        else:
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
        step_counter += 1
        

        # Save every 1000 steps
        if save_path is not None and ((step_counter % 1000) == 0 or (step_counter ==10)):
            current_step_path = save_path.replace("model", f"model_step{step_counter}")
            logging.info(f"[Step {step_counter}] Saving model to {current_step_path}")
            # torch.save(model.state_dict(), current_step_path)
            save_checkpoint(model, optimizer, current_step_path, step_counter)
        if (step_counter % 10000) == 0:
            test_loss = evaluate(model, test_loader, device,mog)
            logging.info(f"[Step {step_counter}] | test loss: {test_loss}")

    return total_loss / num_samples, step_counter

@torch.no_grad()
def evaluate(model, dataloader, device,mog):
    model.eval()
    total_loss = 0.0
    num_samples = 0

    for batch_waveforms in dataloader:
        batch_waveforms = batch_waveforms.to(device)
        
        
        if mog:
            logits, means, log_sig = model(batch_waveforms, cur_gt=None)
            loss = model.casual_loss(logits, means, log_sig, batch_waveforms)
        else:
            means, stds = model(batch_waveforms, cur_gt=None)
            loss = model.casual_loss(means, stds, batch_waveforms)            
        # means, stds = model(batch_waveforms, cur_gt=None)
        # loss = model.casual_loss(means, stds, batch_waveforms)

        batch_size = batch_waveforms.size(0)
        total_loss += loss.item() * batch_size
        num_samples += batch_size

    return total_loss / num_samples


###############################################################################
# 5) Main
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="NetworkNoise Training with/out Diffusion Noise Augmentation")
    parser.add_argument('--data_dir', type=str, default="/data/ephraim/datasets/DNS-Challenge_old/datasets/noise",
                        help="Path to directory containing .wav files.")
    parser.add_argument('--test_files', type=str, nargs='*', default=['wntLte49djU.wav', 'mPlJSgPoiAw.wav', '1DUIzBDv17s.wav', 'ycHlCbP3Gvc.wav', 'uQl3_7PRgiU.wav', 'door_Freesound_validated_458454_3.wav', 'fan_Freesound_validated_361372_19.wav', 'door_Freesound_validated_439434_0.wav', 'door_Freesound_validated_385420_2.wav', 'breath_spit_Freesound_validated_26803_1.wav', 'zRhCXaEYN6I.wav', 'yH4huWPvzfM.wav', 'door_Freesound_validated_179351_3.wav', '2ErbvVnLS3Q.wav', 'PwnYHHLddCM.wav', 'FPKLZ3tHdkU.wav', 'door_Freesound_validated_323558_0.wav', 'iBXl2PXRb-8.wav', 'c257oj8370c.wav', 'fan_Freesound_validated_329714_0.wav', 'R4J9yOJFkb8.wav', '8TI_QD0vvQ4.wav', '1BonlocdKno.wav', 'OFVzrakJhbw.wav', 'XcIpvyl4es0.wav', 'NeXK6-kYUzA.wav', 'typing_Freesound_validated_390343_7.wav', 'QMYTtaizBCI.wav', 'LoiPr_bDqow.wav', 'cp-cFndaRcM.wav', '3ezEit7AyZo.wav', 'eMVevP1mwt8.wav', 's_dSo-zSGDg.wav', 'fCe9bJVte3k.wav', 'LohqmNzxccQ.wav'],
                        help="List of filenames that must go to the test set (space-separated). or: path to a text file containing the test file paths.")
    # parser.add_argument('--test_files', type=str, nargs='*', default="/data/ephraim/datasets/known_noise/undiff_exps/training_all/pure_noises_netwavenet_mog/testset.txt", \
    #     help="List of filenames that must go to the test set (space-separated). or: path to a text file containing the test file paths.")
    
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--network', type=str, default="NetworkNoiseWaveNetMoG")
    parser.add_argument('--save_model_path', type=str, default="/data/ephraim/datasets/known_noise/undiff_exps/training_all/pure_noises_netwavenet_mog/model.pth",
                        help="Path to save model state dict (e.g. 'model.pth').")
    parser.add_argument('--save_test_list', type=str, default=None,
                        help="If provided, save the test file paths to this text file.")
    parser.add_argument('--mog', action='store_true', help="Use mixture of Gaussians")
    parser.add_argument('--no-mog', dest='mog', action='store_false', help="Use single Gaussian")
    parser.set_defaults(mog=True)
    parser.add_argument('--num_mixtures', type=int, default=5,
                        help="Number of mixtures for the mixture of gaussians.")
    parser.add_argument('--diffusion_steps', type=int, default=200,
                        help="Number of diffusion timesteps for the noise schedule.")
    parser.add_argument('--noise_schedule', type=str, default="linear",
                        help="Type of beta schedule: 'linear' or 'cosine'.")

    args = parser.parse_args()
    random.seed(42)
    
    if args.save_test_list == "None":
        args.save_test_list = None
    
    # **Set up logging**
    log_file = setup_logging(args.save_model_path)
    logging.info(f"Logging to: {log_file}")

    # 1) Gather all .wav files
    all_wav_paths = sorted(glob.glob(os.path.join(args.data_dir, '*.wav')))

    # 2) Split train/test
    test_paths = []
    train_paths = []
    defined_test=False
    if os.path.exists(str(args.test_files)) or os.path.exists(str(args.save_test_list)):
        test_path = str(args.test_files)
        if not os.path.exists(str(args.test_files)):
            test_path = args.save_test_list
        with open(test_path, 'r', encoding='utf-8') as f:
            test_basenames = set(os.path.basename(line.strip()) for line in f)
        defined_test = True
        logging.info(f"Loaded {len(test_basenames)} test files from: {args.test_files}")
    else:
        test_basenames = set(args.test_files)

    for path in all_wav_paths:
        basename = os.path.basename(path)
        if basename in test_basenames:
            test_paths.append(path)
        else:
            if defined_test:
                train_paths.append(path)
            else:
                if random.random() < 0.05:
                    test_paths.append(path)
                else:
                    train_paths.append(path)

    logging.info(f"Train Files: {len(train_paths)}, Test Files: {len(test_paths)}")

    # Optionally save the test list
    if args.save_test_list:
        logging.info(f"save test to: {args.save_test_list}")
        os.makedirs(os.path.dirname(args.save_test_list), exist_ok=True)
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

    # 6) Instantiate NetworkNoise
    from create_exp_m import NetworkNoise8,NetworkNoise7, NetworkNoise6,NetworkNoise5,NetworkNoise4, NetworkNoise3, WaveNetCausalModel,NetworkNoiseWaveNetMoG,NetworkNoise6MoG,NetworkNoiseWaveNetMoG2
    from train_on_all_noises_wavenet import WaveNet
    # from train_on_all_noises_8_gt import NetworkNoise8
    network = args.network
    if network == "NetworkNoise8":
        model = NetworkNoise8()
    if network == "NetworkNoise7":
        model = NetworkNoise7()
    if network == "NetworkNoise6":
        model = NetworkNoise6()
    if network == "NetworkNoise5":
        model = NetworkNoise5()
    if network == "NetworkNoise4":
        model = NetworkNoise4()
    if network == "NetworkNoise3":
        model = NetworkNoise3()
    if network == "WaveNetCausalModel":
        model = WaveNetCausalModel()
    if network == "NetworkNoiseWaveNetMoG":
        model = NetworkNoiseWaveNetMoG(num_mixtures=args.num_mixtures)
    if network == "NetworkNoiseWaveNetMoG2":
        model = NetworkNoiseWaveNetMoG2(num_mixtures=args.num_mixtures)
    if network =="NetworkNoise6MoG":
        model = NetworkNoise6MoG(num_mixtures=args.num_mixtures)
    if network == "WaveNet":
        model = WaveNet(
        in_channels=1,
        out_channels=2,
        residual_channels=32,
        skip_channels=64,
        kernel_size=3,
        dilation_depth=8,
        num_stacks=3
    )
    # device = torch.device('cpu')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("deviceis: ", device)
    

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    model.to(device)

    # Track global steps for saving
    global_step = load_checkpoint(model, optimizer, args.save_model_path, device)

    # 7) Training loop
    for epoch in range(args.epochs):
        train_loss,global_step = train_one_epoch(
            model, train_loader, optimizer, device,
            step_counter=global_step,
            save_path=args.save_model_path, test_loader=test_loader,mog=args.mog
        )
        if len(test_paths) > 0:
            test_loss = evaluate(model, test_loader, device, args.mog)
            logging.info(f"Epoch {epoch+1}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
        else:
            logging.info(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | No test set provided.")

    # Final test eval (optional)
    if len(test_paths) > 0:
        final_test_loss = evaluate(model, test_loader, device,args.mog)
        logging.info(f"Final Test Loss: {final_test_loss:.4f}")

    # 8) Save final model
    if args.save_model_path:
        save_checkpoint(model, optimizer, args.save_model_path, global_step)


if __name__ == "__main__":
    main()

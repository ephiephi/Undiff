import pickle
import pandas as pd
import numpy as np
from IPython.display import Audio
import torch
import torchaudio
from IPython.display import Audio, display
import json
import os
from tqdm import tqdm
from glob import glob
from pathlib import Path
import argparse
from asteroid.metrics import get_metrics

from DNSMOS import dnsmos_local

from tqdm import tqdm

# noisy_dir = "/data/ephraim/datasets/known_noise/sgmse/exp_i/noisy_synth_i95/"
# clean_dir = "/data/ephraim/datasets/known_noise/sgmse/exp_i/clean_synth_i95/"
# root = Path("/data/ephraim/datasets/known_noise/sgmse/exp_i/")
# models = ["sgmse_WSJ0Chime3", "sgmse_TIMITChime3","storm_TIMITChime3","storm_vbd"]
# enhanced_dirs = [str(root/model_/"enhanced/") for model_ in models]

noisy_dir = "/data/ephraim/datasets/known_noise/sgmse/exp_i/noisy_i_real/"
clean_dir = "/data/ephraim/datasets/known_noise/sgmse/exp_i/clean_i_real/"
root = Path("/data/ephraim/datasets/known_noise/sgmse/exp_i/")
models = ["sgmse_WSJ0Chime3", "sgmse_TIMITChime3","storm_TIMITChime3","storm_vbd"]
enhanced_dirs = [str(root/model_/"enhanced_real/") for model_ in models]

print(enhanced_dirs)

data=[]
for enhanced_dir in tqdm(enhanced_dirs): 
    mos_args = argparse.Namespace(
            testset_dir=enhanced_dir, personalized_MOS=False, csv_path=None)
    dns_df_all = dnsmos_local.main(mos_args)

    stats_path =os.path.join(enhanced_dir, "measures_storm.pickle")

    wavs_ = glob(enhanced_dir+"/*.wav")
    # print(wavs_)
    if len(wavs_)>0:
        ours_files = glob(enhanced_dir+"/*.wav")

        for ours in ours_files:
            enhance, en_sr = torchaudio.load(ours)
            name = Path(ours).name
            clean_path = Path(clean_dir)/name
            speech_wav, _sr = torchaudio.load(clean_path)
            noisy_path =  Path(noisy_dir)/name
            noisy_wav, _sr = torchaudio.load(noisy_path)
            # Compute metrics
            metrics = get_metrics(clean=speech_wav[0].numpy(), mix=noisy_wav[0].numpy(),estimate=enhance[0].numpy(), sample_rate=en_sr)
            metrics["name"] = Path(ours).name[2:]
            metrics["filename"] = (ours)
            metrics["dir"] = str(Path(ours).name).split("_")[0]
            metrics["snr"] = str(Path(ours).name).split("snr")[1].split("/")[0]
            if "noise" in Path(ours).name:
                print("noise")
                metrics["noise_type"] = Path(ours).name.split("noise")[1].split("_")[0]
                # print(Path(ours).name.split("noise")[1].split("_")[0])
            # Print metrics
            # print(metrics)
            data.append(metrics)
            
        if len(data) >0:
            df = pd.DataFrame.from_records(data,index=range(len(data))) 
            df = df.merge(dns_df_all, on="filename")
        with open(stats_path, "wb") as f:
            pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
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


def main_measure(exp_dir):
    enhanced_dir =  Path(exp_dir)/"enhanced_60"
    clean_dir = Path(exp_dir)/"clean_wav"
    noisy_dir =  Path(exp_dir)/"noisy_wav"
    snr_dirs = glob(os.path.join(enhanced_dir,"*/"))
    df = calc_measures(exp_dir,enhanced_dir,clean_dir,noisy_dir,snr_dirs)
    
    stats_path =os.path.join(exp_dir, "measures.pickle")
    with open(stats_path, "wb") as f:
        pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)



def calc_measures(exp_dir,enhanced_dir,clean_dir,noisy_dir,snr_dirs):
    mos_args = argparse.Namespace(
            testset_dir=enhanced_dir, personalized_MOS=False, csv_path=None)
    dns_df_all = dnsmos_local.main(mos_args)

    data=[]
    for snr_dir in tqdm(snr_dirs): 
        s_dirs =glob(snr_dir+"*/")
        for i,path_ in tqdm(enumerate(s_dirs)):
            ours_files = glob(path_+"*.wav")
            for ours in ours_files:
                # print(glob(path_+"*.wav"))
                enhance, en_sr = torchaudio.load(ours)
                name = Path(ours).name
                clean_path = Path(clean_dir)/name
                speech_wav, _sr = torchaudio.load(clean_path)
                noisy_path =  Path(noisy_dir)/name
                noisy_wav, _sr = torchaudio.load(noisy_path)
                # Compute metrics
                metrics = get_metrics(clean=speech_wav[0].numpy(), mix=noisy_wav[0].numpy(),estimate=enhance[0].numpy(), sample_rate=en_sr)
                metrics["name"] = Path(path_).name
                metrics["filename"] = (ours)
                metrics["snr"] = ours.split("snr")[1].split("/")[0]
                if "noise" in Path(ours).name:
                    metrics["noise_type"] = str(Path(ours).name).split("noise")[1].split("_")[0]
                # Print metrics
                # print(metrics)
                data.append(metrics)
        if len(data) >0:
            df = pd.DataFrame.from_records(data,index=range(len(data))) 
        df = df.merge(dns_df_all, on="filename")
        
        data_noisy = []
        noisy_wavs_ = glob(str(noisy_dir)+"/*.wav")
        for i,cur_noisy_path in tqdm(enumerate(noisy_wavs_)):
            noisy_wav, en_sr = torchaudio.load(cur_noisy_path)
            name = Path(cur_noisy_path).name
            clean_path = Path(clean_dir)/name
            speech_wav, _sr = torchaudio.load(clean_path)
            
            # Compute metrics
            metrics = get_metrics(clean=speech_wav[0].numpy(), mix=noisy_wav[0].numpy(),estimate=noisy_wav[0].numpy(), sample_rate=en_sr)
            metrics["name"] = "clean"
            metrics["filename"] = (cur_noisy_path)
            metrics["snr"] = cur_noisy_path.split("_snr")[1].split("_")[0]
            if "noise" in Path(cur_noisy_path).name:
                metrics["noise_type"] = str(Path(cur_noisy_path).name).split("noise")[1].split("_")[0]
            # Print metrics
            # print(metrics)
            data_noisy.append(metrics)
            #[sgmse2 ,sgmse1,storm1,storm2,ours ]
        if len(data_noisy) >0:
            df_noisy = pd.DataFrame.from_records(data_noisy,index=range(len(data_noisy))) 
            
        mos_args = argparse.Namespace(
            testset_dir=str(noisy_dir), personalized_MOS=False, csv_path=None
        )
        dns_df_noisy = dnsmos_local.main(mos_args)
        df_noisy = df_noisy.merge(dns_df_noisy, on="filename")
        
        df = pd.concat([df,df_noisy])
        return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="measure guided")
    parser.add_argument(
        "-exp_dir",
        default="/data/ephraim/datasets/known_noise/undiff/exp_ar_i_095/c/",
    )


    args = parser.parse_args()
    main_measure(
        args.exp_dir,
    )

import pickle
import pandas as pd
import numpy as np
from IPython.display import Audio
import torch
import torchaudio
from IPython.display import Audio, display
import json
import os
from pathlib import Path
import shutil
from glob import glob



def get_df(exp_root, dir):
    exp_dir = Path(exp_root)/dir
    pkl_results_file = os.path.join(exp_dir, "measures.pickle")

    with open(pkl_results_file, "rb") as handle:
        dfme = pd.read_pickle(handle)
    return dfme


def get_df_storm(pkl_results_file, snr=None):
    # pkl_results_file = "/data/ephraim/datasets/known_noise/sgmse/exp_l/enhanced/storm_vbd/measures_storm.pickle"
    # pkl_results_file = "/data/ephraim/datasets/known_noise/sgmse/exp_k/enhanced/storm_TIMITChime3/measures_storm.pickle"
        # pkl_results_file = "/data/ephraim/datasets/known_noise/sgmse/exp_l/enhanced/sgmse_WSJ0Chime3/measures_storm.pickle"
    # pkl_results_file = "/data/ephraim/datasets/known_noise/sgmse/exp_k/enhanced/sgmse_TIMITChime3/measures_storm.pickle"
    with open(pkl_results_file, "rb") as handle:
        df2 = pd.read_pickle(handle)
    for i in range(len(df2["snr"])):
        df2.at[i, "snr"] = df2["snr"][i].split("_")[0]
    if snr is None:
        df_storm_ = df2.reset_index(drop=True).sort_values(by=['dir'])
    else:
        df_storm_ = df2[df2["snr"]==snr].reset_index(drop=True).sort_values(by=['dir'])
    return df_storm_


def write_results(df, name_alg,analysis_root,noises):
    df.to_excel(os.path.join(analysis_root, f"{name_alg}_all.xlsx"))
    for noisetype in noises:
        noise_mine  = df[df["noise_type"]==noisetype]
        noise_mine.to_excel(os.path.join(analysis_root, f"{name_alg}_{noisetype}.xlsx"))


def drop_rows_without_comparison(df, ours_df):
    for i in range(len(df["dir"])):
        dir_ = df["dir"][i]
        noisetype = df["noise_type"][i]
        mine_parallel = ours_df[(ours_df["dir"]==dir_) & (ours_df["noise_type"]==noisetype)]
        if len(mine_parallel)==0:
            idx = df[(df.dir==dir_)&(df.noise_type==noisetype)].index
            df = df.drop(idx)
        return df


def get_stats_df(df, alg_name, dir_=None, noise_type=None, snr_=None):
    if dir_:
        df = df[df["dir"]==dir_]
    if noise_type:
        df = df[df["noise_type"]==noise_type]
    if snr_:
        df = df[df["snr"]==snr_]
    dfstats = df.describe()[1:3]
    dfstats = dfstats.assign(alg=alg_name)
    return dfstats


def create_mine_df(exp_root,df_noisy,mine,noises,cols,names,snrs):
    for d in names: 
        dfme = get_df(exp_root,d)
        if dfme is None:
            continue
        dfme = dfme.assign(dir=d)
        cur_df_noisy = dfme[dfme["name"]=="noisy"].reset_index(drop=True)
        cur_df_noisy.loc[0,"dir"] = d
        if df_noisy is None:
            df_noisy = cur_df_noisy
        else:
            df_noisy = pd.concat([df_noisy, cur_df_noisy])
        dfme = dfme[dfme["name"] != "noisy"]
        
        dfme = dfme[dfme["stoi"]>0.5]

        for noisetype in noises :
            for c_snr in snrs:
                c_snr=str(c_snr)
                dfme_cur = dfme[dfme["snr"]==c_snr]
                cur_mine = dfme_cur[dfme_cur["noise_type"]==noisetype]
                cur_mine = cur_mine[cur_mine["pesq"] ==cur_mine["pesq"].max()].reset_index(drop=True)[cols]
                if mine is None:
                    mine = cur_mine
                else:
                    mine = pd.concat([mine, cur_mine])
    mine=mine.reset_index(drop=True)
    return mine, df_noisy

def copy_wavs(df,wavdst,algname):
    df = df.reset_index(drop=True)
    for i in df.index.values:
        wavpath = df["filename"][i]
        dir_ = df["dir"][i]
        snr = df["snr"][i]
        noisetype = df["noise_type"][i]
        name = f"{dir_}_{noisetype}_snr{snr}_{algname}.wav"
        dst = wavdst/f"{name}"
        print("origin:", wavpath)
        print("dst:", dst)
        shutil.copyfile(wavpath, dst)


from audio_tools2  import *
def calc_vad(f, verbose=False):
    test_file=f
    fs,s = read_wav(test_file)
    win_len = int(fs*0.025)
    hop_len = int(fs*0.010)
    sframes = enframe(s,win_len,hop_len) # rows: frame index, cols: each frame
    if verbose:
        plot_this(compute_log_nrg(sframes))

    # percent_high_nrg is the VAD context ratio. It helps smooth the
    # output VAD decisions. Higher values are more strict.
    percent_high_nrg = 0.5

    vad = nrg_vad(sframes,percent_high_nrg)

    if verbose:
        plot_these(deframe(vad,win_len,hop_len),s)
    return deframe(vad,win_len,hop_len) 


def analyze_exp(exp_root,noises_names,snrs,names):
    
    storm_enhanced_path = str( Path(exp_root)/"storm/enhanced/")
    storm_clean_wav = str(Path(exp_root)/"storm"/"clean_wav")
    NOISES = noises_names

    cols =["dir","name","stoi","input_stoi","pesq","input_pesq","OVRL","SIG","BAK","si_sdr",'sdr', "sar",'sir',"noise_type", "filename","snr"]
    mine = None
    df_noisy = None
    mine, df_noisy = create_mine_df(exp_root,df_noisy,mine,noises=NOISES,cols=cols,names=names,snrs=snrs)
    
    analysis_root = os.path.join(exp_root, "analysis")
    if not os.path.exists(analysis_root):
        os.mkdir(analysis_root)
    
    SGMSE = "sgmseWSJ0"
    STORM = "sgmseTIMIT"
    storm_results_path = os.path.join(storm_enhanced_path,"sgmse_TIMITChime3/measures_storm.pickle")
    sgmse_results_path = os.path.join(storm_enhanced_path,"sgmse_WSJ0Chime3/measures_storm.pickle")
    df_storm = get_df_storm(storm_results_path)
    df_sg = get_df_storm(sgmse_results_path)
    write_results(mine,"ours",analysis_root=analysis_root, noises=NOISES)
    write_results(df_sg,SGMSE,analysis_root=analysis_root, noises=NOISES)
    write_results(df_storm,STORM,analysis_root=analysis_root, noises=NOISES)

    df_storm = drop_rows_without_comparison(df_storm, mine)
    df_sg= drop_rows_without_comparison(df_sg, mine)


    minestats = get_stats_df(mine, "ours")
    df_storm_stats = get_stats_df(df_storm, STORM)
    df_sg_stats = get_stats_df(df_sg, SGMSE)
    statsdf = pd.concat([minestats,df_sg_stats,df_storm_stats])
    statsdf_path = os.path.join(analysis_root,"all_stats.xlsx")
    statsdf.to_excel(statsdf_path)
    

    for dir__ in names:
        dirminestats = get_stats_df(mine, "ours", dir__)
        dir_storm_stats =  get_stats_df(df_storm[cols], STORM, dir__)
        dir_sg_stats =  get_stats_df(df_sg[cols], SGMSE, dir__)
        dirstatsdf = pd.concat([dirminestats,dir_storm_stats,dir_sg_stats])
        dir_statsdf_path = os.path.join(analysis_root,f"signal{dir__}_stats.xlsx")
        dirstatsdf.to_excel(dir_statsdf_path) 


    for noisetype in NOISES:
        minestats = get_stats_df(mine, "ours", noise_type=noisetype)
        df_storm_stats = get_stats_df(df_storm[cols], STORM, noise_type=noisetype)
        df_sg_stats = get_stats_df(df_sg[cols], SGMSE, noise_type=noisetype)
        statsdf = pd.concat([minestats,df_sg_stats,df_storm_stats])
        noise_statsdf_path = os.path.join(analysis_root,f"stats_noise{noisetype}.xlsx")
        statsdf.to_excel(noise_statsdf_path)
        
    for c_snr in snrs:
        c_snr = str(c_snr)
        minestats = get_stats_df(mine, "ours", snr_=c_snr)
        df_storm_stats = get_stats_df(df_storm[cols], STORM, snr_=c_snr)
        df_sg_stats = get_stats_df(df_sg[cols], SGMSE, snr_=c_snr)
        statsdf = pd.concat([minestats,df_sg_stats,df_storm_stats])
        snr_statsdf_path = os.path.join(analysis_root,f"stats_snr{c_snr}.xlsx")
        statsdf.to_excel(snr_statsdf_path)
    
    wavs_analysis_path = Path(exp_root)/"analysis"/"wavs"
    if not os.path.exists(wavs_analysis_path):
        os.mkdir(wavs_analysis_path)
    
    
    copy_wavs(mine, wavs_analysis_path, "ours")
    copy_wavs(df_sg, wavs_analysis_path, SGMSE)
    copy_wavs(df_storm, wavs_analysis_path, STORM)
    copy_wavs(df_noisy, wavs_analysis_path, "noisy")

    wavs = glob(str(Path(storm_clean_wav)) + "/*.wav")
    for wavpath in wavs:
        c = Path(wavpath).name.split("_")[0]
        if "snr" in wavpath:
            snr = wavpath.split("snr")[1].split("_")[0]
            noisetype = Path(wavpath).name.split("noise")[1].split("_")[0]
            name = f"{c}_{noisetype}_snr{snr}_clean.wav"
            dst = wavs_analysis_path/f"{name}"
            shutil.copyfile(wavpath, dst)
    
    df = mine    
    for i in df.index.values:
        snr = df["snr"][i]
        noisetype = df["noise_type"][i]
        dir_ = df["dir"][i]
        # s = float(df["name"][i].replace("s",""))
        wavpath = df["filename"][i]
        noisy_path = Path(exp_root) / dir_ / "noisy_wav" / Path(wavpath).name

        speech, sr = torchaudio.load(noisy_path)
        vaded_signal = calc_vad(noisy_path)[0:speech.shape[1],:]
        vaded_signal_torch = (speech[0][vaded_signal.T[0]>0])
        vaded_signal_torch = torch.unsqueeze(vaded_signal_torch, dim=0)
        clean_power = float( 1 / vaded_signal_torch.shape[1] * torch.sum(vaded_signal_torch**2))
        simple_power =  float(1 / speech.shape[1] * torch.sum(speech**2))
        vaded_rate = (vaded_signal_torch.shape[1]/speech.shape[1])
        variance = float( torch.var(speech, unbiased=True))
        df.at[i, "clean_power"] = clean_power
        df.at[i, "simple_power"] = simple_power
        df.at[i, "vaded_rate"] = vaded_rate
        df.at[i, "variance"] = variance
    write_results(mine,"ours_params",analysis_root=analysis_root, noises=NOISES)

if __name__ == '__main__':
    # exp_root = "/data/ephraim/datasets/known_noise/undiff_exps/exp_m_long_ar/"
    # noises_names = ["1","2","3"]
    # snrs = ["5"]
    # names = ["j","b","c"]
    names = []
    exp_root = "/data/ephraim/datasets/known_noise/undiff_exps/exp_n_real/"
    for d in os.listdir(exp_root):
        if not d in ['5f_snrs.pickle', 'storm', 'analysis','noises','noisy_wav','clean_wav',"cleans"]:
            names.append(d)
    NOISES = [str(i) for i in range(35) if i!=5]
    names = NOISES
    snrs = []
    for d in os.listdir(exp_root):
        if not d in ['5f_snrs.pickle', 'storm', 'analysis','noises','noisy_wav','clean_wav']:
            enh_dir = Path(exp_root)/d/"enhanced_60"
            snr_dir=os.listdir(enh_dir)
            for s in snr_dir:
                c_snr = s.split("snr")[1]
                if c_snr not in snrs:
                    snrs.append(c_snr)
    analyze_exp(exp_root,NOISES,snrs,names)
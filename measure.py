from pesq import pesq
from pystoi import stoi
import numpy
import os
import pickle
from tqdm import tqdm
import torch
import torchaudio
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import torchaudio.functional as F

import argparse
import json
from glob import glob

from DNSMOS import dnsmos_local


def main_measure(
    enhance_dir,
    clean_dir,
    out_stats_dir,
    noisy_dir="/data/ephraim/datasets/known_noise/noisy_wav/",
    use_previous=False
):
    # doc_file = os.path.join(enhance_dir, "args.json")
    # with open(doc_file) as json_f:
    #     json_args = json.load(json_f)

    # beta = json_args["mode_beta"]
    # guidance_type = json_args["guidance_type"]
    # grad_time = json_args["grad_time"]
    # guidance_scale = json_args["guidance_scale"]
    # fast = json_args["fast"]
    # noisiest = json_args["noisiest"]
    # noisy_dir = json_args["wav_path"]

    pkl_results_file = os.path.join(enhance_dir, "stats.pickle")

    if os.path.exists(pkl_results_file) and use_previous:
        with open(pkl_results_file, "rb") as handle:
            df = pd.read_pickle(handle)
            stats = df.describe()
            print(stats)

    else:
        df = calc_measures(noisy_dir, clean_dir, enhance_dir)
        stats = df.describe()
        print(stats)

        with open(pkl_results_file, "wb") as f:
            pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)

    dns_pickle = os.path.join(enhance_dir, "dnsmos.pickle")
    if os.path.exists(dns_pickle) and use_previous:
        with open(dns_pickle, "rb") as handle:
            dns_df = pd.read_pickle(handle)
            stats_mos = dns_df.describe()
            print(stats_mos)
    else:
        mos_args = argparse.Namespace(
            testset_dir=enhance_dir, personalized_MOS=False, csv_path=None
        )
        try:
            dns_df = dnsmos_local.main(mos_args)
            stats_mos = dns_df.describe()
            with open(dns_pickle, "wb") as f:
                dns_df.to_pickle(f)
        except:
            stats_mos = {"SIG":{"mean":-1},"BAK":{"mean":-1},"OVRL":{"mean":-1}}
        # print(stats_mos)

        
    documentation =  {"pesq": str(stats["pesq_enhanced"]["mean"]),"stoi": str(stats["stoi_enhanced"]["mean"]),"pesq_diff": str(stats["pesq_diff"]["mean"]),"stoi_diff": str(stats["stoi_diff"]["mean"]), "OVRL": str(stats_mos["OVRL"]["mean"]), "SIG": str(stats_mos["SIG"]["mean"]), "BAK": str(stats_mos["BAK"]["mean"]), "enhance_dir": str(enhance_dir), "args": str(args)}
    
    stats_path =os.path.join(out_stats_dir, "stats_dns.json")
    s_param = os.path.basename(os.path.normpath(enhance_dir))
    if os.path.exists(stats_path):
        with open(stats_path, "r+") as file1:
            file_data = json.load(file1)
            file_data[str(s_param)] = documentation
            # file1.write(documentation)
            file1.seek(0)
            json.dump(file_data, file1)
    else:
        with open(stats_path, "w") as file1:
            file_data = {}
            file_data[str(s_param)] = documentation
            # file1.write(documentation)
            # file1.seek(0)
            json.dump(file_data, file1)



def calc_measures(noisy_dir, clean_dir, enhance_dir):
    noises = os.listdir(noisy_dir)
    dont_calculated = []
    results = {
        "pesq_noisy": {},
        "stoi_noisy": {},
        "pesq_enhanced": {},
        "stoi_enhanced": {},
    }

    i = 0

    for ref_filename in tqdm(noises):
        reference = os.path.join(clean_dir, ref_filename)
        test_noisy = os.path.join(noisy_dir, ref_filename)
        # test_enhanced = os.path.join(enhance_dir, "1000k_0.wav")
        print(enhance_dir)
        test_files = glob(enhance_dir+"*.wav")
        print("test_files:", test_files)
        WAVEFORM_SPEECH, SAMPLE_RATE_SPEECH = torchaudio.load(reference)
        WAVEFORM_NOISE, SAMPLE_RATE_NOISE = torchaudio.load(test_noisy)
        
        stoi_enhanced_array = np.zeros(len(test_files))
        pesq_enhanced_array = np.zeros(len(test_files))
        stoi_enhanced = None
        pesq_enhanced = None
        try:
            for i, test_enhanced in enumerate(test_files):
                WAVEFORM_enhanced, SAMPLE_RATE_enhanced = torchaudio.load(test_enhanced)
                
                if WAVEFORM_SPEECH.shape[1] < WAVEFORM_enhanced.shape[1]:
                    WAVEFORM_enhanced = WAVEFORM_enhanced[:, : WAVEFORM_SPEECH.shape[1]]
                else:
                    WAVEFORM_SPEECH = WAVEFORM_SPEECH[:, : WAVEFORM_enhanced.shape[1]]
                if WAVEFORM_NOISE.shape[1] < WAVEFORM_enhanced.shape[1]:
                        WAVEFORM_enhanced = WAVEFORM_enhanced[:, : WAVEFORM_NOISE.shape[1]]
                else:
                    WAVEFORM_NOISE = WAVEFORM_NOISE[:, : WAVEFORM_enhanced.shape[1]]
                
                pesq_enhanced_array[i] = pesq(
                    16000,
                    WAVEFORM_SPEECH[0].numpy(),
                    WAVEFORM_enhanced[0].numpy(),
                    mode="wb",
                )
                stoi_enhanced_array[i] = stoi(
                    WAVEFORM_SPEECH[0].numpy(),
                    WAVEFORM_enhanced[0].numpy(),
                    16000,
                    extended=False,
                )
                stoi_enhanced = float(np.mean(stoi_enhanced_array))
                pesq_enhanced = float(np.mean(pesq_enhanced_array))
                print("pesq_enhanced_array: ",pesq_enhanced_array)
                print("pesq_enhanced: ",pesq_enhanced)
        except: 
            print("------------- failed --------------", enhance_dir)
            dont_calculated.append(ref_filename)
            stoi_enhanced = -1
            pesq_enhanced = -1
        # # print("Computing scores for ", reference)
        # print("noiseshape: ", WAVEFORM_NOISE.shape)
        # print("speechshape: ", WAVEFORM_SPEECH.shape)
        # print("enhancedshape: " , WAVEFORM_enhanced.shape)
        try:
            pesq_noise = pesq(
                16000,
                WAVEFORM_SPEECH[0].numpy(),
                WAVEFORM_NOISE[0].numpy(),
                mode="wb",
            )
            stoi_noise = stoi(
                WAVEFORM_SPEECH[0].numpy(),
                WAVEFORM_NOISE[0].numpy(),
                16000,
                extended=False,
            )
        except: 
            print("------------- failed --------------", enhance_dir)
            dont_calculated.append(ref_filename)
            pesq_noise = -1
            stoi_noise = -1


        results["pesq_noisy"][ref_filename] = pesq_noise
        results["stoi_noisy"][ref_filename] = stoi_noise

        results["stoi_enhanced"][ref_filename] = stoi_enhanced
        results["pesq_enhanced"][ref_filename] = pesq_enhanced
        df = pd.DataFrame.from_dict(results)
        print("pesq_enhanced: ", pesq_enhanced)
        print("pesq_noisy: ", pesq_noise)
        print(df["pesq_noisy"])
        print(df["pesq_enhanced"])
        df["pesq_diff"] = df["pesq_enhanced"].sub(df["pesq_noisy"])
        df["stoi_diff"] = df["stoi_enhanced"].sub(df["stoi_noisy"])
        # else:
        #     df["pesq_noisy"][ref_filename] = pesq_noise
        #     df["stoi_noisy"][ref_filename] = stoi_noise

        #     df["stoi_enhanced"][ref_filename] = stoi_enhanced
        #     df["pesq_enhanced"][ref_filename] = pesq_enhanced
        #     df["pesq_diff"] = -1
        #     df["stoi_diff"] = -1
        # except:
        #     results["pesq_noisy"][ref_filename] = None
        #     results["stoi_noisy"][ref_filename] = None

        #     results["stoi_enhanced"][ref_filename] = None
        #     results["pesq_enhanced"][ref_filename] = None
        #     df["pesq_diff"] = None
        #     df["stoi_diff"] = None
        #     df = df = pd.DataFrame.from_dict(results)
            
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="measure guided")
    parser.add_argument(
        "-enhanced_dir",
        default="/data/ephraim/datasets/known_noise/enhanced_diffwave_1sec/snr5/s0.001/",
    )
    parser.add_argument(
        "-out_stats_dir",
        default="/data/ephraim/repos/diffwave-sashimi/measure/",
    )
    parser.add_argument(
        "-clean_dir", default="/data/ephraim/datasets/known_noise/clean_wav/"
    )
    parser.add_argument(
        "-noisy_dir", default="/data/ephraim/datasets/known_noise/noisy_wav/"
    )

    args = parser.parse_args()
    main_measure(
        args.enhanced_dir,
        args.clean_dir,
        args.out_stats_dir,
        noisy_dir=args.noisy_dir
    )

import copy
import itertools
import os
import random
from collections import OrderedDict

import numpy as np
import torch
from tqdm import tqdm


def get_tensor_from_list(audio_list):
    return torch.cat(audio_list, dim=0)


def log_results(results_dir, res, name="metrics"):
    print(res)
    file_exp_res = os.path.join(results_dir, f"{name}.txt")
    with open(file_exp_res, "w+") as f:
        for k, v in res.items():
            print(f"{k}/mean: {v[0]:.3f}", file=f)
            print(f"{k}/std: {v[1]:.3f}", file=f)


def create_state_dict_from_ema(state_dict, model, ema_params):
    ema_state_dict = copy.deepcopy(state_dict)
    for i, (name, _) in enumerate(model.named_parameters()):
        ema_state_dict[name] = ema_params[i]

    return ema_state_dict


def remove_prefix_from_state_dict(state_dict, j: int = 1):
    new_state_dict = OrderedDict()
    for k, _ in state_dict.items():
        tokens = k.split(".")
        new_state_dict[".".join(tokens[j:])] = state_dict[k]

    return new_state_dict


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def compute_metric_result(pred_tensor, metrics, real_tensor=None):
    res = {}
    for metric in metrics:
        metric.val_size = pred_tensor.size(0)
        metric.compute(pred_tensor, real_tensor, 0, res)
        metric.save_result(res)
    return res


def calculate_all_metrics(wavs, metrics, n_max_files=None, reference_wavs=None):
    scores = {metric.name: [] for metric in metrics}
    if reference_wavs is None:
        reference_wavs = wavs
    for x, y in tqdm(
        itertools.islice(zip(wavs, reference_wavs), n_max_files),
        total=n_max_files if n_max_files is not None else len(wavs),
        desc="Calculating metrics",
    ):
        try:
            x = x.view(1, 1, -1)
            y = y.view(1, 1, -1)
            for metric in metrics:
                metric._compute(x, y, None, None)
                scores[metric.name] += [metric.result["mean"]]
        except Exception:
            pass
    scores = {k: (np.mean(v), np.std(v)) for k, v in scores.items()}
    return scores


def calc_se_metrics(wavs, metrics, n_max_files=None, reference_wavs=None):
    test_noisy = wavs

    reference = os.path.join(clean_dir, ref_filename)
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
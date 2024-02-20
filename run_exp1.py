from glob import glob
import os


outdirname="enhanced"
root = "/data/ephraim/datasets/known_noise/undiff/exp3b/"
outbasedir = os.path.join(root, outdirname)
noisy_dir = os.path.join(root, "noisy_wav")
noiswavs = glob(noisy_dir+"/*")
print(noiswavs)
clean_dir = os.path.join(root, "clean_wav")


MDL='model=wavenet'

if not os.path.exists:
    os.path.mkdir(outbasedir)
for wav in noiswavs:
    snr = wav.split("snr")[1].split("_power")[0]
    if snr not in ["5", "10", "0"]:
        continue
    OUTPATH = os.path.join(outbasedir,"snr{}".format(snr))


    s_array=[ 0.0001, 0.001]
    for s in s_array: 
        command = "HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 python main.py model=diffwave task=unconditional output_dir=results/guided audio_dir=1 guid_s={} y_noisy={} outpath={}".format(s,wav, OUTPATH)
        os.system(command)
        
# command = "cd measure; python run_measure.py -exp_dir {}  -clean_dir {} -noisy_dir {}".format(outbasedir, clean_dir, noisy_dir)
# os.system(command)
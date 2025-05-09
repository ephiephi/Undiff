from glob import glob
import os
from tqdm import tqdm

s_schedule="50"
outdirname="enhanced"
noise_model_path = "/data/ephraim/repos/Pytorch-VAE-tutorial/exp3_scaled.pickle"
roots = ["/data/ephraim/datasets/known_noise/undiff/exp_ar_5e/b"]
for root in roots:
    outbasedir = os.path.join(root, outdirname)
    noisy_dir = os.path.join(root, "noisy_wav")
    noiswavs = glob(noisy_dir+"/*")
    print(noiswavs)
    clean_dir = os.path.join(root, "clean_wav")


    if not os.path.exists(outbasedir):
        os.mkdir(outbasedir)
    for wav in tqdm(noiswavs):
        snr = wav.split("snr")[1].split("_power")[0]
        
        for j in [0]:
            print("s_sch: ", s_schedule)
            OUTPATH = os.path.join(outbasedir,"snr{}".format(snr))
            if not os.path.exists(OUTPATH):
                os.mkdir(OUTPATH)

            s_array1 = []#[0.07,0.08,0.1,0.12,0.15,0.2,0.22,0.25,0.3]#[0.03,0.05,0.07,0.08,0.09,0.12,0.15]#[0.1,0.1,1,5,7,8,8.5,9]
            s_array2=[0.1,0.]#[0.13,0.155,0.17].14,0.16
            s_array= s_array1 + s_array2
            # s_array = [1]###############
            
            for s in s_array: 
                newOUTPATH = os.path.join(OUTPATH,"s{}".format(s))
                if not os.path.exists(newOUTPATH):
                    os.mkdir(newOUTPATH)
                command = "HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 python main.py model=diffwave task=unconditional output_dir=results/guided audio_dir=1 guid_s={} y_noisy={} outpath={} s_schedule={} noise_type={} noise_model_path={}".format(s,wav, newOUTPATH, s_schedule, "loss_model",noise_model_path)
                print(command)
                os.system(command)
                
    command = "python run_measure.py -exp_dir {} -clean_dir {} -noisy_dir {}".format(outbasedir, clean_dir, noisy_dir)
    os.system(command)
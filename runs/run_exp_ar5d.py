from glob import glob
import os
from tqdm import tqdm

s_schedule=[5.9,6.1] #2.5,3,4.5,7,,10,4,5,5.5,8,9
outdirname="enhanced"
noise_model_path = "/data/ephraim/repos/Pytorch-VAE-tutorial/exp1.pickle"
roots = ["/data/ephraim/datasets/known_noise/undiff/exp_ar_5d/b"]
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
        
        for s_sch in s_schedule:
            print("s_sch: ", s_sch)
            OUTPATH = os.path.join(outbasedir,"snr{}_schdl{}".format(snr,s_sch))
            if not os.path.exists(OUTPATH):
                os.mkdir(OUTPATH)

            s_array1 = [9.4] #[7,7.5,7.8,8,8.2,8.5,9
            s_array2=[]
            s_array= s_array1 + s_array2
            # s_array = [1]###############
            
            for s in s_array: 
                newOUTPATH = os.path.join(OUTPATH,"s{}".format(s))
                if not os.path.exists(newOUTPATH):
                    os.mkdir(newOUTPATH)
                command = "HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 python main.py model=diffwave task=unconditional output_dir=results/guided audio_dir=1 guid_s={} y_noisy={} outpath={} s_schedule={} noise_type={} noise_model_path={}".format(s,wav, newOUTPATH, s_sch, "loss_model",noise_model_path)
                print(command)
                os.system(command)
                
    command = "python run_measure.py -exp_dir {} -clean_dir {} -noisy_dir {}".format(outbasedir, clean_dir, noisy_dir)
    os.system(command)
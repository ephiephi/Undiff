from glob import glob
import os
from tqdm import tqdm
from pathlib import Path


s_schedule="60"
outdirname="enhanced_60"

mainroot = Path("/data/ephraim/datasets/known_noise/undiff_exps/exp_l_long_ar/")
roots = [ mainroot/"j",mainroot/"b",mainroot/"c"] #,h,ij
for root in roots:
    print("root: ", root)
    outbasedir = os.path.join(root, outdirname)
    noisy_dir = os.path.join(root, "noisy_wav")
    noiswavs = glob(noisy_dir+"/*")
    print(noiswavs)
    clean_dir = os.path.join(root, "clean_wav")


    if not os.path.exists(outbasedir):
        os.mkdir(outbasedir)
    for wav in tqdm(noiswavs):
        snr = wav.split("snr")[1].split("_power")[0]
        noise_type = str(Path(wav).name).split("noise")[1].split("_digits")[0]
        noise_model_path = Path(root) / f"0_snr{snr}_{noise_type}_models.pickle"
        print(noise_model_path)
        
        for j in [0]:
            print("s_sch: ", s_schedule)
            OUTPATH = os.path.join(outbasedir,"snr{}".format(snr))
            if not os.path.exists(OUTPATH):
                os.mkdir(OUTPATH)

            s_array1 =[0.08,0.09,0.1,0.11, 0.12,0.15,0.01,0.05,]
            s_array2=[0.2,0.07] #[0.13,0.14,0.17,0.18,0.19,0.21,0.22,0.23,0.25,0.27,0.3,0.35,0.4,0.45,0.5]
            s_array= s_array1 + s_array2
            # s_array = [1]###############
            
            for s in s_array: 
                newOUTPATH = os.path.join(OUTPATH,"s{}".format(s))
                if not os.path.exists(newOUTPATH):
                    os.mkdir(newOUTPATH)
                command = "HYDRA_FULL_ERROR=2 CUDA_VISIBLE_DEVICES=3 python main.py model=diffwave task=unconditional output_dir=results/guided audio_dir=1 guid_s={} y_noisy={} outpath={} s_schedule={} noise_type={} noise_model_path={}".format(s,wav, newOUTPATH, s_schedule, "loss_model",noise_model_path) #CUDA_VISIBLE_DEVICES=0
                print(command)
                os.system(command)
    command = "python run_measure_new.py -exp_dir {}".format(root)
    os.system(command)
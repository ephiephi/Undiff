from glob import glob
import os
from tqdm import tqdm
from pathlib import Path

s_schedule="60"
outdirname="enhanced_60"

mainroot = Path("/data/ephraim/datasets/known_noise/undiff/exp_ar_j_real_2sec_divided")
roots = [ mainroot/"b"] 
for root in roots:
    print("root: ", root)
    outbasedir = os.path.join(root, outdirname)
    noisy_dir = os.path.join(root, "noisy_wav")
    noiswavs = glob(noisy_dir+"/*")
    print(noiswavs)
    clean_dir = os.path.join(root, "clean_wav")
    

    
    for wav in tqdm(noiswavs):
        snr = wav.split("snr")[1].split("_power")[0]
        noise_type = str(Path(wav).name).split("noise")[1].split("_digits")[0]
        noise_model_path = Path(root) / f"0_snr{snr}_{noise_type}_models.pickle"
        print(noise_model_path)
        if noise_type != "Babble":
            continue
        
        # for j in [0]:
        for l_low in [1.0,0.9,0.8,0.5]:
            print("s_sch: ", s_schedule)
            cur_outbasedir = outbasedir+"_"+str(l_low)
            if not os.path.exists(cur_outbasedir):
                os.mkdir(cur_outbasedir)
            OUTPATH = os.path.join(cur_outbasedir,"snr{}".format(snr))
            if not os.path.exists(OUTPATH):
                os.mkdir(OUTPATH)

            # s_array1 =[0.0001,0.001,0.005,0.05,0.01]#,0.00001,0.08,0.1
            # s_array2=[0.2,0.3,0.5,0.7,0.8]
            s_array1 =[0.08,0.1,0.11, 0.12,0.15]
            # s_array2=[0.2,0.3,0.5,0.7,0.8]
            s_array2=[]
            s_array= s_array1 + s_array2
            # s_array = [1]###############
            
            for s in s_array: 
                newOUTPATH = os.path.join(OUTPATH,"s{}".format(s))
                if not os.path.exists(newOUTPATH):
                    os.mkdir(newOUTPATH)
                command = "HYDRA_FULL_ERROR=2 CUDA_VISIBLE_DEVICES=3 python main.py model=diffwave task=unconditional output_dir=results/guided audio_dir=1 guid_s={} y_noisy={} outpath={} s_schedule={} noise_type={} noise_model_path={} l_low={}".format(s,wav, newOUTPATH, s_schedule, "loss_model_double",noise_model_path, l_low) #CUDA_VISIBLE_DEVICES=0
                print(command)
                os.system(command)
    # command = "python run_measure.py -exp_dir {} -clean_dir {} -noisy_dir {}".format(outbasedir, clean_dir, noisy_dir)
    # os.system(command)
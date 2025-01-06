from glob import glob
import os
from tqdm import tqdm

s_schedule="50"
outdirname="enhanced"
noise_model_path = "/data/ephraim/datasets/known_noise/undiff/exp_ar_g/b/snr5_models.pickle"
roots = ["/data/ephraim/datasets/known_noise/undiff/exp_ar_g/b"]
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


            s_array1 =[]#[0.0801,0.101,0.1201,0.1401, 0.1601]
            s_array2=[0.111]#[0.1251,0.1051]
            s_array= s_array1 + s_array2
            # s_array = [1]###############
            
            for s in s_array: 
                newOUTPATH = os.path.join(OUTPATH,"s{}".format(s))
                if not os.path.exists(newOUTPATH):
                    os.mkdir(newOUTPATH)
                command = "HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=2 python main.py model=diffwave task=unconditional output_dir=results/guided audio_dir=1 guid_s={} y_noisy={} outpath={} s_schedule={} noise_type={} noise_model_path={}".format(s,wav, newOUTPATH, s_schedule, "loss_model",noise_model_path)
                print(command)
                os.system(command)
                
    command = "python run_measure.py -exp_dir {} -clean_dir {} -noisy_dir {}".format(outbasedir, clean_dir, noisy_dir)
    os.system(command)
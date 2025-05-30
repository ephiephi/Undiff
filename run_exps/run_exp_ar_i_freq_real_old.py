from glob import glob
import os
from tqdm import tqdm
from pathlib import Path

s_schedule="60"
outdirname="enhanced_60"

mainroot = Path("/data/ephraim/datasets/known_noise/undiff/exp_ar_i_real_freq")
roots = [ mainroot/"b", mainroot/"c", mainroot/"d", mainroot/"e", mainroot/"f",mainroot/"g",mainroot/"h",mainroot/"i",mainroot/"j"] #,h,ij
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

            s_array1 =[0.01,0.05,0.15]
            s_array2=[0.04,0.06,0.07,0.08,0.09, 0.1]
            s_array= s_array1 + s_array2
            # s_array = [0.1]
            
            for s in s_array: 
                newOUTPATH = os.path.join(OUTPATH,"s{}".format(s))
                if not os.path.exists(newOUTPATH):
                    os.mkdir(newOUTPATH)
                command = "HYDRA_FULL_ERROR=2 CUDA_VISIBLE_DEVICES=0 python main.py model=diffwave task=unconditional output_dir=results/guided audio_dir=1 guid_s={} y_noisy={} outpath={} s_schedule={} noise_type={} noise_model_path={}".format(s,wav, newOUTPATH, s_schedule, "freq_gaussian",noise_model_path) 
                print(command)
                os.system(command)
                print("\n s: ", s)
                print("\n noise_type: ", noise_type)
                print("\n snr: ", snr)
                os.system(command)
    # command = "python run_measure.py -exp_dir {} -clean_dir {} -noisy_dir {}".format(outbasedir, clean_dir, noisy_dir)
    # os.system(command)
    command = "python measure_new.py -exp_dir{}".format(root)
    os.system(command)
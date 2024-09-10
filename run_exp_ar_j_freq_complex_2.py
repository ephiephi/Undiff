from glob import glob
import os
from tqdm import tqdm
from pathlib import Path

s_schedule="60"
outdirname="enhanced_60"

mainroot = Path("/data/ephraim/datasets/known_noise/undiff/exp_ar_j_real_freq_complex/")
roots = [mainroot/"b"]#,mainroot/"c",mainroot/"a"
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
        ####################
        if snr != "5":
            continue
        noise_type = str(Path(wav).name).split("noise")[1].split("_digits")[0]
        ####################
        if noise_type != "Babble":
            continue
        #############
        noise_model_path = Path(root) / f"0_snr{snr}_{noise_type}_models.pickle"
        print(noise_model_path)
        
        for j in [0]:
            print("s_sch: ", s_schedule)
            OUTPATH = os.path.join(outbasedir,"snr{}".format(snr))
            if not os.path.exists(OUTPATH):
                os.mkdir(OUTPATH)

            s_array1 =[0.00005,0.0005,0.0001,0.001,0.01,0.02,0.03,0.06,0.07]
            # s_array1 = [0.00001]
            # s_array2=[0.08,0.09, 0.1,0.12,0.15,0.2,0.3,0.5,0.8,1.0]
            s_array2 = []
            s_array= s_array1 + s_array2
            # s_array = [0.1]
            
            for s in s_array: 
                newOUTPATH = os.path.join(OUTPATH,"s{}".format(s))
                if not os.path.exists(newOUTPATH):
                    os.mkdir(newOUTPATH)
                command = "HYDRA_FULL_ERROR=2 CUDA_VISIBLE_DEVICES=2 python main.py model=diffwave task=unconditional output_dir=results/guided audio_dir=1 guid_s={} y_noisy={} outpath={} s_schedule={} noise_type={} noise_model_path={}".format(s,wav, newOUTPATH, s_schedule, "freq_gaussian_complex",noise_model_path) 

                print(command) #
                print("dir: ", str(root)[-1])
                print("\n s: ", s)
                print("\n noise_type: ", noise_type)
                print("\n s: ", s)
                print("\n snr: ", snr)
                os.system(command)
                # break
    # command = "python run_measure.py -exp_dir {} -clean_dir {} -noisy_dir {}".format(outbasedir, clean_dir, noisy_dir)
    # os.system(command)
    command = "python measure_new.py -exp_dir {}".format(root)
    os.system(command)
from glob import glob
import os

s_schedule="exponential3"
outdirname="enhanced"
roots = ["/data/ephraim/datasets/known_noise/undiff/exp5_7/6/"]
for root in roots:
    outbasedir = os.path.join(root, outdirname)
    noisy_dir = os.path.join(root, "noisy_wav")
    noiswavs = glob(noisy_dir+"/*")
    print(noiswavs)
    clean_dir = os.path.join(root, "clean_wav")


    if not os.path.exists(outbasedir):
        os.mkdir(outbasedir)
    for wav in noiswavs:
        snr = wav.split("snr")[1].split("_power")[0]
        
        OUTPATH = os.path.join(outbasedir,"snr{}".format(snr))
        if not os.path.exists(OUTPATH):
            os.mkdir(OUTPATH)

        s_array1 = [0.01, 0.001,0.005, 0.0001,0.0005, 0.00001]
        s_array2=[ 0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0 ]
        s_array= s_array1 + s_array2
        # s_array = [1]###############
        
        for s in s_array: 
            newOUTPATH = os.path.join(OUTPATH,"s{}".format(s))
            if not os.path.exists(newOUTPATH):
                os.mkdir(newOUTPATH)
            command = "HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=3 python main.py model=diffwave task=unconditional output_dir=results/guided audio_dir=1 guid_s={} y_noisy={} outpath={} s_schedule={}".format(s,wav, newOUTPATH, s_schedule)
            print(command)
            os.system(command)
            
    command = "python run_measure.py -exp_dir {}  -clean_dir {} -noisy_dir {}".format(outbasedir, clean_dir, noisy_dir)
    os.system(command)
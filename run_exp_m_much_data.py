from glob import glob
import os
from tqdm import tqdm
from pathlib import Path
import shutil
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor


def run_command(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Command failed: {cmd}")
    else:
        print(f"Completed: {cmd}")
        

def run_exp(exp_dir, dirnames, s_array=None,reset=False,s_schedule="60",scheduler_type="linear",noise_mosel_path="0", network=None,mog=0,loss_model="loss_model",outdirname="enhanced_60"):
    #attention: always 60
    diffusion = None
    if scheduler_type == "cosine":
        diffusion = "gaussian_diffusion_cosine"
    elif scheduler_type == "linear":
        diffusion = "gaussian_diffusion"
    else:
        print("Unknown scheduler type")
        raise Exception

    mainroot = Path(exp_dir)
    # roots = [ mainroot/"j",mainroot/"b",mainroot/"c"] #,h,ij
    roots = [ mainroot/dname for dname in dirnames]
    
    commands = []
    commands_measuere = []
    
    gpu_idx = -1
    for root in roots:
        print("root: ", root)
        outbasedir = os.path.join(root, outdirname)
        noisy_dir = os.path.join(root, "noisy_wav")
        noiswavs = glob(noisy_dir+"/*")
        print(noiswavs)
        clean_dir = os.path.join(root, "clean_wav")


        if not os.path.exists(outbasedir):
            os.mkdir(outbasedir)
        else:
            if reset:
                dir_path = Path(outbasedir)
                shutil.rmtree(dir_path)
                os.mkdir(outbasedir)
        for wav in tqdm(noiswavs):
            # snr = wav.split("snr")[1].split("_")[0]
            snr = wav.rsplit("snr", 1)[1].split("_")[0]
            noise_type = str(Path(wav).name).split("noise")[1].split("_")[0]
            if noise_mosel_path == "0":
                noise_model_path = Path(root) / f"0_snr{snr}_{noise_type}_models.pickle"
            else:
                noise_model_path = noise_mosel_path
            print(noise_model_path)
            
            for j in [0]:
                
                # cuda_idx = f"cuda:{gpu_idx}"
                print("s_sch: ", s_schedule)
                OUTPATH = os.path.join(outbasedir,"snr{}".format(snr))
                if not os.path.exists(OUTPATH):
                    os.mkdir(OUTPATH)

                if s_array==None:
                    s_array= [0.08,0.09,0.1,0.11, 0.12,0.15,0.05,0.2,0.3,0.35]
                # s_array = [1]###############
                
                for s in s_array: 
                    gpu_idx  = (gpu_idx+1)%4
                    # gpu_idx  = 2
                    print(gpu_idx)
                    newOUTPATH = os.path.join(OUTPATH,"s{}".format(s))
                    if not os.path.exists(newOUTPATH):
                        os.mkdir(newOUTPATH)
                    outwavpath = os.path.join(newOUTPATH, os.path.basename(wav))
                    # if not os.path.exists(outwavpath):
                    if True:
                        run_network="null"
                        if network is not None:
                            run_network = network
                        
                        if run_network.endswith("MoG") or mog>0:
                            if mog<1:
                                print("attention: mog is not set")
                                raise Exception
                            loss_model = "loss_model_mog"
                        
                        command = f"HYDRA_FULL_ERROR=2 CUDA_VISIBLE_DEVICES={gpu_idx} python main.py diffusion={diffusion} model=diffwave task=unconditional output_dir=results/guided audio_dir=1 guid_s={s} y_noisy={wav} outpath={newOUTPATH} s_schedule={s_schedule} noise_type={loss_model} noise_model_path={noise_model_path} network={run_network} mog={mog}"#
                        commands.append(command)
                        # os.system(command)
                    else:
                        print(outwavpath, " already exist")
        command = "python run_measure_new.py -exp_dir {}".format(root)
        commands_measuere.append(command)
        # os.system(command)
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            executor.map(run_command, commands)
        with ThreadPoolExecutor(max_workers=4) as executor:
            executor.map(run_command, commands_measuere)
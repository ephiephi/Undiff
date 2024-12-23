from glob import glob
from pathlib import Path
import shutil
import os
import subprocess
from run_storm_measure import measure_storm


def run_storm(exp_root,storm_root):
    
    noisy_dir = Path(storm_root)/"noisy_wav" 
    clean_dir =  Path(storm_root)/"clean_wav"
    root = Path(storm_root)/"enhanced/"
    for p in [storm_root, noisy_dir,clean_dir,root]:
        if not os.path.exists(str(p)):
            os.mkdir(p)

    dirs = glob(exp_root+"/*")
    for d in dirs:
        if os.path.basename(d) in ["storm","analysis","cleans","noises","5f_snrs.pickle"]:
            continue
        if Path(d).name == "scaling_snrs.pickle" or Path(d).name == "_scaling_snrs.pickle" or Path(d).name == "5f_snrs.pickle":
            continue
        path_ = Path(d) / "clean_wav" 
        noisy_wavs = glob(str(path_)+"/*")
        for wav in noisy_wavs:
            src = wav
            dst_clean = Path(storm_root)/"clean_wav/"/(str(f"{Path(d).name}_")+str(Path(wav).name))
            print(src)
            print(dst_clean)

            shutil.copyfile(src, dst_clean)
            print("copy clean")
            
            path_ = Path(d) / "noisy_wav" 
        noisy_wavs = glob(str(path_)+"/*")
        for wav in noisy_wavs:
            src = wav
            dst_noisy = Path(storm_root)/"noisy_wav/"/(str(f"{Path(d).name}_")+str(Path(wav).name))
            print(src)
            print(dst_noisy)

            shutil.copyfile(src, dst_noisy)
            print("copy noisy")

    
    command = f"bash /data/ephraim/Undiff/run_storm.sh --test_dir {str(noisy_dir)} --enhanced_base_dir {storm_root}/enhanced/"
    print(command)
    result = subprocess.run(command, shell=True, check=True, text=True)
    
    print("starting measures")
    measure_storm(storm_root=storm_root)



if __name__ == "__main__":
    exp_root = "/data/ephraim/datasets/known_noise/undiff_exps/exp_m_long_ar"
    storm_root = "/data/ephraim/datasets/known_noise/undiff_exps/exp_m_long_ar/storm/"
    run_storm(exp_root,storm_root)
import os
from glob import glob
from tqdm import tqdm
import argparse
from pathlib import Path
from measure_new import main_measure

# exp_dir = "/data/ephraim/datasets/known_noise/enhanced_diffwave_1sec/"
def main(root_dir,clean_dir, noisy_dir ):
    for d in glob(root_dir+"*/"):
        if Path(d).name not in ["noises", "clean_wav", "noisy_wav", "analysis", "storm", "cleans"]:
            # command = "python measure_new.py -exp_dir={}".format(d)
            # print(command)
            # os.system(command)
            main_measure(d)
            
        

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="measure guided")
    parser.add_argument(
        "-exp_dir",
        default="/data/ephraim/datasets/known_noise/undiff_exps/exp_n_real/",
    )
    parser.add_argument(
        "-clean_dir", default=""
    )
    parser.add_argument(
        "-noisy_dir", default=""
    )


    args = parser.parse_args()
    main(
        args.exp_dir,
        args.clean_dir,
        args.noisy_dir
    )

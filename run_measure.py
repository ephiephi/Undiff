import os
from glob import glob
from tqdm import tqdm
import argparse



# exp_dir = "/data/ephraim/datasets/known_noise/enhanced_diffwave_1sec/"
def main(exp_dir, clean_dir, noisy_dir):
    snr_dirs = glob(os.path.join(exp_dir,"*/"))
    for snr_dir in tqdm(snr_dirs): 
        # snr_dir = "/data/ephraim/datasets/known_noise/enhanced_diffwave_1sec/snr10/"
        stats_path =os.path.join(snr_dir, "stats_dns.json")
        if os.path.exists(stats_path):
            os.remove(stats_path)
        s_dirs =glob(snr_dir+"*/")
        for d in tqdm(s_dirs):
            command = "python measure.py -enhanced_dir={} -out_stats_dir={} -clean_dir {} -noisy_dir {}".format(d,snr_dir, clean_dir, noisy_dir)
            os.system(command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="measure guided")
    parser.add_argument(
        "-exp_dir",
        default="/data/ephraim/datasets/known_noise/enhanced_diffwave_1sec/",
    )
    parser.add_argument(
        "-clean_dir", default="/data/ephraim/datasets/known_noise/clean_wav/"
    )
    parser.add_argument(
        "-noisy_dir", default="/data/ephraim/datasets/known_noise/noisy_wav/"
    )


    args = parser.parse_args()
    main(
        args.exp_dir,
        args.clean_dir,
        args.noisy_dir
    )

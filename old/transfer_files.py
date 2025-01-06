from glob import glob
from pathlib import Path
import shutil

dirs = glob("/data/ephraim/datasets/known_noise/undiff/exp_ar_i_095/*")
for d in dirs:
    print()

    if Path(d).name == "scaling_snrs.pickle" or Path(d).name == "_scaling_snrs.pickle":
        continue
    path_ = Path(d) / "clean_wav" 
    noisy_wavs = glob(str(path_)+"/*")
    for wav in noisy_wavs:
        src = wav
        dst = Path("/data/ephraim/datasets/known_noise/sgmse/exp_i/clean_synth_i95/")/(str(f"{Path(d).name}_")+str(Path(wav).name))
        print(src)
        print(dst)

        shutil.copyfile(src, dst)
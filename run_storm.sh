

#!/bin/bash

#usage: bash /data/ephraim/Undiff/run_storm.sh --test_dir "/data/ephraim/datasets/known_noise/undiff_exps/exp_m_long_ar/storm/noisy_wav" --enhanced_base_dir "/data/ephraim/datasets/known_noise/undiff_exps/exp_m_long_ar/storm/enhanced/"



# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --test_dir)
      TEST_DIR="$2"
      shift
      shift
      ;;
    --enhanced_base_dir)
      ENHANCED_BASE_DIR="$2"
      shift
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done


# Ensure required arguments are provided
if [ -z "$TEST_DIR" ] || [ -z "$ENHANCED_BASE_DIR" ]; then
  echo "Usage: $0 --test_dir <path> --enhanced_base_dir <path>"
  exit 1
fi


# Define other fixed parameters
CKPT_PATH_1="/data/ephraim/repos/checkpoints/sgmse_pretrained/WSJ0+Chime3/epoch=451-pesq=0.00.ckpt"
CKPT_PATH_2="/data/ephraim/repos/checkpoints/storm_pretrained/VoiceBank-DEMAND/epoch=203-pesq=0.00.ckpt"
CKPT_PATH_3="/data/ephraim/repos/checkpoints/sgmse_pretrained/TIMIT+Chime3/epoch=907-pesq=0.00.ckpt"
CKPT_PATH_4="/data/ephraim/repos/checkpoints/storm_pretrained/TIMIT+Chime3/epoch=761-pesq=0.00.ckpt"

# Enhanced directories dynamically constructed
ENHANCED_DIR_SGMSE_WSJ0="${ENHANCED_BASE_DIR}/sgmse_WSJ0Chime3"
ENHANCED_DIR_STORM_VBD="${ENHANCED_BASE_DIR}/storm_vbd"
ENHANCED_DIR_SGMSE_TIMIT="${ENHANCED_BASE_DIR}/sgmse_TIMITChime3"
ENHANCED_DIR_STORM_TIMIT="${ENHANCED_BASE_DIR}/storm_TIMITChime3"

# Run the commands using the dynamically passed enhanced_base_dir
echo "RUNNING ENHANCEMENT SCRIPTS..."
CUDA_VISIBLE_DEVICES=0 python /data/ephraim/repos/storm/enhancement.py \
  --test_dir "$TEST_DIR" \
  --enhanced_dir "$ENHANCED_DIR_SGMSE_WSJ0" \
  --ckpt "$CKPT_PATH_1" \
  --mode "score-only" --N 200 &

CUDA_VISIBLE_DEVICES=1 python /data/ephraim/repos/storm/enhancement.py \
  --test_dir "$TEST_DIR" \
  --enhanced_dir "$ENHANCED_DIR_STORM_VBD" \
  --ckpt "$CKPT_PATH_2" \
  --mode "storm" --N 200 &

CUDA_VISIBLE_DEVICES=2 python /data/ephraim/repos/storm/enhancement.py \
  --test_dir "$TEST_DIR" \
  --enhanced_dir "$ENHANCED_DIR_SGMSE_TIMIT" \
  --ckpt "$CKPT_PATH_3" \
  --mode "score-only" --N 200 &

CUDA_VISIBLE_DEVICES=3 python /data/ephraim/repos/storm/enhancement.py \
  --test_dir "$TEST_DIR" \
  --enhanced_dir "$ENHANCED_DIR_STORM_TIMIT" \
  --ckpt "$CKPT_PATH_4" \
  --mode "storm" --N 200 



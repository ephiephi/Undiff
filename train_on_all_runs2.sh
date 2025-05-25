# CUDA_VISIBLE_DEVICES=2 python train_on_all_noises.py --data_dir /data/ephraim/datasets/DNS-Challenge_old/datasets/noise \
#     --batch_size 1 \
#     --epochs 5 \
#     --network NetworkNoiseWaveNetMoG \
#     --lr 1e-3 \
#     --save_model_path /data/ephraim/datasets/known_noise/undiff_exps/training_all/pure_noises_netwavenet_mog/model.pth \
#     --test_files /data/ephraim/datasets/known_noise/undiff_exps/training_all/pure_noises_netwavenet_mog/testset.txt \
#     --save_test_list None \
#     --mog True


CUDA_VISIBLE_DEVICES=1 python train_on_all_noises.py --data_dir /data/ephraim/datasets/DNS-Challenge_old/datasets/noise \
    --batch_size 1 \
    --epochs 2 \
    --network WaveNetCausalModel \
    --lr 1e-3 \
    --save_model_path /data/ephraim/datasets/known_noise/undiff_exps/training_all/pure_noises_netwavenet/model.pth \
    --save_test_list /data/ephraim/datasets/known_noise/undiff_exps/training_all/pure_noises_netwavenet/testset.txt \
    --test_files /data/ephraim/datasets/known_noise/undiff_exps/training_all/pure_noises_netwavenet/testset.txt \
    --no-mog

# CUDA_VISIBLE_DEVICES=" " python train_on_all_noises.py --data_dir /data/ephraim/datasets/DNS-Challenge_old/datasets/noise \
#     --batch_size 1 \
#     --epochs 5 \
#     --network NetworkNoise6 \
#     --lr 1e-3 \
#     --save_model_path /data/ephraim/datasets/known_noise/undiff_exps/training_all/pure_noises_net6/model.pth \
#     --save_test_list /data/ephraim/datasets/known_noise/undiff_exps/training_all/pure_noises_net6/testset.txt \


CUDA_VISIBLE_DEVICES=0 python train_on_all_noises.py --data_dir /data/ephraim/datasets/DNS-Challenge_old/datasets/noise \
    --batch_size 1 \
    --epochs 5 \
    --network NetworkNoise4 \
    --lr 1e-3 \
    --save_model_path /data/ephraim/datasets/known_noise/undiff_exps/training_all/pure_noises_net4/model.pth \
    --save_test_list /data/ephraim/datasets/known_noise/undiff_exps/training_all/pure_noises_net4/testset.txt \

CUDA_VISIBLE_DEVICES=0 python train_on_all_noises.py --data_dir /data/ephraim/datasets/DNS-Challenge_old/datasets/noise \
    --batch_size 1 \
    --epochs 4 \
    --network NetworkNoise6MoG \
    --lr 1e-3 \
    --save_model_path /data/ephraim/datasets/known_noise/undiff_exps/training_all/pure_noises_net6_mog/model.pth \
    --save_test_list /data/ephraim/datasets/known_noise/undiff_exps/training_all/pure_noises_net6_mog/testset.txt \
    --test_files /data/ephraim/datasets/known_noise/undiff_exps/training_all/pure_noises_netwavenet_mog/testset.txt \
    --mog \
    --num_mixtures 50

CUDA_VISIBLE_DEVICES=0 python train_on_all_noises.py --data_dir /data/ephraim/datasets/DNS-Challenge_old/datasets/noise \
    --batch_size 1 \
    --epochs 4 \
    --network NetworkNoiseWaveNetMoG \
    --lr 1e-3 \
    --save_model_path /data/ephraim/datasets/known_noise/undiff_exps/training_all/pure_noises_netwavenet_mog/model.pth \
    --test_files /data/ephraim/datasets/known_noise/undiff_exps/training_all/pure_noises_netwavenet_mog/testset.txt \
    --save_test_list None \
    --mog


CUDA_VISIBLE_DEVICES=1 python train_on_all_noises.py --data_dir /data/ephraim/datasets/DNS-Challenge_old/datasets/noise \
    --batch_size 1 \
    --epochs 5 \
    --network NetworkNoiseWaveNetMoG2 \
    --lr 1e-3 \
    --save_model_path /data/ephraim/datasets/known_noise/undiff_exps/training_all/pure_noises_netwavenet_mog2/model.pth \
    --test_files /data/ephraim/datasets/known_noise/undiff_exps/training_all/pure_noises_netwavenet_mog/testset.txt \
    --save_test_list None \
    --mog

CUDA_VISIBLE_DEVICES=0 python create_exp_m.py -config exps_configs/m_ar_netwavenet.yaml
CUDA_VISIBLE_DEVICES=0 python create_exp_m.py -config exps_configs/m_ar_net8.yaml
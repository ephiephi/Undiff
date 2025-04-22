python train_on_all_noises.py --data_dir /data/ephraim/datasets/DNS-Challenge_old/datasets/noise \
    --batch_size 1 \
    --epochs 5 \
    --network NetworkNoise6 \
    --lr 1e-3 \
    --save_model_path /data/ephraim/datasets/known_noise/undiff_exps/training_all/pure_noises_net6/model.pth \
    --save_test_list /data/ephraim/datasets/known_noise/undiff_exps/training_all/pure_noises_net6/testset.txt \


python train_on_all_noises.py --data_dir /data/ephraim/datasets/DNS-Challenge_old/datasets/noise \
    --batch_size 1 \
    --epochs 5 \
    --network NetworkNoise8 \
    --lr 1e-3 \
    --save_model_path /data/ephraim/datasets/known_noise/undiff_exps/training_all/pure_noises_net8/model.pth \
    --save_test_list /data/ephraim/datasets/known_noise/undiff_exps/training_all/pure_noises_net8/testset.txt \
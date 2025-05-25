#!/bin/bash


factors=(4 6 7)


for i in "${factors[@]}"; do
    echo "python create_exp_n_real.py -config exps_configs/o_net3_6_amplified_${i}.yaml"
    python create_exp_n_real.py -config "exps_configs/o_net3_6_amplified_${i}.yaml"
done
#!/bin/bash

factors=(3 8 15 20 40 80 120 150 300 500 800)

for i in "${factors[@]}"; do
    echo "python create_exp_n_real.py -config exps_configs/o_net3_6_amplified_${i}.yaml"
    python create_exp_n_real.py -config "exps_configs/o_net3_6_amplified_${i}.yaml"
done
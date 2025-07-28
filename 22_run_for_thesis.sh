#!/bin/bash

# python create_exp_n_real.py -config exps_configs/librDemandSttnr_net40_fast_snrs.yaml
# python create_exp_n_real.py -config exps_configs/librDemandSttnr_net30_fast_snrs.yaml
# CUDA_VISIBLE_DEVICES=2,3 python create_exp_p.py -config exps_configs/librDemandSttnr_p_net3_snrm5.yaml #bug - run measures

# python create_exp_n_real.py -config exps_configs/librDemandSttnr_net3_6S_fast.yaml 

# python create_exp_n_real.py -config exps_configs/librMUSAN_net3_6_snr5.yaml
# python create_exp_n_real.py -config exps_configs/librDemandSttnr_net30_snr5.yaml
# python create_exp_n_real.py -config exps_configs/librDemandSttnr_net40_snr5.yaml

# python create_exp_p.py -config exps_configs/librDemandSttnr_p_net3_snr5.yaml
# CUDA_VISIBLE_DEVICES=1,2,3 python create_exp_n_real.py -config exps_configs/librDemandSttnr_net3_6_snr5.yaml
CUDA_VISIBLE_DEVICES=2,3 python create_exp_n_real.py -config exps_configs/librDemandSttnr_net40_snr5.yaml
python create_exp_p.py -config exps_configs/librDemandSttnr_p_net3_6_snr5.yaml

python create_exp_n_real.py -config exps_configs/librAE_net3_6_snr5.yaml 


python create_exp_p_using_y.py -config exps_configs/librDemandSttnr_pusingy_net3_6_snr5.yaml

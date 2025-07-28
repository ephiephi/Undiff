#!/bin/bash




python create_exp_n_real.py -config exps_configs/librBBC20_net3_6_snr5.yaml 
python create_exp_n_real.py -config exps_configs/librBBC20_net3_6_snr0.yaml 
python create_exp_n_real.py -config exps_configs/librBBC20_net3_6_snr-5.yaml 
python create_exp_n_real.py -config exps_configs/librBBC20_net3_6_snr10.yaml 

python create_exp_p.py -config exps_configs/librBBC20_p_net3_6_snr5.yaml 
python create_exp_p.py -config exps_configs/librBBC20_p_net2_6_snr5.yaml 


python create_exp_n_real.py -config exps_configs/librBBC20_net30_snr5.yaml 
python create_exp_n_real.py -config exps_configs/librBBC20_net40_snr5.yaml 






python create_exp_p_using_y.py -config exps_configs/librDemandSttnr_pusingy_net3_6_snr5.yaml
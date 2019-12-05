#!/usr/bin/env bash

lps=3001
lpsw=500
for lwf in 2.0
do
for lwc in 0.05
do
for lwr in 0.1
do
wd=1e-4
m=0.9
lr=0.001
bs=32
bn_eval=1
max=3
wua=1
lpsaw=500
tstep=100
itc=0
itf=0
det=True
t=1

for i in `seq 0 $max`
do
python main_epi_fcr.py \
--lr=$lr  \
--num_classes=7 \
--test_every=$tstep \
--logs='epi_fcr/itf='$itf'itc='$itc'wua='$wua'lwr='$lwr'lwc='$lwc'lwf='$lwf'lpsaw='$lpsaw'lpsw='$lpsw'lps='$lps'/run='$t'/logs_'$i'/' \
--batch_size=$bs \
--model_path='epi_fcr/itf='$itf'itc='$itc'wua='$wua'lwr='$lwr'lwc='$lwc'lwf='$lwf'lpsaw='$lpsaw'lpsw='$lpsw'lps='$lps'/run='$t'/models_'$i'/' \
--unseen_index=$i \
--loops_train=$lps \
--loops_warm=$lpsw \
--step_size=$lps \
--state_dict=$2 \
--data_root=$1 \
--loss_weight_epif=$lwf \
--loss_weight_epic=$lwc \
--loss_weight_epir=$lwr \
--weight_decay=$wd \
--momentum=$m \
--bn_eval=$bn_eval \
--warm_up_agg=$wua \
--loops_agg_warm=$lpsaw \
--ite_train_epi_c=$itc \
--ite_train_epi_f=$itf \
--deterministic=$det
done
done
done
done

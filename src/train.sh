#!/bin/bash

ctt_model_names="dkvmn_c_q_ctt sakt_c_q_ctt"

for ctt_model_name in ${ctt_model_names}
do
    python \
    train.py \
    --model_fn ${ctt_model_name}0722.pth \
    --model_name ${ctt_model_name} \
    --dataset_name dkt_assist2009_pid_diff \
    --batch_size 512 \
    --fivefold True \
    --n_epochs 1000
done

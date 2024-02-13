#!/bin/bash
# dkt_dbe22_pid

python \
train.py \
--model_fn dkt_c_q.pth \
--model_name dkt_c_q \
--dataset_name dkt_dbe22_pid \
--batch_size 512 \
--fivefold True \
--n_epochs 1000

model_names="dkvmn_c_q akt"

for model_name in ${model_names}
do
    python \
    train.py \
    --model_fn ${model_name}.pth \
    --model_name ${model_name} \
    --dataset_name dbe22_pid \
    --batch_size 512 \
    --fivefold True \
    --n_epochs 1000
done

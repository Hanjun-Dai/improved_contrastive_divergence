#!/bin/bash

export CUDA_VISIBLE_DEVICES=4

python train.py \
	--exp=cifar10_model \
	--step_lr=100.0 \
	--num_steps=40 \
	--cuda \
	--ensembles=1 \
	--kl_coeff=1.0 \
	--kl=True \
	--multiscale \
	--self_attn

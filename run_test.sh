#!/bin/bash

export CUDA_VISIBLE_DEVICES=4

python test.py \
	--exp=cifar10_model_bak \
	--step_lr=100.0 \
	--num_steps=40 \
	--cuda \
	--ensembles=1 \
	--kl_coeff=1.0 \
	--kl=True \
	--multiscale \
	--self_attn \
	--resume_iter=16000 \

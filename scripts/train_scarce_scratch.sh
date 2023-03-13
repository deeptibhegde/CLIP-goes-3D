#!/usr/bin/env bash

set -x
GPUS=$1


PY_ARGS=${@:2}



for per_samples in {10,20,30,50}


   

do 

     CUDA_VISIBLE_DEVICES=${GPUS} python main_BERT.py  --per_samples $per_samples  --exp_name scarce_scratch_$per_samples ${PY_ARGS}  



done;
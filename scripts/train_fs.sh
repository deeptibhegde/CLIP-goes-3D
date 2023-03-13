#!/usr/bin/env bash

set -x
GPUS=$1


PY_ARGS=${@:2}



i=6
while [ $i -ne 10 ]


   

do 

     CUDA_VISIBLE_DEVICES=${GPUS} python main_BERT.py  --way 40 --shot 16 --fold $i --clip  --exp_name clasp_fs_16s_$i ${PY_ARGS} 

     i=$(($i+1))

done;
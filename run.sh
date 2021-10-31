#!/bin/bash
#PBS -l nodes=1:gold6258r
#PBS -N chexnet
#PBS -j oe
#PBS -o output.log

PYTHON=python

if [ ${PBS_O_WORKDIR} ]; then cd ${PBS_O_WORKDIR}; fi

source ~/openvino/bin/activate

${PYTHON} infer.py --mode fp32

${PYTHON} infer.py --mode int8

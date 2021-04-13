#!/bin/bash
#PBS -l nodes=1:ppn=2
#PBS -N chexnet_int8
#PBS -j oe
#PBS -o log/output_int8.log

if [ $(which python) ]; then PYTHON=python; fi
if [ $(which python3) ]; then PYTHON=python3; fi

if [ ${PBS_O_WORKDIR} ]; then cd ${PBS_O_WORKDIR}; fi

${PYTHON} ov-inference.py --int8

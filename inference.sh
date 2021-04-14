#!/bin/bash
#PBS -l nodes=1:ppn=2
#PBS -N chexnet_run
#PBS -j oe
#PBS -o log/output_inference.log

if [ $(which python) ]; then PYTHON=python; fi
if [ $(which python3) ]; then PYTHON=python3; fi

if [ ${PBS_O_WORKDIR} ]; then cd ${PBS_O_WORKDIR}; fi

${PYTHON} ov-inference.py --fp32
${PYTHON} ov-inference.py --int8

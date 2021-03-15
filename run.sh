#!/bin/bash
#PBS -l nodes=1:ppn=2
#PBS -N chexnet
#PBS -j oe
#PBS -o log/output.log

if [ ${PBS_O_WORKDIR} ]; then cd ${PBS_O_WORKDIR}; fi

python main.py

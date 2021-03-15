#!/bin/bash
#PBS -l nodes=1:ppn=2
#PBS -N chexnet_fp32
#PBS -j oe
#PBS -o log/output_fp32.log

if [ ${PBS_O_WORKDIR} ]; then cd ${PBS_O_WORKDIR}; fi

# l_openvino_toolkit_p_2019.3.376.tgz
#source ${PBS_O_HOME}/intel/openvino/bin/setupvars.sh

python ov-inference.py --fp32
python ov-inf-async.py --fp32

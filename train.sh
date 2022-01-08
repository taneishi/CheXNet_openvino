#!/bin/bash
#PBS -l nodes=1:gold6258r
#PBS -N chexnet
#PBS -j oe
#PBS -o output.log

if [ ${PBS_O_WORKDIR} ]; then
    cd ${PBS_O_WORKDIR}
fi

if [ ! -d openvino ]; then
    python3 -m venv openvino
    source openvino/bin/activate
    pip install --upgrade pip
    pip install openvino_dev torchvision onnx
else
    source openvino/bin/activate
fi

CPUS=$(grep "physical id" /proc/cpuinfo | sort -u | wc -l)
CORES=$(grep "core id" /proc/cpuinfo | sort -u | wc -l)
TOTAL_CORES=$((${CPUS}*${CORES}))

echo "CPUS=${CPUS} CORES=${CORES} TOTAL_CORES=${TOTAL_CORES}"
export OMP_NUM_THREADS=${TOTAL_CORES}
export KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"

python train.py

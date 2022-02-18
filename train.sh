#!/bin/bash
#PBS -l nodes=1:gold6338n
#PBS -N chexnet
#PBS -j oe
#PBS -o output.log

if [ ${PBS_O_WORKDIR} ]; then
    cd ${PBS_O_WORKDIR}
fi

if [ -d openvino ]; then
    source openvino/bin/activate
else
    python3 -m venv openvino
    source openvino/bin/activate
    pip install --upgrade pip
    pip install openvino_dev torchvision onnx
fi

CPUS=2
CORES=32
TOTAL_CORES=$((${CPUS}*${CORES}))

echo "CPUS=${CPUS} CORES=${CORES} TOTAL_CORES=${TOTAL_CORES}"
export OMP_NUM_THREADS=${TOTAL_CORES}
export KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"

python train.py

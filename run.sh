#!/bin/bash
#PBS -l nodes=1:gold6330
#PBS -N chexnet
#PBS -j oe
#PBS -o output.log

if [ ${PBS_O_WORKDIR} ]; then
    cd ${PBS_O_WORKDIR}
fi

CPUS=2
CORES=24
TOTAL_CORES=$((${CPUS}*${CORES}))

echo "CPUS=${CPUS} CORES=${CORES} TOTAL_CORES=${TOTAL_CORES}"
export OMP_NUM_THREADS=${TOTAL_CORES}
export KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"

pip install -r requirements.txt

if [ ! -f model/densenet121.onnx ]; then
    rm -f model/*.xml
    python export_onnx.py --batch_size 100
fi

if [ ! -f model/densenet121.xml ]; then
    # fp32 model optimization
    mo --input_model model/densenet121.onnx --output_dir model
fi

if [ ! -f model/chexnet-pytorch.xml ]; then
    # make annotations
    mkdir -p annotations
    python annotation.py chestxray14 --annotation_file labels/test_list.txt -ss 1000 \
        -o annotations -a chestxray14.pickle -m chestxray14.json --data_dir images

    # int8 quantization
    pot -c config/chexnet_int8.yaml -e
    cp $(ls results/chexnet_DefaultQuantization/*/optimized/* | tail -3) model
fi

python main.py --mode torch --batch_size 10
python main.py --mode fp32 --batch_size 10
python main.py --mode int8 --batch_size 10

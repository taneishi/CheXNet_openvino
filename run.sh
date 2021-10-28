#!/bin/bash
#PBS -l nodes=1:gold6258r
#PBS -N chexnet
#PBS -j oe
#PBS -o output.log

PYTHON=python

if [ ${PBS_O_WORKDIR} ]; then cd ${PBS_O_WORKDIR}; fi

source ~/openvino/bin/activate

${PYTHON} export_onnx.py

# fp32 model optimization
mo --input_model ${PWD}/model/densenet121.onnx --output_dir model

# make annotations
mkdir -p annotations
${PYTHON} annotation.py chest_xray --annotation_file labels/val_list.txt -ss 320 -o annotations -a chestx.pickle -m chestx.json --data_dir images

# int8 quantization
pot -c config/chexnet_int8.yaml -e

${PYTHON} ov-infer.py --int8

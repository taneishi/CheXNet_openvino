#!/bin/bash
#PBS -l nodes=1:gold6258r
#PBS -N chexnet
#PBS -j oe
#PBS -o output.log

PYTHON=python3

if [ ${PBS_O_WORKDIR} ]; then cd ${PBS_O_WORKDIR}; fi

# fp32 model optimization
#${PYTHON} ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/mo.py --input_model ${PWD}/model/densenet121.onnx --output_dir model

# make annotations
#mkdir -p annotations
#${PYTHON} annotation.py chest_xray --annotation_file labels/val_list.txt -ss 320 -o annotations -a chestx.pickle -m chestx.json --data_dir images

# accuracy check
#accuracy_check -c config/chexnet.yaml -m model

# int8 quantization
#pot -c config/chexnet_int8_pot.yaml -e

# accuracy check
#accuracy_check -c config/chexnet_int8.yaml -m model

${PYTHON} ov-inference.py --int8 --batch_size 2048

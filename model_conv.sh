#!/bin/bash
#PBS -l nodes=1:ppn=2
#PBS -N chexnet_conv
#PBS -j oe
#PBS -o log/output_conv.log

if [ $(which python) ]; then PYTHON=python; fi
if [ $(which python3) ]; then PYTHON=python3; fi

if [ ${PBS_O_WORKDIR} ]; then cd ${PBS_O_WORKDIR}; fi

# fp32 model optimization
${PYTHON} ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/mo_onnx.py \
    --input_model ${PWD}/model/densenet121.onnx --data_type=FP32 --batch 1 --output_dir model

mkdir -p annotations

${PYTHON} annotation.py chest_xray --annotation_file labels/val_list.txt -ss 200 \
    -o annotations -a chestx.pickle -m chestx.json --data_dir images

# int8 quantization
${PYTHON} calibrate.py --config config/chestx.yml -d config/def.yml \
    -M ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer \
    --models model --annotations annotations --batch_size 64

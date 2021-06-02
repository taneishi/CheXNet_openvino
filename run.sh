#!/bin/bash
#PBS -l nodes=1:ppn=2
#PBS -N chexnet
#PBS -j oe
#PBS -o log/output.log

if [ $(which python) ]; then PYTHON=python; fi
if [ $(which python3) ]; then PYTHON=python3; fi

#if [ ${PBS_O_WORKDIR} ]; then cd ${PBS_O_WORKDIR}; fi

${PYTHON} ${INTEL_OPENVINO_DIR}/deployment_tools/tools/benchmark_tool/benchmark_app.py -m model/densenet121.onnx -i images

# fp32 model optimization
${PYTHON} ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/mo.py --input_model ${PWD}/model/densenet121.onnx --output_dir model

# make annotations
#mkdir -p annotations
#${PYTHON} annotation.py chest_xray --annotation_file labels/val_list.txt -ss 320 -o annotations -a chestx.pickle -m chestx.json --data_dir images

# accuracy check
accuracy_check -c config/chexnet.yaml -m model

# benchmark
${PYTHON} ${INTEL_OPENVINO_DIR}/deployment_tools/tools/benchmark_tool/benchmark_app.py -m model/densenet121.xml -i images

# int8 quantization
pot -c config/chexnet_int8_pot.yaml -e

# accuracy check
accuracy_check -c config/chexnet_int8.yaml -m model

# benchmark
${PYTHON} ${INTEL_OPENVINO_DIR}/deployment_tools/tools/benchmark_tool/benchmark_app.py -m model/chexnet-pytorch.xml -i images

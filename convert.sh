#!/bin/bash

#python3 -m venv openvino
#source ~/openvino/bin/activate
#pip install --upgrade pip
#pip install openvino_dev torchvision onnx==1.8.1

python export_onnx.py

# fp32 model optimization
mo --input_model ${PWD}/model/densenet121.onnx --output_dir model

# make annotations
mkdir -p annotations
python annotation.py chest_xray --annotation_file labels/val_list.txt -ss 320 -o annotations -a chestx.pickle -m chestx.json --data_dir images

# int8 quantization
pot -c config/chexnet_int8.yaml -e
cp $(ls results/chexnet-pytorch_DefaultQuantization/*/optimized/* | tail -3) model

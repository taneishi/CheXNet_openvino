import torch
import argparse
import os

from model import DenseNet121, N_CLASSES

def main(args):
    device = torch.device('cpu')

    # initialize and load the model
    model = DenseNet121(N_CLASSES)

    if os.path.isfile(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print('model state has loaded')
    else:
        print('=> model state file not found')

    model.train(False)

    dummy_input = torch.randn(args.batch_size, 3, 224, 224)
    torch_out = model(dummy_input)

    torch.onnx.export(model,
            dummy_input,
            'model/densenet121.onnx',
            verbose=False)
    print('ONNX model exported.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='model/model.pth', type=str)
    parser.add_argument('--batch_size', default=10, type=int)
    args = parser.parse_args()

    main(args)

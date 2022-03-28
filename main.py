import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torchvision.transforms as transforms
from openvino.inference_engine import IECore
import argparse
import timeit

from datasets import ChestXrayDataSet
from model import DenseNet121, CLASS_NAMES, N_CLASSES

def main(args):
    if args.mode == 'torch':
        net = DenseNet121(N_CLASSES)
        net.load_state_dict(torch.load('model/model.pth', map_location=torch.device('cpu')))
        print('model state has loaded')

        if args.num_threads:
            torch.set_num_threads(args.num_threads)
        print('number of threads %d' % (torch.get_num_threads()))

    elif args.mode == 'fp32' or args.mode == 'int8':
        if args.mode == 'fp32':
            modelfile = 'densenet121.xml'
        elif args.mode == 'int8':
            modelfile = 'chexnet.xml'
        
        model_xml = 'model/%s' % (modelfile)
        model_bin = model_xml.replace('.xml', '.bin')

        print('Creating Inference Engine')
        ie = IECore()
        net = ie.read_network(model=model_xml, weights=model_bin)

        # loading model to the plugin
        print('Loading model to the plugin')
        exec_net = ie.load_network(network=net, device_name='CPU')

        print('Preparing input blobs')
        input_blob = next(iter(net.input_info))
        output_blob = next(iter(net.outputs))

        model_batch_size, c, h, w = net.input_info[input_blob].input_data.shape

    # for image load
    normalize = transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])

    test_dataset = ChestXrayDataSet(
            data_dir=args.data_dir,
            image_list_file=args.test_image_list,
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.TenCrop(224),
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
                ]))
    
    test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=False,
            drop_last=True)

    y_true = torch.FloatTensor()
    y_pred = torch.FloatTensor()
    
    for index, (data, labels) in enumerate(test_loader):
        start_time = timeit.default_timer()

        batch_size, n_crops, c, h, w = data.size()
        data = data.view(-1, c, h, w)

        if args.mode == 'torch':
            with torch.no_grad():
                outputs = net(data)
            outputs = outputs.view(batch_size, n_crops, -1).mean(1)
            outputs = outputs.numpy()

        elif args.mode == 'fp32' or args.mode == 'int8':
            images = np.zeros(shape=(model_batch_size, c, h, w))
            images[:n_crops * args.batch_size, :c, :h, :w] = data.numpy()

            outputs = exec_net.infer(inputs={input_blob: images})
            outputs = outputs[output_blob]

            outputs = outputs[:n_crops * args.batch_size].reshape(args.batch_size, n_crops, -1)
            outputs = np.mean(outputs, axis=1)
            outputs = outputs[:args.batch_size, :outputs.shape[1]]

        y_true = torch.cat((y_true, labels), 0)
        y_pred = torch.cat((y_pred, torch.from_numpy(outputs)), 0)
        
        print('\r%4d/%4d, time: %6.3fsec' % (index, len(test_loader), (timeit.default_timer() - start_time)), end='')

        aucs = [roc_auc_score(y_true[:, i], y_pred[:, i]) if y_true[:, i].sum() > 0 else np.nan for i in range(N_CLASSES)]
        auc_classes = ' '.join(['%5.3f' % (aucs[i]) for i in range(N_CLASSES)])
        print(' average AUC %5.3f (%s)' % (np.mean(aucs), auc_classes))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['torch', 'fp32', 'int8'], default='torch', type=str)
    parser.add_argument('--num_threads', default=None, type=int)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--data_dir', default='images', type=str)
    parser.add_argument('--test_image_list', default='labels/test_list.txt', type=str)
    args = parser.parse_args()
    print(vars(args))

    main(args)

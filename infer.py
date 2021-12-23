import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torchvision.transforms as transforms
from openvino.inference_engine import IECore
import argparse
import timeit

from datasets import ChestXrayDataSet
from model import CLASS_NAMES, N_CLASSES

def main(modelfile):
    model_xml = 'model/%s' % modelfile
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

    gt = torch.FloatTensor()
    pred = torch.FloatTensor()
    
    start = timeit.default_timer()

    for index, (data, target) in enumerate(test_loader):
        start_time = timeit.default_timer()

        batch_size, n_crops, c, h, w = data.size()
        data = data.view(-1, c, h, w).numpy()

        images = np.zeros(shape=(model_batch_size, c, h, w))
        images[:n_crops * args.batch_size, :c, :h, :w] = data

        outputs = exec_net.infer(inputs={input_blob: images})
        outputs = outputs[output_blob]

        outputs = outputs[:n_crops * args.batch_size].reshape(args.batch_size, n_crops, -1)
        outputs = np.mean(outputs, axis=1)
        outputs = outputs[:args.batch_size, :outputs.shape[1]]

        gt = torch.cat((gt, target), 0)
        pred = torch.cat((pred, torch.from_numpy(outputs)), 0)
        
        print('%03d/%03d, time: %6.3f sec' % (index, len(test_loader), (timeit.default_timer() - start_time)))

    print('Elapsed time: %0.2f sec.' % (timeit.default_timer() - start))

    AUCs = [roc_auc_score(gt[:, i], pred[:, i]) if gt[:, i].sum() > 0 else np.nan for i in range(N_CLASSES)]
    print('The average AUC is %6.3f' % np.mean(AUCs))

    for i in range(N_CLASSES):
        print('The AUC of %s is %6.3f' % (CLASS_NAMES[i], AUCs[i]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['fp32', 'int8'], default='fp32', type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--data_dir', default='images', type=str)
    parser.add_argument('--test_image_list', default='labels/test_list.txt', type=str)
    args = parser.parse_args()
    print(vars(args))

    if args.mode == 'fp32':
        main(modelfile='densenet121.xml')
    elif args.mode == 'int8':
        main(modelfile='chexnet-pytorch.xml')
    else:
        parser.print_help()

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torchvision.transforms as transforms
from openvino.inference_engine import IECore, StatusCode
import argparse
import timeit

from datasets import ChestXrayDataSet
from model import CLASS_NAMES, N_CLASSES

def main(modelfile):
    model_xml = 'model/%s' % (modelfile)
    model_bin = model_xml.replace('.xml', '.bin')

    print('Creating Inference Engine')
    ie = IECore()
    net = ie.read_network(model=model_xml, weights=model_bin)
    
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

    # loading model to the plugin
    print('Loading model to the plugin')
    exec_net = ie.load_network(network=net, num_requests=args.num_requests, device_name='CPU')

    y_true = torch.FloatTensor()

    start = timeit.default_timer()

    for index, (data, labels) in enumerate(test_loader):
        if index == 100:
            break

        batch_size, n_crops, c, h, w = data.size()
        data = data.view(-1, c, h, w).numpy()

        images = np.zeros(shape=(model_batch_size, c, h, w))
        images[:n_crops * args.batch_size, :c, :h, :w] = data

        exec_net.requests[index].async_infer(inputs={input_blob: images})

        y_true = torch.cat((y_true, labels), 0)

    output_queue = list(range(args.num_requests))
    y_pred = list(range(args.num_requests))

    # wait the latest inference executions
    while True:
        for index in output_queue:
            infer_status = exec_net.requests[index].wait(0)

            if infer_status == StatusCode.RESULT_NOT_READY:
                continue

            if infer_status == StatusCode.OK:
                outputs = exec_net.requests[index].output_blobs[output_blob].buffer

                outputs = outputs[:n_crops * args.batch_size].reshape(args.batch_size, n_crops, -1)
                outputs = np.mean(outputs, axis=1)
                outputs = outputs[:args.batch_size, :outputs.shape[1]]

                y_pred[index] = torch.from_numpy(outputs)

                output_queue.remove(index)

        if len(output_queue) == 0:
            break

    y_pred = torch.cat(y_pred, 0)

    print('Elapsed time: %0.2fsec.' % (timeit.default_timer() - start), end='')

    aucs = [roc_auc_score(y_true[:, i], y_pred[:, i]) if y_true[:, i].sum() > 0 else np.nan for i in range(N_CLASSES)]
    auc_classes = ' '.join(['%5.3f' % (aucs[i]) for i in range(N_CLASSES)])
    print(' average AUC %5.3f (%s)' % (np.mean(aucs), auc_classes))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['fp32', 'int8'], default='fp32', type=str)
    parser.add_argument('--num_requests', default=100, type=int)
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

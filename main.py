import numpy as np
import torchvision.transforms as transforms
import torch
from sklearn.metrics import roc_auc_score
import argparse
import timeit

from datasets import ChestXrayDataSet
from model import DenseNet121, CLASS_NAMES, N_CLASSES

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using %s device.' % (device))
    
    # initialize and load the model
    net = DenseNet121(N_CLASSES)
    net.load_state_dict(torch.load(args.model_path, map_location=device))
    print('model state has loaded.')

    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net).to(device)
        print('Using %d cuda devices.' % (torch.cuda.device_count()))
    else:
        net = net.to(device)

    # switch to evaluate mode
    net.eval()

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

    # initialize the ground truth and output tensor
    y_true = torch.FloatTensor()
    y_pred = torch.FloatTensor()

    for index, (data, target) in enumerate(test_loader):
        start_time = timeit.default_timer()

        # each image has 10 crops.
        batch_size, n_crops, c, h, w = data.size()
        data = data.view(-1, c, h, w).to(device)

        with torch.no_grad():
            outputs = net(data)

        outputs_mean = outputs.view(batch_size, n_crops, -1).mean(1)

        y_true = torch.cat((y_true, target))
        y_pred = torch.cat((y_pred, outputs_mean))
            
        print('batch %5d/%5d %6.3fsec' % (index, len(test_loader), (timeit.default_timer() - start_time)))

    aucs = [roc_auc_score(y_true[:, i], y_pred[:, i]) for i in range(N_CLASSES)]
    auc_classes = ' '.join(['%5.3f' % (aucs[i]) for i in range(N_CLASSES)])
    print('The average AUC is %5.3f (%s)' % (np.mean(AUCs), auc_classes))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='model/model.pth', type=str)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--data_dir', default='images', type=str)
    parser.add_argument('--test_image_list', default='labels/test_list.txt', type=str)
    args = parser.parse_args()
    print(vars(args))

    main(args)

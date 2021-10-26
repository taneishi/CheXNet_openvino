import numpy as np
import torchvision.transforms as transforms
import torch
from sklearn.metrics import roc_auc_score
import argparse
import timeit

from read_data import ChestXrayDataSet
from model import DenseNet121, CLASS_NAMES, N_CLASSES

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using %s device.' % device)
    
    # initialize and load the model
    net = DenseNet121(N_CLASSES).to(device)

    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net).to(device)

    net.load_state_dict(torch.load(args.model_path, map_location=device))
    print('model state has loaded')

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
    gt = torch.FloatTensor()
    pred = torch.FloatTensor()

    # switch to evaluate mode
    net.eval()

    for index, (data, target) in enumerate(test_loader):
        start_time = timeit.default_timer()

        target = target.to(device)
        batch_size, n_crops, c, h, w = data.size()
        data = data.view(-1, c, h, w).to(device)

        with torch.no_grad():
            output = net(data)

        output_mean = output.view(batch_size, n_crops, -1).mean(1)

        gt = torch.cat((gt, target))
        pred = torch.cat((pred, output_mean))
            
        print('batch %03d/%03d %6.3fsec' % (index, len(test_loader), (timeit.default_timer() - start_time)))

    AUCs = [roc_auc_score(gt[:, i], pred[:, i]) for i in range(N_CLASSES)]
    print('The average AUC is %6.3f' % np.mean(AUCs))

    for i in range(N_CLASSES):
        print('The AUC of %s is %6.3f' % (CLASS_NAMES[i], AUCs[i]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='model/model.pth', type=str)
    parser.add_argument('--batch_size', default=24, type=int)
    parser.add_argument('--data_dir', default='images', type=str)
    parser.add_argument('--test_image_list', default='labels/test_list.txt', type=str)
    args = parser.parse_args()
    print(vars(args))

    main(args)

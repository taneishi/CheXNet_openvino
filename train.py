import numpy as np
import torchvision
import torch
from sklearn.metrics import roc_auc_score
import argparse
import timeit

from datasets import ChestXrayDataSet
from model import DenseNet121, CLASS_NAMES, N_CLASSES

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using %s device.' % device)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    train_dataset = ChestXrayDataSet(
            data_dir=args.data_dir,
            image_list_file=args.train_image_list,
            transform=transform,
            )

    train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            pin_memory=False)

    val_dataset = ChestXrayDataSet(
            data_dir=args.data_dir,
            image_list_file=args.val_image_list,
            transform=transform,
            )

    val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            pin_memory=False)

    print('training images: %d' % (len(train_loader)))
    print('validation images: %d' % (len(val_loader)))
    
    # initialize and load the model
    net = DenseNet121(N_CLASSES)
    #net.load_state_dict(torch.load(args.model_path, map_location=device))
    print('model state has loaded')

    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    net = net.to(device)

    criterion = torch.nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(args.epochs):
        start_time = timeit.default_timer()

        # initialize the ground truth and output tensor
        y_true = torch.FloatTensor()
        y_pred = torch.FloatTensor()

        train_loss = 0
        net.train()
        for index, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            loss = criterion(outputs, labels)

            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            y_true = torch.cat((y_true, labels), 0)
            y_pred = torch.cat((y_pred, outputs.detach()), 0)
                
            print('\rbatch %5d/%5d train loss %6.4f' % (index+1, len(train_loader), train_loss / (index+1)), end='')
            print(' %6.3fsec' % (timeit.default_timer() - start_time), end='')

            AUCs = [roc_auc_score(y_true[:, i], y_pred.detach().numpy()[:, i]) if y_true[:, i].sum() > 0 else np.nan for i in range(N_CLASSES)]
            print(' average AUC is %6.3f' % np.mean(AUCs), end='')

        print('\repoch %5d/%5d train loss %6.4f' % (epoch+1, args.epochs, train_loss / len(train_loader)), end='')
        print(' %6.3fsec' % (timeit.default_timer() - start_time))

        AUCs = [roc_auc_score(y_true[:, i], y_pred.detach().numpy()[:, i]) if y_true[:, i].sum() > 0 else np.nan for i in range(N_CLASSES)]
        print('The average AUC is %6.3f' % np.mean(AUCs))

        for i in range(N_CLASSES):
            print('The AUC of %s is %6.3f' % (CLASS_NAMES[i], AUCs[i]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='model/model.pth', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--data_dir', default='images', type=str)
    parser.add_argument('--train_image_list', default='labels/train_list.txt', type=str)
    parser.add_argument('--val_image_list', default='labels/val_list.txt', type=str)
    args = parser.parse_args()
    print(vars(args))

    main(args)

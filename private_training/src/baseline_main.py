import os
import torch
from torch import nn
from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt

from datasets import get_dataset
from options import args_parser
from utils import test_inference
from models import CNNMnistRelu, CNNMnistTanh, CNNFashion_MnistRelu, CNNFashion_MnistTanh
from torchvision import models 
from torchsummary import summary

if __name__ == '__main__':

    ############# Common ###################
    args = args_parser()
    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # load datasets
    train_dataset, test_dataset, _ = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            if args.activation == 'relu':
                global_model = CNNMnistRelu()
            elif args.activation == 'tanh':
                global_model = CNNMnistTanh()
            global_model.to(device)
            summary(global_model, input_size=(1, 28, 28), device=device)
        elif args.dataset == 'fmnist':
            if args.activation == 'relu':
                global_model = CNNFashion_MnistRelu()
            elif args.activation == 'tanh':
                global_model = CNNFashion_MnistTanh()
            global_model.to(device)
            summary(global_model, input_size=(1, 28, 28), device=device)
        elif args.dataset == 'cifar10':
            # global_model = models.resnet18(num_classes=10)  
            if args.activation == 'relu':
                global_model = CNNCifar10Relu()
            elif args.activation == 'tanh':
                global_model = CNNCifar10Tanh()
            global_model.to(device)
            summary(global_model, input_size=(3, 32, 32), device=device)
        elif args.dataset == 'dr':
            if args.activation == 'relu':
                global_model = CNNCifar10Relu()
            elif args.activation == 'tanh':
                global_model = CNNCifar10Tanh()
            global_model.to(device)
            summary(global_model, input_size=(3, 32, 32), device=device)
        elif args.dataset == 'cifar100':    
            global_model = models.resnet50(num_classes=100)
            global_model.to(device)
            summary(global_model, input_size=(3, 32, 32), device=device)
    else:
        exit('Error: unrecognized model')
    ############# Common ###################

    # Set the model to train and send it to device.
    global_model.train()    
    
    # Set optimizer and criterion
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr)

    trainloader = DataLoader(train_dataset, batch_size=args.local_bs, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    
    epoch_loss = []

    # train_log = []
    test_log = []
    for epoch in range(args.epochs):
        batch_loss = []

        for batch_idx, (images, labels) in enumerate(trainloader):
            
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = global_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # if batch_idx % args.verbose == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch+1, batch_idx * len(images), len(trainloader.dataset),
            #         100. * batch_idx / len(trainloader), loss.item()))
            batch_loss.append(loss.item())

        loss_avg = sum(batch_loss)/len(batch_loss)
        print("\nEpoch:", epoch+1)
        print('Train loss:', loss_avg)
        epoch_loss.append(loss_avg)

        # training accuracy
        # _acc, _loss = test_inference(args, global_model, train_dataset)
        # print('Train on', len(train_dataset), 'samples')
        # print("Train Accuracy: {:.2f}%".format(100*_acc))
        # train_log.append([_acc, _loss])

        # testing accuracy
        _acc, _loss = test_inference(args, global_model, test_dataset)
        
        print('Test on', len(test_dataset), 'samples')
        print("Test Accuracy: {:.2f}%".format(100*_acc))
        test_log.append([_acc, _loss])

        log_dir = './test_log/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        with open(log_dir+args.exp_name+'_test_log.txt', 'w') as f:
            for item in test_log:
                f.write("%s\n" % item)

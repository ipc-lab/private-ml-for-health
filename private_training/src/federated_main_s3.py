import os
import copy
import time
import pickle
import numpy as np
import torch

from torchsummary import summary
import torch.nn as nn

from options import args_parser
from update_s3 import LocalUpdate
from utils import test_inference
from models import CNNMnistRelu, CNNMnistTanh
from models import CNNFashion_MnistRelu, CNNFashion_MnistTanh
from models import CNNCifar10Relu, CNNCifar10Tanh
from utils import average_weights, exp_details
from datasets import get_dataset
from torchvision import models
from logging_results import logging

from opacus.dp_model_inspector import DPModelInspector
from opacus.utils import module_modification
from opacus import PrivacyEngine


if __name__ == '__main__':
    
    ############# Common ###################
    args = args_parser()    
    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'    
    
    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    
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
            global_model = models.squeezenet1_1(pretrained=True)           
            # for param in global_model.parameters():
            #     param.requires_grad = False
            global_model.classifier[1] = nn.Conv2d(512, 5, kernel_size=(1,1), stride=(1,1))
            global_model.num_classes = 5
            global_model.to(device)
            summary(global_model, input_size=(3, 224, 224), device=device)
    else:
        exit('Error: unrecognized model')
    ############# Common ###################

    ######### DP Model Compatibility #######
    if args.withDP:
        try:
            inspector = DPModelInspector()
            inspector.validate(global_model)
            print("Model's already Valid!\n")
        except:
            global_model = module_modification.convert_batchnorm_modules(global_model)
            inspector = DPModelInspector()
            print(f"Is the model valid? {inspector.validate(global_model)}")
            print("Model is convereted to be Valid!\n")        
    ######### DP Model Compatibility #######

    ######### Local Models and Optimizers #############
    u_steps = np.zeros(args.num_users)  
    epsilons = np.zeros(args.num_users)    
    ######## Local  Models and Optimizers #############

    # Training
    train_loss = []
    test_log = []
    epsilon_log = []

    for epoch in range(args.epochs):    
        ## Sample the users ##        
        idxs_users = np.random.choice(range(args.num_users),
                                      max(int(args.frac * args.num_users), 1),
                                      replace=False)
        #####

        local_weights, local_losses = [], []        
        
        for u in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, 
                                      u_id=u, idxs=user_groups[u])
            w, loss, u_step, epsilon = local_model.update_weights(
                                                    model=copy.deepcopy(global_model),
                                                    global_round=epoch,
                                                    u_step= u_steps[u])
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            u_steps[u] = u_step 
            epsilons[u] = epsilon

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)
        # for u in range(args.num_users):
        #     local_models[u].load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)        
        train_loss.append(loss_avg)

        _acc, _loss = test_inference(args, global_model, test_dataset)        
        test_log.append([_acc, _loss])  

        if args.withDP:                        
            epsilon_log.append(list(epsilons))
        else:
            epsilon_log = None
        
        logging(args, epoch, train_loss, test_log, epsilon_log)


    
    
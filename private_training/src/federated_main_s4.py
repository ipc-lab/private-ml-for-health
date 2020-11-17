import os
import copy
import time
import pickle
import numpy as np
import torch
from torch import nn

from torchsummary import summary

from options import args_parser
from update_s4 import LocalUpdate
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
    local_models = []
    local_optimizers = []
    local_privacy_engine = []

    for u in range(args.num_users):
        local_models.append(copy.deepcopy(global_model))

        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(local_models[u].parameters(), lr=args.lr, 
                                        momentum=args.momentum)        
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(local_models[u].parameters(), lr=args.lr)             

        if args.withDP:
          # This part is buggy intentionally. It makes privacy engine avoid giving error with vhp.
            
            privacy_engine = PrivacyEngine(
                local_models[u],
                batch_size = int(len(train_dataset)*args.sampling_prob), 
                sample_size = len(train_dataset), 
                alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                noise_multiplier = args.noise_multiplier/np.sqrt(args.num_users),
                max_grad_norm =  args.max_grad_norm,
            )

            privacy_engine.attach(optimizer)            
            local_privacy_engine.append(privacy_engine)

        local_optimizers.append(optimizer)


    if args.optimizer == 'sgd':
        g_optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr, 
                                    momentum=args.momentum)        
    elif args.optimizer == 'adam':
        g_optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr)        
    if args.withDP:
        local_dataset_size = int(len(train_dataset)/args.num_users)
        actual_train_ds_size = local_dataset_size*args.num_users
        global_privacy_engine = PrivacyEngine(
            global_model,
            batch_size = int(actual_train_ds_size*args.sampling_prob),
            sample_size = actual_train_ds_size,
            alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
            noise_multiplier = args.noise_multiplier,
            max_grad_norm =  args.max_grad_norm)  
        global_privacy_engine.attach(g_optimizer)
    ######## Local  Models and Optimizers #############

    # Training
    train_loss = []
    test_log = []
    epsilon_log = []
    
    print("Avg batch_size: ", int(actual_train_ds_size*args.sampling_prob))

    for epoch in range(args.epochs):    
        ## Sample the users ##        
        idxs_users = np.random.choice(range(args.num_users),
                                      max(int(args.frac * args.num_users), 1),
                                      replace=False)
        #####
        local_weights, local_losses = [], []        
        

        for u in idxs_users:
            
            torch.cuda.empty_cache()

            local_model = LocalUpdate(args=args, dataset=train_dataset, 
                                      u_id=u, idxs=user_groups[u], 
                                      sampling_prob=args.sampling_prob,
                                      optimizer = local_optimizers[u])

            w, loss, local_optimizers[u] = local_model.update_weights(
                                                    model=local_models[u],
                                                    global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)
        for u in range(args.num_users):
            local_models[u].load_state_dict(global_weights)

        if epoch !=0 and epoch%30==0:
            torch.cuda.empty_cache()          
            loss_avg = sum(local_losses) / len(local_losses)        
            train_loss.append(loss_avg)

            _acc, _loss = test_inference(args, global_model, test_dataset)        
            test_log.append([_acc, _loss])  
          
            if args.withDP:
                global_privacy_engine.steps = epoch+1
                epsilons, _ = global_privacy_engine.get_privacy_spent(args.delta)                                        
                epsilon_log.append([epsilons])
            else:
                epsilon_log = None

            logging(args, epoch, train_loss, test_log, epsilon_log)
            print(global_privacy_engine.steps)

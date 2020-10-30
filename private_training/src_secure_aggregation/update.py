#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from opacus import FLPrivacyEngine
import numpy as np


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, u_id, idxs, logger):
        self.u_id = u_id
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """

        ### Moh ###

        # split indexes for train, validation, and test (80, 10, 10)
        # idxs_train = idxs[:int(0.8*len(idxs))]
        # idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        # idxs_test = idxs[int(0.9*len(idxs)):]

        # trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
        #                          batch_size=self.args.local_bs, shuffle=True)
        # validloader = DataLoader(DatasetSplit(dataset, idxs_val),
        #                          batch_size=int(len(idxs_val)/10), shuffle=False)
        # testloader = DataLoader(DatasetSplit(dataset, idxs_test),
        #                         batch_size=int(len(idxs_test)/10), shuffle=False)
        # return trainloader, validloader, testloader

        idxs_train = idxs[:int(np.round((1/3)*len(idxs)))]
        idxs_val = idxs[int(np.round((1/3)*len(idxs))):int(np.round((2/3)*len(idxs)))]
        idxs_test = idxs[int(np.round((2/3)*len(idxs))):]
        
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)), shuffle=False)
        return trainloader, validloader, testloader
        ###########

        
    def update_weights(self, model, global_round, u_step=0):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        #### DPSGD OPACUS ####
        if self.args.withDP:
            privacy_engine = FLPrivacyEngine(
                model,
                batch_size=self.args.virtual_batch_size,
                sample_size=len(self.trainloader.dataset),
                alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                noise_multiplier=self.args.noise_multiplier,
                max_grad_norm=self.args.max_grad_norm,
            )
            privacy_engine.attach(optimizer)
            assert self.args.virtual_batch_size % self.args.local_bs == 0 # VIRTUAL_BATCH_SIZE should be divisible by BATCH_SIZE
            virtual_batch_rate = int(self.args.virtual_batch_size / self.args.local_bs)
            
            ### This is for keeping the track of epsilom across global rounds.
            privacy_engine.steps = u_step
        #############
        
        for iter in range(self.args.local_ep):
            batch_loss = []
            optimizer.zero_grad()
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()

                ### Moh ####
                if self.args.withDP:
                # optimizer.step()
                # take a real optimizer step after N_VIRTUAL_STEP steps t
                    if ((batch_idx + 1) % virtual_batch_rate == 0) or ((batch_idx + 1) == len(self.trainloader)):
                        optimizer.step()
                        optimizer.zero_grad()                        
                    else:                        
                        optimizer.virtual_step() # take a virtual step
                #else:
                optimizer.step()
                optimizer.zero_grad()
                #############

                if self.args.verbose and (batch_idx % self.args.verbose == 0):
                    if self.args.withDP:
                        epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(self.args.delta)
                        print('| Global Round : {} | User ID : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f} | ε = {:.2f} | α = {:.2f} | δ = {}'.format(
                            global_round+1, self.u_id, iter, batch_idx * len(images),
                            len(self.trainloader.dataset),
                            100. * batch_idx / len(self.trainloader), loss.item(), 
                            epsilon, best_alpha, self.args.delta))
                    #else:
                    print('| Global Round : {} | User ID : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f} '.format(
                        global_round+1, self.u_id, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
            if self.args.withDP:
                state_dict = model.state_dict()
                if not self.args.smpc:
                    grad_l2_sens = self.args.lr * self.args.max_grad_norm * self.args.local_ep / self.args.local_bs
                else:
                    num_of_active_users = max(int(self.args.frac * self.args.num_users), 1)
                    grad_l2_sens = self.args.lr * self.args.max_grad_norm * self.args.local_ep / (self.args.local_bs * np.sqrt(num_of_active_users))
                for i in state_dict:
                    noise = optimizer.privacy_engine._generate_noise(grad_l2_sens, state_dict[i])
                    model.state_dict()[i] += noise
        if self.args.withDP:
            return model.state_dict(), sum(epoch_loss) / len(epoch_loss), privacy_engine.steps
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), 0
    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss

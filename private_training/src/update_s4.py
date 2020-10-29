import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from opacus import PrivacyEngine


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
        return image.clone().detach(), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, u_id, idxs, sampling_prob, optimizer): ####!!
        self.u_id = u_id
        self.args = args
        self.trainloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.dataset_size = len(dataset) ####!!
        self.sasampling_prob = sampling_prob ####!!
        self.optimizer = optimizer

    def train_val_test(self, dataset, idxs):
        """
        Returns train and test dataloaders for a given dataset
        and user indexes.
        """
        _split = 0
        if self.args.local_test_split > 0.0:
            _split = max(int(np.round((self.args.local_test_split)*len(idxs))), 1)
        
        idxs_train = idxs[_split:]        
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size = len(idxs), ####!!
                                 shuffle=True, drop_last=True)        
        testloader = None
        if _split > 0:
            idxs_test = idxs[:_split]
            testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)), shuffle=False)
        return trainloader, testloader
        ###########

        
    def update_weights(self, model, global_round):
        # Set mode to train model
        model.to(self.device)
        model.train()
        epoch_loss = []
    

        for iter in range(self.args.local_ep):            
            batch_loss = []
            self.optimizer.zero_grad()
            if self.args.withDP:
                virtual_batch_rate = int(self.args.virtual_batch_size / self.args.local_bs)            
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                
                ####!!
                indices = np.random.permutation(len(labels))
                rnd_sampled = np.random.binomial(len(labels), self.sasampling_prob)
                if rnd_sampled > 0:
                    images = images[indices][:rnd_sampled]
                    labels = labels[indices][:rnd_sampled]
                else:
                    return model.state_dict(), 0., self.optimizer
                ####!!                
              
                images, labels = images.to(self.device), labels.to(self.device)
                model_preds = model(images)
                loss = self.criterion(model_preds, labels)
                loss.backward()
                
                if self.args.withDP:
                    # take a real optimizer step after N_VIRTUAL_STEP steps t                                        
                    if ((batch_idx + 1) % virtual_batch_rate == 0) or ((batch_idx + 1) == len(self.trainloader)):
                        self.optimizer.step()
                        self.optimizer.zero_grad()                        
                    else:                        
                        self.optimizer.virtual_step() # take a virtual step                        
                else:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                #############
                batch_loss.append(loss.item())
                
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), self.optimizer
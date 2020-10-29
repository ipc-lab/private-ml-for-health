import numpy as np
import copy
import torch
import matplotlib.image as mpimg
import urllib.request
import zipfile
import os
import pandas as pd
from torchvision import datasets, transforms
from sampling import dist_datasets_iid, dist_datasets_noniid
from options import args_parser
from torch.utils.data import Dataset, TensorDataset
import gdown

class DRDataset(Dataset):
    def __init__(self, data_label, data_dir, transform):
        super().__init__()
        self.data_label = data_label
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_label)

    def __getitem__(self, index):
        img_name = self.data_label.id_code[index] + '.png'
        label = self.data_label.diagnosis[index]
        img_path = os.path.join(self.data_dir, img_name)
        image = mpimg.imread(img_path)
        image = (image + 1) * 127.5
        image = image.astype(np.uint8)
        image = self.transform(image)
        return image, label


def get_dataset(args):

    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        if args.dataset == 'cifar10':
            train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)        
            test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)
        elif args.dataset == 'cifar100':
            train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                       transform=apply_transform)
            test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                      transform=apply_transform)    

    elif args.dataset == 'mnist' or args.dataset =='fmnist':
        if args.dataset == 'mnist':
            apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])

            data_dir = '../data/mnist/'
            train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)
            test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        else:
            apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])

            data_dir = '../data/fmnist/'
            train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)
            test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)
    
    elif args.dataset == 'dr':

        if args.dr_from_np == 1:
            _x = torch.Tensor(np.load("dr_train_images.npy"))
            _y = torch.Tensor(np.load("dr_train_labels.npy")).long()
            train_dataset = TensorDataset(_x,_y)
            _x = torch.Tensor(np.load("dr_test_images.npy"))
            _y = torch.Tensor(np.load("dr_test_labels.npy")).long()
            test_dataset = TensorDataset(_x,_y)            
        
        else:
            data_dir = '../data/diabetic_retinopathy/'
            if not os.path.exists(data_dir):               
                os.makedirs(data_dir)
            
            #download ZIP, unzip it, delete zip file
            dataset_url = "https://drive.google.com/uc?id=1G-4UhPKiQY3NxQtZiWuOkdocDTW6Bw0u"
            zip_path = data_dir + 'images.zip'
            gdown.download(dataset_url, zip_path, quiet=False)
            print("Extracting...!")
        
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            print("Extracted!")
            os.remove(zip_path)

            #download train and test dataframes
            test_csv_link = 'https://drive.google.com/uc?id=1dmeYLURzEvx962th4lAxaVN3r6nlhTjS'
            train_csv_link = 'https://drive.google.com/uc?id=1SMb9CRHjB6UH2WnTZDFVSgpA6_nh75qN'
            test_csv_path = data_dir + 'test_set.csv'
            train_csv_path = data_dir + 'train_set.csv'
            urllib.request.urlretrieve(train_csv_link, train_csv_path)
            urllib.request.urlretrieve(test_csv_link, test_csv_path)
            df_train = pd.read_csv(train_csv_path)
            df_test = pd.read_csv(test_csv_path)

            #create train and test datasets
            apply_transform = transforms.Compose([transforms.ToPILImage(mode='RGB'),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.Resize(265),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            image_directory = data_dir + 'images/'
            train_dataset = DRDataset(data_label = df_train, data_dir = image_directory,
                                        transform = apply_transform)
            test_dataset = DRDataset(data_label = df_test, data_dir = image_directory,
                                        transform = apply_transform)


    if args.sub_dataset_size > 0:
        rnd_indices = np.random.RandomState(seed=0).permutation(len(train_dataset.data))        
        train_dataset.data = train_dataset.data[rnd_indices]
        if torch.is_tensor(train_dataset.targets):
            train_dataset.targets = train_dataset.targets[rnd_indices]    
        else:
            train_dataset.targets = torch.tensor(train_dataset.targets)[rnd_indices]
        train_dataset.data = train_dataset.data[:args.sub_dataset_size]
        train_dataset.targets = train_dataset.targets[:args.sub_dataset_size]
        print("\nThe chosen sub dataset has the following shape:")
        print(train_dataset.data.shape, train_dataset.targets.shape,"\n")        

    if args.iid:                   
        user_groups = dist_datasets_iid(train_dataset, args.num_users)         
    else:
        user_groups = dist_datasets_noniid(train_dataset, args.num_users,
                                            num_shards=1000,                                                
                                            unequal=args.unequal)    
    
    return train_dataset, test_dataset, user_groups

## For test
if __name__ == '__main__':
    args = args_parser()
    train_dataset, test_dataset, user_groups = get_dataset(args)
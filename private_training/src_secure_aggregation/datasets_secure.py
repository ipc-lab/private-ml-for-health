import numpy as np
import copy
import torch
from torchvision import datasets, transforms
from sampling_secure import dist_datasets_iid, dist_datasets_noniid
from options import args_parser


def get_train_dataset(args,rank):
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        data_dir = './data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        if args.dataset == 'cifar10':
            train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                             transform=apply_transform)
        elif args.dataset == 'cifar100':
            train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                              transform=apply_transform)
    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))])

            data_dir = './data/mnist/'
            train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                           transform=apply_transform)
        else:
            apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))])

            data_dir = './data/fmnist/'
            train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                                  transform=apply_transform)

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
        print(train_dataset.data.shape, train_dataset.targets.shape, "\n")

    if args.iid:
        user_groups = dist_datasets_iid(train_dataset, args.num_users)
    else:
        user_groups = dist_datasets_noniid(train_dataset, args.num_users,
                                           num_shards=1000,unequal=args.unequal)

    data_ind = user_groups[rank-1]
    train_dataset = [train_dataset[int(i)] for i in data_ind]
    return(train_dataset, user_groups[rank-1])

def get_test_dataset(args):
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        data_dir = './data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        if args.dataset == 'cifar10':
            test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
            transform = apply_transform)
        elif args.dataset == 'cifar100':
            test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
            transform = apply_transform)
    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))])

            data_dir = './data/mnist/'
            test_dataset = datasets.MNIST(data_dir, train=False, download=True,
            transform=apply_transform)
        else:
            apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))])

            data_dir = './data/fmnist/'
            test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
            transform = apply_transform)

    return(test_dataset)

## For test
if __name__ == '__main__':
    args = args_parser()
    rank=2
    train_dataset, user_groups = get_train_dataset(args,rank)
    test_dataset = get_test_dataset(args)
    print(train_dataset)
    print(test_dataset)
    print(np.unique([len(v) for k, v in user_groups.items()]))

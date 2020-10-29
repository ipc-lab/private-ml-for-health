import os
import copy
import time
import pickle
import numpy as np
import torch
from options import args_parser
from datasets import get_dataset



if __name__ == '__main__':
    
    ############# Common ###################
    args = args_parser()    
    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'    
    
    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)
    
    print("train_dataset size:", len(train_dataset))
    print("test_dataset size:", len(test_dataset))
    # print("data shape:", train_dataset[0][0].shape)
    print("train")
    dr_images = []
    dr_labels = []
    for i in range(len(train_dataset)):        
        _image, _label = train_dataset[i]
        dr_images.append(_image.numpy())
        dr_labels.append(_label)
        print("  ", i, end="\r")
    print("")
    dr_images = np.array(dr_images)
    dr_labels = np.array(dr_labels)
    np.save("dr_train_images.npy", dr_images)
    np.save("dr_train_labels.npy", dr_labels)
    print("test")
    dr_images = []
    dr_labels = []
    for i in range(len(test_dataset)):        
        _image, _label = test_dataset[i]
        dr_images.append(_image.numpy())
        dr_labels.append(_label)
        print("  ", i, end="\r")
    print("")
    dr_images = np.array(dr_images)
    dr_labels = np.array(dr_labels)
    np.save("dr_test_images.npy", dr_images)
    np.save("dr_test_labels.npy", dr_labels)

    print("Done!")
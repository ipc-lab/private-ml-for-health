
import os
import numpy as np
from utils import test_inference


def logging(args, epoch, train_loss, test_log, epsilon_log=None):
    print("\nEpoch:", epoch+1)

    log_dir_train = './logs/train_log/'
    if not os.path.exists(log_dir_train):
        os.makedirs(log_dir_train)
    
    print('Average train loss:', train_loss[-1])    
    with open(log_dir_train+args.exp_name+'_train.txt', 'w') as f:
        for item in train_loss:
            f.write("%s\n" % item)

    log_dir_test = './logs/test_log/'
    if not os.path.exists(log_dir_test):
        os.makedirs(log_dir_test)

    print("Test Accuracy: {:.2f}%".format(100*test_log[-1][0]))      
    with open(log_dir_test+args.exp_name+'_test.txt', 'w') as f:
        for item in test_log:
            f.write("%s\n" % item)

    if epsilon_log:
        log_dir_eps = './logs/privacy_log/'
        if not os.path.exists(log_dir_eps):
            os.makedirs(log_dir_eps)
        print("epsilons: max {:.2f},  mean {:.2f}, std {:.2f}".format(np.max(epsilon_log[-1]), np.mean(epsilon_log[-1]), np.std(epsilon_log[-1])))
        with open(log_dir_eps+args.exp_name+'_eps.txt', 'w') as f:
            for item in epsilon_log:
                f.write("%s\n" % item)


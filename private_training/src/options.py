import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=1,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=1,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=1.,
                        help='the fraction of clients')
    parser.add_argument('--local_ep', type=int, default=4,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=50,
                        help="local batch size: B")

    # optimizer arguments
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.0)')
    
    

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--activation', type=str, default="relu", help='activation')

    ## DP arguments
    parser.add_argument('--withDP', type=int, default=0, help='WithDP')
    parser.add_argument('--max_grad_norm', type=float, default=1.5, help='DP MAX_GRAD_NORM')
    parser.add_argument('--noise_multiplier', type=float, default=.75, help='DP NOISE_MULTIPLIER')
    parser.add_argument('--delta', type=float, default=.00001, help='DP DELTA')
    parser.add_argument('--virtual_batch_size', type=int, default=50, help='DP VIRTUAL_BATCH_SIZE')
    parser.add_argument('--sampling_prob', type=int, default=0.03425 , help='sampling_prob') 

    # dataset arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--sub_dataset_size', type=int, default=-1, help='To reduce original data to a smaller \
                        sized dataset. For experimental purposes.')
    parser.add_argument('--local_test_split', type=float, default=0., help='local_test_split')                    
    parser.add_argument('--dr_from_np', type=float, default=0, help='for diabetic_retinopathy dataset')                    
    parser.add_argument('--exp_name', type=str, default="exp_results", help="The name of current experiment for logging.")
   

    args = parser.parse_args()
    return args

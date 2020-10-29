import numpy as np
from torchvision import datasets, transforms

def dist_datasets_iid(dataset, num_users):
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.RandomState(seed=i).choice(all_idxs, num_items,replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])

        if i%5000==0:
            print(i//5000)
            
    return dict_users

def dist_datasets_noniid(dataset, num_users, num_shards=None, num_imgs=None, 
                        unequal=0, min_shard = None, max_shard = None):
    
    if num_shards is None:
        num_shards = num_users
    if num_imgs is None:
        num_imgs = len(dataset)//num_shards
    assert num_shards//num_users > 0

    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    if not unequal:
        random_shard_size = np.array([num_shards//num_users]*num_users)
    else:  
        if min_shard is None:
            min_shard = min(1,(num_shards//num_users)-1)
        if max_shard is None:
            max_shard = (num_shards//num_users)+1
        # Divide the shards into random chunks for every client
        # s.t. the sum of these chunks = num_shards        
        random_shard_size = np.random.RandomState(seed=0).randint(min_shard, max_shard+1,
                                            size=num_users)        
        random_shard_size = np.around(random_shard_size /
                                    sum(random_shard_size) * num_shards)
        random_shard_size = random_shard_size.astype(int)
        diffs = sum(random_shard_size)-num_shards
        
        if diffs > 0:
            random_shard_size = np.sort(random_shard_size)[::-1]
        else:
            random_shard_size = np.sort(random_shard_size)
        for i in range(int(abs(diffs))):
            random_shard_size[i] -= np.sign(diffs)
    print(random_shard_size)
    for i in range(num_users):
        shard_size = random_shard_size[i]
        rand_set = set(np.random.RandomState(seed=i).choice(idx_shard, shard_size,replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], 
                                            idxs[rand*num_imgs:(rand+1)*num_imgs]),axis=0)
        if i%5000==0:
            print(i//5000)

    return dict_users

## For test
if __name__ == '__main__':
    dataset_train = datasets.MNIST('./data/mnist/',
                                    train=True, download=True,
                                    transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,),
                                                            (0.3081,))
                                    ]))

    # d = dist_datasets_iid(dataset_train, num_users)
    d = dist_datasets_noniid(dataset_train, num_users=50, unequal=1,
    num_shards=100, num_imgs=600, min_shard = 4, max_shard = 4)
    print(np.unique([len(v) for k, v in d.items()]))
    
# Private Training 

This repository produces source code of the training stage for the proposed method in the report.

> In this repository we have used some part of codes from : https://github.com/AshwinRJ/Federated-Learning-PyTorch

## Dataset Preparation 

To prepare the [diabetic retinopathy](https://www.kaggle.com/c/aptos2019-blindness-detection/notebooks?sortBy=scoreDescending) dataset:

 1. First (and just for one time) you need to run this file: `src/dr_dataset_to_numpy.py` with the following commmand:

    > python src/dr_dataset_to_numpy.py --dataset=dr --dr_from_np=0

 2. Running the above command will take sometimes to download the dataset and transforms it into numpy arrays and saves it.
 
 3. After this is completed, for the rest of experiments you need to set `--dr_from_np=1`
 

## How To Run Experiments

In the following we explain how you can reproduce the results shown in Figure 1 of the report.

### 1. Non-Private Centralized (C)

> python src/federated_main_s2.py  --epochs=1 --num_users=1 --frac=1. --local_ep=100  --local_bs=50  --virtual_batch_size=50  --optimizer='sgd' --lr=0.001 --momentum=0.9 --dataset='dr' --dr_from_np=1 --gpu="cuda:0" --withDP=0


### 2. Centralized with Central DP (CDP)

> python src/federated_main_s2.py  --epochs=1 --num_users=1 --frac=1. --local_ep=100  --local_bs=50  --virtual_batch_size=50  --optimizer='sgd' --lr=0.001 --momentum=0.9 --dataset='dr' --dr_from_np=1 --gpu="cuda:0" --withDP=1 --max_grad_norm = 2. --noise_multiplier = 1. --delta = 1e-4


### 3. Non-Private Federated (F)

> python src/federated_main_s3.py  --epochs=100 --num_users=10 --frac=.5 --local_ep=5  --local_bs=50  --virtual_batch_size=50  --optimizer='sgd' --lr=0.002 --momentum=0.9 --dataset='dr' --dr_from_np=1 --gpu="cuda:0" --withDP=0

### 4. Federated with Parallel DP (FPDP)

> python src/federated_main_s3.py  --epochs=100 --num_users=10 --frac=.5 --local_ep=5  --local_bs=50  --virtual_batch_size=50  --optimizer='sgd' --lr=0.002 --momentum=0.9 --dataset='dr' --dr_from_np=1 --gpu="cuda:0" --withDP=1 --max_grad_norm = 2. --noise_multiplier = 2. --delta = 1e-4

### 5. Our Federated with Semi-Central DP (FSCDP)

> python src/federated_main_s4.py  --epochs=30001 --num_users=10 --frac=1. --local_ep=1  --local_bs=1  --virtual_batch_size=1  --optimizer='sgd' --lr=0.002 --momentum=0.9 --dataset='dr' --dr_from_np=1 --gpu="cuda:0" --withDP=1 --max_grad_norm = 2. --noise_multiplier = 1.15 --delta = 1e-4 --sampling_prob= 0.03425


Note that in FSCDP `--epochs=30001` is actually the number of iterations and not epochs. Based on the setting, `--epochs=30001` in FSCDP is similar to having 100 epochs in other FPDP and F setting. Moreover, `--sampling_prob= 0.03425` will translate into a batch size of 10 per user, 100 in total.

### 6. Secure aggregation with Homomorphic Encryption: (under development, so there might still be bugs)
Install PySEAL by running the shell file `build_pyseal.sh`:
``` 
> cd [path to build_pyseal.sh>]
> chmod +x build_pyseal.sh
> ./build_pyseal.sh
```
This downloads and builds PySEAL from the source: https://github.com/Lab41/PySEAL.

Ensure that mpi4py is installed. Else install it with `pip install mpi4py`.

For running 3 processes, with rank 0 process being the server, and rank 1 and rank 2 processes being 2 federated workers (hospitals), run the following for training on MNIST dataset:
> mpiexec -n 3 python src_secure_aggregation/FLDP_secure_aggregation.py --model=cnn --dataset=mnist --iid=1 --withDP=0 --local_bs=32 --num_users=2 --frac=.5 --local_ep=1 --epochs=20 --verbose=1000

## Tutorial:

Please see the file `JNotebook_running_FSCDP_on_Colab.ipynb` if you want to perform training on Google Colab.


* Updated for running Tutorial:

If you receive an error about `Nvidia CUDA`, then there are two solutions at the moment:

  1. In the first line at the beginning of the tutorial, use `torch==1.7.0+cu101` instead of `torch==1.6.0+cu101`.
  
  2. Or, remove that line and just use the second line to install `Opacus`, then restart the notebook, and finally in the colab system files go to `usr —> local —> lib —> python 3.6 —> dist-packages —> opacus` and open `privacy_engine.py` and turn the line `import torchcsprng as csprng` into comment `#import torchcsprng as csprng`.

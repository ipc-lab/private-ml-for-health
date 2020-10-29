# Private Training 

This repository produces source code of the training stage for the proposed method in the report.

> In this repository we have used some part of codes from : https://github.com/AshwinRJ/Federated-Learning-PyTorch

## How To Run Code

In the following we explain how you can reproduce the results shown in Figure 1 of the report.

### 1. Non-Private Centralized (C)

> python src/federated_main_s2.py  --epochs = 1 --num_users = 1 --frac = 1. --local_ep = 100  --local_bs = 50  --virtual_batch_size = 50  --optimizer = 'sgd' --lr = 0.001 --momentum = 0.9 --dataset = 'dr' --dr_from_np = 1 --gpu = "cuda:0" --withDP = 0





### 2. Non-Private Federated (F)
### 3. Centralized with Central DP (CDP)
### 4. Federated with Parallel DP (FPDP)
### 5. Our Federated with Semi-Central DP (FSCDP)



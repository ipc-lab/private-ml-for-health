#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate
from utils import test_inference
from models import CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import average_weights, exp_details
from datasets_secure import get_train_dataset, get_test_dataset

from opacus.dp_model_inspector import DPModelInspector
from opacus.utils import module_modification
from mpi4py import MPI
import random
import seal
from seal import ChooserEvaluator, \
	Ciphertext, \
	Decryptor, \
	Encryptor, \
	EncryptionParameters, \
	Evaluator, \
	IntegerEncoder, \
	FractionalEncoder, \
	KeyGenerator, \
	MemoryPoolHandle, \
	Plaintext, \
	SEALContext, \
	EvaluationKeys, \
	GaloisKeys, \
	PolyCRTBuilder, \
	ChooserEncoder, \
	ChooserEvaluator, \
	ChooserPoly

from itertools import islice, chain, repeat
def chunk_pad(it, size, padval=None):
    it = chain(iter(it), repeat(padval))
    return iter(lambda: tuple(islice(it, size)), (padval,) * size)

def send_enc():
    list_params = []
    for param_tensor in model.state_dict():
        list_params += model.state_dict()[param_tensor].flatten().tolist()
    length = len(list_params)
    list_params_int = [0]*length
    for h in range(length):
        # Convert float values to integer so that they can be encrypted
        # length_integer is the maximum number of digits in the integer representation.
        # The returned value min_prec is the min number of dec places before first non-zero digit in float value.
        list_params_int[h] = round(list_params[h],3)*1000
        #length_integer = 3
        #list_params_int[h], min_prec = conv_int(list_params[h], length_integer)
    slot_count = int(crtbuilder.slot_count())
    pod_iter = iter(list_params_int)
    pod_sliced = list(chunk_pad(pod_iter, slot_count, 0))  # partitions the vector pod_vec into chunks of size equal to the number of batching slots
    for h in range(len(pod_sliced)):
        pod_sliced[h] = [int(pod_sliced[h][j]) for j in range(len(pod_sliced[h]))]
        for j in range(len(pod_sliced[h])):
            if pod_sliced[h][j] < 0:
                pod_sliced[h][j] = parms.plain_modulus().value() + pod_sliced[h][j]
    comm.send(len(pod_sliced), dest = 0, tag = 1)
    for chunk in range(len(pod_sliced)):
        encrypted_vec = Ciphertext()
        plain_vector = Plaintext()
        crtbuilder.compose(list(pod_sliced[chunk]), plain_vector)
        encryptor.encrypt(plain_vector, encrypted_vec)
        comm.send(encrypted_vec, dest = 0, tag=chunk+2)


def recv_enc(idx_local):
    num_chunks = comm.recv(source=idx_local, tag=1)  # receive the number of chunks to be sent by the worker
    list_params_enc = []
    for chunk in range(num_chunks):
        list_params_enc += [comm.recv(source = idx_local, tag=chunk+2)]
    return list_params_enc

def average_weights_enc():
    list_sums = []
    for j in range(len(local_weights[0])):
        enc_sum = Ciphertext()
        list_columns = [local_weights[h][j] for h in range(len(local_weights))]  # length of list_columns is equal to the number of active workers
        evaluator.add_many(list_columns,enc_sum)
        list_sums.append(enc_sum)
    return list_sums   # element-wise sum of encrypted weight vectors received from active workers

def dec_recompose(enc_wts): # function to decrypt the aggregated weight vector received from parameter server
    dec_wts = []
    global_aggregate = {}
    chunks = len(enc_wts)
    for h in range(chunks):
        plain_agg = Plaintext()
        decryptor.decrypt(enc_wts[h], plain_agg)
        crtbuilder.decompose(plain_agg)
        dec_wts += [plain_agg.coeff_at(h) for h in range(plain_agg.coeff_count())]
    for h in range(len(dec_wts)):
        if dec_wts[h] > int(parms.plain_modulus().value() - 1) / 2:
            dec_wts[h] = dec_wts[h] - parms.plain_modulus().value()
    for h in range(len(dec_wts)):
        dec_wts[h] = float(dec_wts[h])/(1000*m)
    pos_start=0
    for param_tensor in flattened_lengths:
        pos_end = pos_start + flattened_lengths[param_tensor]
        global_aggregate.update({param_tensor:torch.tensor(dec_wts[pos_start:pos_end]).reshape(shapes[param_tensor])})
        pos_start=pos_end
    return global_aggregate


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nworkers = comm.Get_size()-1
    workers = list(range(1,nworkers+1))
    start_time = time.time()
    print("Ids of workers: ", str(workers))
    print("Number of workers: ", str(nworkers))
    ## Generate encryption parameters at worker 1 and share them with all workers.
    # Share context object with the parameter server, so that it can aggregate encrypted values.
    if rank==1:
        # Initialize the parameters used for FHE
        parms = EncryptionParameters()
        parms.set_poly_modulus("1x^4096 + 1")
        parms.set_coeff_modulus(seal.coeff_modulus_128(4096))  # 128-bit security
        parms.set_plain_modulus(40961)
        context = SEALContext(parms)
        comm.send(context, dest=0, tag=0)  # send the context to the parameter server
        print("Sent context to server")
        for i in workers[1:]:
            print("Sent context to worker ", str(i))
            comm.send(context, dest=i, tag=0)  # send the context to the other workers

        keygen = KeyGenerator(context)
        public_key = keygen.public_key()
        secret_key = keygen.secret_key()
        for i in workers[1:]:
            print("Sending keys to worker ", str(i))
            comm.send(public_key, dest=i, tag=1)  # send the public key to worker i
            comm.send(secret_key, dest=i, tag=2)  # send the secret key to worker i

        encryptor = Encryptor(context, public_key)
        decryptor = Decryptor(context, secret_key)
        # Batching is done through an instance of the PolyCRTBuilder class so need
        # to start by constructing one.
        crtbuilder = PolyCRTBuilder(context)

    elif rank==0:  # Parameter server
        context = comm.recv(source=1, tag=0)  # parameter server receives only the context from worker 1
        evaluator = Evaluator(context)

    else: # Workers 2:num_users
        # The workers receive the context, public key, and secret key from worker 1
        context = comm.recv(source=1, tag=0)
        parms = context.parms()
        public_key = comm.recv(source=1, tag=1)
        secret_key = comm.recv(source=1,tag=2)
        encryptor = Encryptor(context, public_key)
        decryptor = Decryptor(context, secret_key)
        # Batching is done through an instance of the PolyCRTBuilder class so need
        # to start by constructing one.
        crtbuilder = PolyCRTBuilder(context)

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'
    m = max(int(args.frac * args.num_users), 1)  # Number of active workers in each round
    # load datasets in each user. Note that the datasets are not loaded in the parameter server.
    if rank >= 1:
        train_dataset, user_groups = get_train_dataset(args,rank)  # user_groups consists of the ids of the data samples that belong to current worker

    if rank == 0:
        # BUILD MODEL
        test_dataset = get_test_dataset(args)
        if args.model == 'cnn':
            # Convolutional neural network
            if args.dataset == 'mnist':
                global_model = CNNMnist(args=args)
            elif args.dataset == 'fmnist':
                global_model = CNNFashion_Mnist(args=args)
            elif args.dataset == 'cifar10' or args.dataset == 'cifar100':
                global_model = CNNCifar(args=args)
        else:
            exit('Error: unrecognized model')

        ### DPSGD OPACUS ###
        if args.withDP:
            try:
                inspector = DPModelInspector()
                inspector.validate(global_model)
                print("Model's already Valid!\n")
            except:
                global_model = module_modification.convert_batchnorm_modules(global_model)
                inspector = DPModelInspector()
                print(f"Is the model valid? {inspector.validate(global_model)}")
                print("Model is convereted to be Valid!\n")

        u_steps = np.zeros(args.num_users)
        ###########

        # Set the model to train and send it to device.
        global_model.to(device)
        global_model.train()
        print(global_model)

        # copy weights
        global_weights = global_model.state_dict()

        # Training
        val_acc_list, net_list = [], []
        cv_loss, cv_acc = [], []
        print_every = 10
        val_loss_pre, counter = 0, 0

        for idx in range(1, args.num_users + 1):  # this loop sends the global model to all the workers
            comm.send(global_model, dest=idx, tag=idx)

        for epoch in range(args.epochs):
            local_weights, local_losses = [], []
            idxs_users = np.random.choice(range(1,args.num_users+1), m, replace=False)
            print("active users: ", str(idxs_users))
            for i in idxs_users:  # send the epoch values only to the active workers in the current round
                print("Sending epoch to active_user: ", str(i))
                comm.send(epoch, dest=i,tag=i)

            if (epoch > 0) and (epoch < args.epochs -1):
                for idx in idxs_users:  # this loop sends the encrypted global aggregate to all active workers of the current round
                    comm.send(global_weights, dest=idx, tag=idx)

            #The receives are decoupled from the sends, ELSE the code will send global model, then wait for updates, and only then
            #move to send the global model to the second worker. But we want the processing to be in parallel. Therefore, all the sends are
            #implemented in a loop, and after that all the receives are implemented in a loop
            for idx in idxs_users: # this loop receives the updates from the active workers
                u_steps[idx-1] = comm.recv(source=idx, tag=idx)
                print(idx)
                if epoch == args.epochs-1:
                    local_weights.append(comm.recv(source=idx, tag=idx))  # receive unencrypted model update parameters from worker i
                elif epoch < args.epochs-1:
                    local_weights.append(recv_enc(idx))  # receive encrypted model update parameters from worker i

            #In the last epoch, the weights are received unencrypted,
            #therefore, the weights are aggregated and the global model is updated
            print("Length of local weights: ", str(len(local_weights)))
            if epoch == args.epochs-1:
                # update global weights
                global_weights = average_weights(local_weights)  # Add the unencrypted weights received
                global_model.load_state_dict(global_weights)
                workers = list(range(1, nworkers + 1))
                print("Workers: ", str(workers))
                print("Active users in last round: ", str(idxs_users))
                for wkr in idxs_users:
                    workers.remove(wkr)
                print("Residue workers: ", str(workers)) # Printing the ids of workers which are still listening for next round's communication.
                for i in workers:
                    print("Sending exit signal to residue worker: ", str(i))
                    comm.send(-1, dest=i, tag=i)
                break  # break out of the epoch loop.
            elif epoch < args.epochs-1:
                # Add the encrypted weights
                global_weights = average_weights_enc()  # Add the encrypted weights received

    elif rank >= 1:
        local_model = LocalUpdate(args=args, dataset=train_dataset,
                                  u_id=rank,
                                  idxs=user_groups, logger=logger)
        u_step=0
        model = comm.recv(source=0, tag=rank)  # global model is received from the parameter server
        print("Worker ", str(rank), " received global model")
        flattened_lengths = {param_tensor: model.state_dict()[param_tensor].numel() for param_tensor in
                             model.state_dict()}
        shapes = {param_tensor: list(model.state_dict()[param_tensor].size()) for param_tensor in
                  model.state_dict()}  # dictionary of shapes of tensors

        while True:
            epoch = comm.recv(source=0, tag=rank) # receive the epoch/communication round that the parameter server is in
            if epoch == -1:
                break
            #The server sends the latest weight aggregate to this worker,
            #based on which the local model is updated before starting to train.
            if (epoch < args.epochs-1) and (epoch > 0):
                enc_global_aggregate = comm.recv(source=0, tag=rank)
                ## decrypt and recompose enc_global_aggregate
                global_aggregate = dec_recompose(enc_global_aggregate)
                model.load_state_dict(global_aggregate)
            u_step += 1
            # Now perform one iteration
            w, loss, u_step = local_model.update_weights(model=model, global_round=epoch,u_step=u_step)
            comm.send(u_step, dest=0, tag=rank)  # send the step number
            if epoch < args.epochs-1:
                send_enc()  # send encrypted model update parameters to the global agent
            elif epoch == args.epochs-1:
                comm.send(w, dest=0, tag = rank)  # send unencrypted model update parameters to the global agent
                break

    if rank==0:
        # Test inference after completion of training
        test_acc, test_loss = test_inference(args, global_model, test_dataset)

        print(f' \n Results after {args.epochs} global rounds of training:')
        print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

        print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))




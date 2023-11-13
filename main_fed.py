#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import time

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img

if __name__ == '__main__':
    # parse args
    args = args_parser()
    print(f"active client number = {int(args.frac * args.num_users)}")
    print(f"avg round = {args.epochs}")
    print(f"local epoch = {args.local_ep}")
    print(f"local batch size = {args.local_bs}")
    print(f"dl model = {args.model}")
    print(f"dataset = {args.dataset}")
    print(f"iid = {args.iid}")
    print(f"all_clients = {args.all_clients}")

    # args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = torch.device('cpu')
    print(f"Device: {args.device}")

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
            print(f"IID client sample MNIST")
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
            print(f"NON-IID client sample MNIST")
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
            print(f"IID client sample CIFAR10.")
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
        print(f"CNN model and CIFAR dataset")
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
        print(f"CNN model and MNIST dataset")
    elif args.model == 'mlp':
        print(f"MLP model and MNIST dataset")
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')

    total_params = sum(p.numel() for p in net_glob.parameters())
    print(net_glob)
    print(f"Total parameters: {total_params}")
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()
    print(w_glob)

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]

    max_training_times_per_round = []
    aggregation_times_per_round = []
    for iter in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        client_training_times = []

        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])

            start_time = time.time()  # 记录训练开始时间
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            end_time = time.time()  # 记录训练结束时间

            local_training_time = end_time - start_time
            print(f'Round {iter+1}: Client {idx} training time = {local_training_time:.8f} seconds')
            client_training_times.append(local_training_time)

            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

        max_training_time = max(client_training_times)
        print(f'Round {iter+1}: Max training time across all clients = {max_training_time:.8f} seconds')

        # update global weights
        start_time = time.time()  # 记录平均开始时间
        w_glob = FedAvg(w_locals)
        end_time = time.time()  # 记录平均结束时间

        aggregation_time = end_time - start_time
        print(f'Round {iter+1}: aggregation time = {aggregation_time:.8f} seconds')
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        max_training_times_per_round.append(max_training_time)
        aggregation_times_per_round.append(aggregation_time)

    avg_max_training_time = np.mean(max_training_times_per_round)
    avg_aggregation_time = np.mean(aggregation_times_per_round)
    print(f'Average Max Training Time per Round = {avg_max_training_time:.8f} seconds')
    print(f'Average Aggregation Time per Round = {avg_aggregation_time:.8f} seconds')
    #     # print loss
    #     loss_avg = sum(loss_locals) / len(loss_locals)
    #     print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
    #     loss_train.append(loss_avg)
    #
    # # plot loss curve
    # plt.figure()
    # plt.plot(range(len(loss_train)), loss_train)
    # plt.ylabel('train_loss')
    # plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))
    #
    # # testing
    # net_glob.eval()
    # acc_train, loss_train = test_img(net_glob, dataset_train, args)
    # acc_test, loss_test = test_img(net_glob, dataset_test, args)
    # print("Training accuracy: {:.2f}".format(acc_train))
    # print("Testing accuracy: {:.2f}".format(acc_test))


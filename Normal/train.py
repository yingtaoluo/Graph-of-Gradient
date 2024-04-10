'''add the root path here'''
import sys
sys.path.append(r"..")

import os
os.chdir("D:/PycharmProjects/Causal")

'''add packages here'''
import time
import joblib
import argparse
from statistics import mean
import pdb
import torch.nn as nn
from torch import optim
import torch.utils.data as data
from tqdm import tqdm
from utils import *

# N: batch_size, U: number of unique ICDs, T: number of timestamps, H: embedding dimension, B: size of bucket
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='lstm', help='choose which model to train')
parser.add_argument('--seed', type=int, default='42', help='choose which dataset to load')
parser.add_argument('--gpu', type=int, default='0', help='choose which gpu device to use')
parser.add_argument('--lr', type=float, default='1e-2', help='learning rate')
parser.add_argument('--data', type=str, default='mimic3', help='choose a dataset, MIMIC3 or CMS')

aux_args = parser.add_argument_group('auxiliary')
aux_args.add_argument('--batch_size', type=int, help='choose a batch size')
aux_args.add_argument('--hidden_size', type=int, help='choose a hidden size')
aux_args.add_argument('--drop', type=float, help='choose a drop out rate')
aux_args.add_argument('--reg', type=float, help='choose a regularization coefficient')
parser.set_defaults(batch_size=64, hidden_size=16, drop=0, reg=0)

args = parser.parse_args()
torch.manual_seed(args.seed)
GPU = args.gpu
device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")
# print('train_device:{}'.format(device))
# print('train_gpu:{}'.format(GPU))
# python Normal/train.py dipole 0 0 1e-2 mimic3

# python Normal/train.py lstm 0 0 1e-2 mimic3
# python Normal/train.py lstm 1 1 1e-2 mimic4 --batch_size 64
# python Normal/train.py lstm 2 2 1e-2 mimic4 --batch_size 128
# python Normal/train.py lstm 0 3 1e-2 mimic4 --batch_size 256
# python Normal/train.py lstm 0 4 1e-2 mimic4 --batch_size 16
# python Normal/train.py stagenet 2 2 1e-2 mimic4 --batch_size 16


def calculate_hsic(dia_feat, pro_feat, weight, gpu):
    # [N, T, H] --> [N*T, H]
    batch_size, time_size, feat_size = dia_feat.shape
    dia_feat = dia_feat.reshape(batch_size * time_size, -1)
    pro_feat = pro_feat.reshape(batch_size * time_size, -1)

    # weighting
    weight = weight.reshape(batch_size * time_size, -1)
    dia_feat *= weight
    pro_feat *= weight

    kx = torch.unsqueeze(dia_feat, 1) - torch.unsqueeze(dia_feat, 2)
    Kx = torch.exp(- kx ** 2)
    ky = torch.unsqueeze(pro_feat, 1) - torch.unsqueeze(pro_feat, 2)
    Ky = torch.exp(- ky ** 2)

    Kxy = torch.matmul(Kx, Ky)
    hsic = torch.zeros((batch_size * time_size, )).to(gpu)
    for i, matrix in enumerate(Kxy):
        n = matrix.shape[0]
        h = torch.trace(matrix) / n ** 2 + torch.mean(Kx[i]) * torch.mean(Ky[i]) - 2 * torch.mean(matrix) / n
        hsic[i] = h * n**2 / (n - 1)**2

    return hsic.reshape(batch_size, time_size)


class DataSet(data.Dataset):
    def __init__(self, input1, input2, labels, length, ixs):
        # (NUM, T, U), (NUM, T, U), (NUM)
        self.input1, self.input2, self.labels, self.length, self.ixs = input1, input2, labels, length, ixs

    def __getitem__(self, index):
        input1 = self.input1[index].to(device)
        input2 = self.input2[index].to(device)
        labels = self.labels[index].to(device)
        length = self.length[index].to(device)
        ix = self.ixs[index].to(device)
        return input1, input2, labels, length, ix

    def __len__(self):
        return len(self.input1)


class Trainer:
    def __init__(self, model, all_data, record):
        # load other parameters
        self.model = model.to(device)
        self.records = record
        self.start_epoch = record['epoch'][-1] if load else 1
        self.batch_size = args.batch_size
        self.learning_rate = args.lr
        self.num_epoch = 300
        self.early_max = 20
        self.threshold = np.mean(record['acc_valid'][-1]) if load else 0  # 0 if not update

        [self.train_dia, self.valid_dia, self.test_dia,
         train_pro, valid_pro, test_pro,
         train_label, valid_label, test_label,
         train_len, valid_len, test_len,
         train_ixs, valid_ixs, self.test_ixs] = all_data

        max_times = int(len(self.train_dia) / self.batch_size) * self.batch_size
        self.train_dia, train_label = self.train_dia[0:max_times], train_label[0:max_times]

        self.train_dataset = DataSet(self.train_dia, train_pro, train_label, train_len, train_ixs)
        self.valid_dataset = DataSet(self.valid_dia, valid_pro, valid_label, valid_len, valid_ixs)
        self.test_dataset = DataSet(self.test_dia, test_pro, test_label, test_len, self.test_ixs)
        self.train_loader = data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = data.DataLoader(dataset=self.valid_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = data.DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def train(self):
        # set optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=args.reg)
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        best_acc, best_ndcg, epoch = 0, 0, 0

        for t in range(self.num_epoch):
            print('{} epoch ...'.format(t))
            # settings or validation and test
            train_size, loss_sum = 0, 0

            # training
            print('training ...')
            self.model.train()
            corr = []
            bar = tqdm(total=self.train_dia.shape[0])
            for step, item in enumerate(self.train_loader):
                # get batch data, (N, T, U), (N, T, U), (N)
                batch_dia, batch_pro, batch_label, batch_len, batch_ix = item
                batch_prob_dia = self.model(batch_dia, 'dia')  # , batch_feature_dia
                batch_prob_pro = self.model(batch_pro, 'pro')  # , batch_feature_pro
                batch_prob = batch_prob_dia + batch_prob_pro  # (N, T, U)
                loss_train = criterion(batch_prob, batch_label)
                loss_train.backward()
                loss_sum += loss_train.sum()
                train_size += to_npy(batch_len).sum()
                optimizer.step()
                optimizer.zero_grad()

                # ones = torch.nn.Parameter(torch.ones(size=(batch_dia.shape[0], self.train_dia.shape[1]))).to(device)
                # loss_cor = calculate_hsic(batch_feature_dia, batch_feature_pro, ones, device).mean()
                # corr.append(loss_cor.item())
                bar.update(batch_dia.shape[0])
            bar.close()

            print('training loss is: {}'.format(loss_sum/train_size))
            # print('training HSIC = {}'.format(mean(corr)))

            self.model.eval()

            '''
            # calculating training recall rates
            print('calculating training acc...')
            bar = tqdm(total=self.train_data.shape[0])
            for step, item in enumerate(self.train_loader):
                batch_input, batch_label, batch_len = item
                batch_prob = self.model(batch_input)  # (N, T, U)
                acc_train += calculate_acc(batch_prob, batch_label, batch_len)
                bar.update(batch_input.shape[0])
            bar.close()
            acc_train = np.array(acc_train) / train_size
            print('epoch:{}, train_acc:{}'.format(self.start_epoch + t, acc_train))
            '''

            # calculating validation recall rates
            print('calculating validation acc...')
            eth_list = list(dict(Counter(demo_array[:, n].tolist())).keys())[:m]
            eth_size = np.array([[0] * len(acc_rank)] * len(eth_list))
            eth_acc, eth_ndcg = [0] * len(eth_list), [0] * len(eth_list)

            bar = tqdm(total=self.valid_dia.shape[0])
            for step, item in enumerate(self.valid_loader):
                batch_dia, batch_pro, batch_label, batch_len, batch_ix = item
                batch_prob = self.model(batch_dia, 'dia') + self.model(batch_pro, 'pro')  # (N, T, U)

                for i in range(len(batch_prob)):
                    try:
                        eth_ix = eth_list.index(demo_array[:, n][batch_ix[i]])
                    except ValueError:
                        continue
                    eth_acc[eth_ix] += calculate_acc(batch_prob[i:i + 1], batch_label[i:i + 1], batch_len[i:i + 1])
                    eth_ndcg[eth_ix] += calculate_ndcg(batch_prob[i:i + 1], batch_label[i:i + 1], batch_len[i:i + 1])
                    eth_size[eth_ix] += batch_len[i:i + 1].item()

                bar.update(batch_dia.shape[0])
            bar.close()

            valid_size = eth_size[:, 0].sum()
            acc_valid = np.sum(np.array(eth_acc), 0) / valid_size
            ndcg_valid = np.sum(np.array(eth_ndcg), 0) / valid_size
            eth_acc = eth_acc / eth_size
            eth_ndcg = eth_ndcg / eth_size
            print('epoch:{}, valid_acc:{}, valid_ndcg:{}'.format(self.start_epoch + t, acc_valid, ndcg_valid))

            print(eth_list)
            print(eth_acc)
            print(eth_ndcg)

            # calculating testing recall rates
            print('calculating testing acc...')
            test_corr = []
            print('Number of each ethnicity in test set:{}'.format(Counter(demo_array[:, -1].tolist())))
            eth_size = np.array([[0] * len(acc_rank)] * len(eth_list))
            eth_acc, eth_ndcg = [0] * len(eth_list), [0] * len(eth_list)

            bar = tqdm(total=self.test_dia.shape[0])
            for step, item in enumerate(self.test_loader):
                batch_dia, batch_pro, batch_label, batch_len, batch_ix = item
                batch_prob = self.model(batch_dia, 'dia') + self.model(batch_pro, 'pro')  # (N, T, U)

                for i in range(len(batch_prob)):
                    try:
                        eth_ix = eth_list.index(demo_array[:, n][batch_ix[i]])
                    except ValueError:
                        continue
                    eth_acc[eth_ix] += calculate_acc(batch_prob[i:i + 1], batch_label[i:i + 1], batch_len[i:i + 1])
                    eth_ndcg[eth_ix] += calculate_ndcg(batch_prob[i:i + 1], batch_label[i:i + 1], batch_len[i:i + 1])
                    eth_size[eth_ix] += batch_len[i:i + 1].item()

                # ones = torch.nn.Parameter(torch.ones(size=(batch_dia.shape[0], self.train_dia.shape[1]))).to(device)
                # test_cor = calculate_hsic(batch_feature_dia, batch_feature_pro, ones, device).mean()
                # test_corr.append(test_cor.item())
                bar.update(batch_dia.shape[0])
            bar.close()

            test_size = eth_size[:, 0].sum()
            acc_test = np.sum(np.array(eth_acc), 0) / test_size
            ndcg_test = np.sum(np.array(eth_ndcg), 0) / test_size
            eth_acc = eth_acc / eth_size
            eth_ndcg = eth_ndcg / eth_size
            # group_test = hit / num
            print('epoch:{}, test_acc:{}, test_ndcg:{}'.format(self.start_epoch + t, acc_test, ndcg_test))
            # print('test HSIC = {}'.format(mean(test_corr)))

            print(eth_list)
            print(eth_acc)
            print(eth_ndcg)

            self.records['acc_valid'].append(acc_valid)
            self.records['acc_test'].append(acc_test)
            self.records['acc_eth'].append(eth_acc)
            # self.records['corr'].append(mean(test_corr))
            self.records['epoch'].append(self.start_epoch + t)

            if self.threshold < np.mean(np.min(eth_acc, 0)):  # np.mean(acc_valid)
                epoch = 0
                self.threshold = np.mean(np.min(eth_acc, 0))  # np.mean(acc_valid)
                best_acc, best_ndcg, best_eth_acc = acc_test, ndcg_test, eth_acc
                # save the model
                torch.save({'state_dict': self.model.state_dict(),
                            'records': self.records,
                            'time': time.time() - start},
                           'checkpoints/best_' + args.model + '_normal_' + args.data + '.pth')
            else:
                epoch += 1

            # if acc_valid does not increase, early stop it
            if self.early_max <= epoch:
                break

        print('Stop training Normal!')
        print('Final test_acc:{}, test_ndcg:{}'.format(best_acc, best_ndcg))
        print('{} test_ethnicity_acc:{}'.format('Normal', best_eth_acc))


if __name__ == '__main__':
    # load data and model
    if args.data == 'mimic3' or args.data == 'cms':
        file = open('./data/' + args.data + '/dataset.pkl', 'rb')
        file_data = joblib.load(file)
        [data_dia, data_pro, label_seq, real_len, demo_array] = file_data  # tensor (NUM, T, U)
    elif args.data == 'mimic4':
        data_dia = torch.tensor(np.load('./data/mimic4/data_dia.npy'))
        data_pro = torch.tensor(np.load('./data/mimic4/data_pro.npy'))
        label_seq = torch.tensor(np.load('./data/mimic4/label_dia.npy'))
        real_len = np.load('./data/mimic4/true_len.npy')
        demo_array = np.load('./data/mimic4/demo_array.npy')
    else:
        raise NotImplementedError

    n, m = -5, -2

    print('Number of each group:{}'.format(list(dict(Counter(demo_array[:, n].tolist())).keys())[:m]))

    # randomly divide train/dev/test datasets
    divide_ratio = (0.75, 0.1, 0.15)
    # divide_ratio = (0.4, 0.3, 0.3)
    data_combo = construct_data_combo(data_dia, data_pro, label_seq, real_len, divide_ratio)

    if args.model == 'lstm':
        import Normal.models.lstm
        model = Normal.models.lstm.LSTM(icd_size=data_dia.shape[2], pro_size=data_pro.shape[2],
                                        hidden_size=args.hidden_size,
                                        dropout=args.drop, batch_first=True)
    elif args.model == 'retain':
        import Normal.models.retain
        model = Normal.models.retain.RETAIN(icd_size=data_dia.shape[2], pro_size=data_pro.shape[2],
                                            hidden_size=args.hidden_size,
                                            dropout=args.drop, batch_first=True)
    elif args.model == 'dipole':
        import Normal.models.dipole
        model = Normal.models.dipole.Dipole(attention_type='location_based', icd_size=data_dia.shape[2],
                                            pro_size=data_pro.shape[2], attention_dim=args.hidden_size,
                                            hidden_size=args.hidden_size, dropout=args.drop,
                                            batch_first=True)

    elif args.model == 'stagenet':
        args.hidden_size = 384
        import Normal.models.stagenet
        model = Normal.models.stagenet.StageNet(icd_size=data_dia.shape[2], pro_size=data_pro.shape[2],
                                                hidden_size=args.hidden_size, dropout=args.drop)

    elif args.model == 'setor':
        import Normal.models.setor
        model = Normal.models.setor.SETOR(alpha=0.5, hidden_size=args.hidden_size,
                                          intermediate_size=args.hidden_size, hidden_act='relu',
                                          icd_size=data_dia.shape[2], pro_size=data_pro.shape[2],
                                          max_position=100, num_attention_heads=2, num_layers=1, dropout=args.drop)

    elif args.model == 'concare':
        import Normal.models.concare
        model = Normal.models.concare.ConCare(icd_size=data_dia.shape[2], pro_size=data_pro.shape[2],
                                              hidden_dim=args.hidden_size, MHD_num_head=4, drop=args.drop)

    else:
        raise NotImplementedError

    num_params = 0

    # for name in model.state_dict():
    #     print(name)

    for param in model.parameters():
        num_params += param.numel()
    print('num of params', num_params)

    load = False

    if load:
        checkpoint = torch.load('checkpoints/best_' + args.model + '_normal_' + args.data + '.pth')
        model.load_state_dict(checkpoint['state_dict'])
        start = time.time() - checkpoint['time']
        records = checkpoint['records']
    else:
        records = {'epoch': [], 'acc_valid': [], 'acc_test': [], 'acc_eth': [], 'corr': []}
        start = time.time()

    trainer = Trainer(model, data_combo, records)
    trainer.train()

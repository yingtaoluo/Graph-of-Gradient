'''add the root path here'''
import sys

import os
os.chdir("D:/PycharmProjects/Causal")

'''add packages here'''
import time
import joblib
import argparse
import pdb
from itertools import chain
import torch.nn as nn
from torch import optim
import torch.utils.data as data
from tqdm import tqdm
from utils import *
from Fair.models import *


# N: batch_size, U: number of unique ICDs, T: number of timestamps, H: embedding dimension, B: size of bucket
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='retain', help='choose which model to train')
parser.add_argument('--seed', type=int, default='42', help='choose which dataset to load')
parser.add_argument('--gpu', type=int, default='0', help='choose which gpu device to use')
parser.add_argument('--lr', type=float, default='3e-3', help='learning rate')
parser.add_argument('--data', type=str, default='mimic3', help='choose a dataset, MIMIC3 or CMS')

aux_args = parser.add_argument_group('auxiliary')
aux_args.add_argument('--batch_size', type=int, help='choose a batch size')
aux_args.add_argument('--hidden_size', type=int, help='choose a hidden size')
aux_args.add_argument('--drop', type=float, help='choose a drop out rate')
aux_args.add_argument('--reg', type=float, help='choose a regularization coefficient')
aux_args.add_argument('--k', type=int, help='choose the K for K-nearest neighbor algorithm')
parser.set_defaults(batch_size=64, hidden_size=16, drop=0, reg=0, K=3)

args = parser.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
GPU = args.gpu
device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")
# print('train_device:{}'.format(device))
# print('train_gpu:{}'.format(GPU))

# python Normal/train.py lstm 0 0 1e-2 mimic3
# python Normal/train.py lstm 1 1 1e-2 mimic4 --batch_size 64
# python Normal/train.py lstm 2 2 1e-2 mimic4 --batch_size 128
# python Normal/train.py lstm 0 3 1e-2 mimic4 --batch_size 256
# python Normal/train.py lstm 0 4 1e-2 mimic4 --batch_size 16
# python Normal/train.py stagenet 2 2 1e-2 mimic4 --batch_size 16


class Last_Layer(nn.Module):
    def __init__(self, hidden_size=args.hidden_size):
        super(Last_Layer, self).__init__()
        self.relu = nn.ReLU6()
        self.layer = nn.Linear(hidden_size*2, data_dia.shape[2])

    def forward(self, input_dia, input_pro):
        input_dia, input_pro = self.relu(input_dia), self.relu(input_pro)
        input_rep = torch.cat((input_dia, input_pro), dim=-1)
        input_rep.requires_grad_(True)
        input_rep.retain_grad()
        output = self.layer(input_rep)
        return output, input_rep


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
    def __init__(self, model, layer, adv, all_data, record):
        # load other parameters
        self.model = model.to(device)
        self.layer = layer.to(device)
        self.adversary = adv.to(device)
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

        # make sure that each batch contains full batch size of samples
        max_times = int(len(self.train_dia) / self.batch_size) * self.batch_size
        self.train_dia, train_label = self.train_dia[0:max_times], train_label[0:max_times]

        self.train_dataset = DataSet(self.train_dia, train_pro, train_label, train_len, train_ixs)
        self.valid_dataset = DataSet(self.valid_dia, valid_pro, valid_label, valid_len, valid_ixs)
        self.test_dataset = DataSet(self.test_dia, test_pro, test_label, test_len, self.test_ixs)
        self.train_loader = data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = data.DataLoader(dataset=self.valid_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = data.DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def cluster(self):
        embedding, ethnicity, weighting = [], [], []
        for i in range(int(len(data_dia)/self.batch_size)):
            batch_dia = torch.tensor(data_dia[i*self.batch_size:(i+1)*self.batch_size], dtype=torch.float32).to(device)
            batch_pro = torch.tensor(data_pro[i*self.batch_size:(i+1)*self.batch_size], dtype=torch.float32).to(device)
            weight, rnn_output = self.adversary(batch_dia, batch_pro)
            weighting.extend(to_npy(weight).tolist())
            embedding.extend(to_npy(rnn_output).tolist())
            ethnicity.extend(demo_array[i*self.batch_size:(i+1)*self.batch_size, -1].tolist())

        unique_ethnicity = list(set(ethnicity))
        encoded_ethnicity = [unique_ethnicity.index(item) for item in ethnicity]
        threshold = np.percentile(weighting, 95)
        pdb.set_trace()

        # Label the weights as 1 or 0 based on the threshold
        labels = [2 if weight >= threshold else 0 for weight in weighting]

        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt

        X = embedding
        y_true = encoded_ethnicity

        # Apply T-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=0)
        X_2d = tsne.fit_transform(X)

        # Visualize the Ground-Truth Clusters
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_true, s=30, cmap='viridis')
        plt.title("Ground Truth Clusters")

        # Visualize the Predicted Clusters
        plt.subplot(1, 2, 2)
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, s=30, marker='x', cmap='viridis')
        plt.title("Predicted Clusters")

        plt.show()

        pdb.set_trace()

    def train(self):
        # set optimizer
        optimizer = optim.Adam(chain(self.model.parameters(), self.layer.parameters()),
                               lr=self.learning_rate, weight_decay=args.reg)
        optimizer_adv = optim.Adam(self.adversary.parameters(), lr=self.learning_rate, weight_decay=args.reg)
        torch.autograd.set_detect_anomaly(True)
        criterion = nn.BCEWithLogitsLoss(reduction='none')

        best_acc, best_ndcg, epoch = 0, 0, 0

        for t in range(self.num_epoch):
            print('{} epoch ...'.format(t))
            # settings or validation and test
            train_size, loss_sum = 0, 0

            # training
            print('training ...')
            self.model.train()
            bar = tqdm(total=self.train_dia.shape[0])
            for step, item in enumerate(self.train_loader):
                # get batch data, (N, T, U), (N, T, U), (N)
                batch_dia, batch_pro, batch_label, batch_len, batch_ix = item
                batch_dia_prob, batch_pro_prob = self.model(batch_dia, 'dia'), self.model(batch_pro, 'pro')  # (N, T, U)
                batch_prob, last_representation = self.layer(batch_dia_prob, batch_pro_prob)

                loss_patients = torch.mean(torch.mean(criterion(batch_prob, batch_label), dim=-1), dim=-1).mean()
                loss_patients.backward(retain_graph=True)

                # Gradient of the input to the last layer
                grad_attn_output = last_representation.grad
                grad_flat = grad_attn_output.reshape(args.batch_size, -1)
                # torch.Size([64, 448])

                dists = torch.norm(grad_flat[:, None] - grad_flat, dim=2, p=2)
                values, indices = torch.topk(dists, k=args.K+1, largest=False, dim=1)
                # For example, [5, 2, 8] means that for patient index, the closest neighbors are 5, 2, and 8.
                neighbors_indices = indices[:, 1:].cpu().detach().numpy()  # Skip the 0th which is the dist to itself

                # Transform neighbors_indices into edge list format
                # Each pair (i, j) in the edge list means there is an edge from node i to node j
                edges = [(i, j) for i in range(neighbors_indices.shape[0]) for j in neighbors_indices[i]]
                # Convert edges into a tensor if needed for a GNN framework
                edges_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
                # edges_tensor is now in the shape [2, E] where E is the number of edges,
                # print("Edge List Tensor Shape:", edges_tensor.shape)

                weight = self.adversary(batch_dia, batch_pro, edges_tensor).squeeze()
                print('weight: {}'.format(weight))
                print('loss_patients: {}'.format(loss_patients))

                last_representation.detach_()
                self.layer.zero_grad()
                self.model.zero_grad()
                optimizer.zero_grad()

                loss_patients = torch.mean(torch.mean(criterion(batch_prob, batch_label), dim=-1), dim=-1)
                loss_train = (loss_patients * weight).mean()
                loss_train.backward(retain_graph=True)

                loss_sum += loss_patients.sum()
                train_size += to_npy(batch_len).sum()
                # optimizer.zero_grad()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                optimizer_adv.zero_grad()
                loss_patients = torch.mean(torch.mean(criterion(batch_prob.detach(), batch_label), dim=-1), dim=-1)
                adversary_loss = -(loss_patients * weight).mean()
                adversary_loss.backward()
                # optimizer_adv.zero_grad()
                optimizer_adv.step()

                bar.update(batch_dia.shape[0])
            bar.close()

            print('training loss is: {}'.format(loss_sum/train_size))

            self.model.eval()
            self.layer.eval()

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
                batch_dia_prob, batch_pro_prob = self.model(batch_dia, 'dia'), self.model(batch_pro, 'pro')  # (N, T, U)
                batch_prob, last_representation = self.layer(batch_dia_prob, batch_pro_prob)

                for i in range(len(batch_prob)):
                    try:
                        eth_ix = eth_list.index(demo_array[:, n][batch_ix[i].item()])
                    except ValueError:
                        continue  # if it is others, just skip and do not summarize
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
            # pdb.set_trace()

            # calculating testing recall rates
            print('calculating testing acc...')
            eth_size = np.array([[0] * len(acc_rank)] * len(eth_list))
            eth_acc, eth_ndcg = [0] * len(eth_list), [0] * len(eth_list)

            bar = tqdm(total=self.test_dia.shape[0])
            for step, item in enumerate(self.test_loader):
                batch_dia, batch_pro, batch_label, batch_len, batch_ix = item
                batch_dia_prob, batch_pro_prob = self.model(batch_dia, 'dia'), self.model(batch_pro, 'pro')  # (N, T, U)
                batch_prob, last_representation = self.layer(batch_dia_prob, batch_pro_prob)

                for i in range(len(batch_prob)):
                    try:
                        eth_ix = eth_list.index(demo_array[:, n][batch_ix[i].item()])
                    except ValueError:
                        continue
                    eth_acc[eth_ix] += calculate_acc(batch_prob[i:i+1], batch_label[i:i+1], batch_len[i:i+1])
                    eth_ndcg[eth_ix] += calculate_ndcg(batch_prob[i:i + 1], batch_label[i:i + 1], batch_len[i:i + 1])
                    eth_size[eth_ix] += batch_len[i:i+1].item()

                bar.update(batch_dia.shape[0])
            bar.close()

            test_size = eth_size[:, 0].sum()
            acc_test = np.sum(np.array(eth_acc), 0) / test_size
            ndcg_test = np.sum(np.array(eth_ndcg), 0) / test_size
            eth_acc = eth_acc / eth_size
            eth_ndcg = eth_ndcg / eth_size
            # group_test = hit / num
            print('epoch:{}, test_acc:{}, test_ndcg:{}'.format(self.start_epoch + t, acc_test, ndcg_test))

            print(eth_acc)
            print(eth_ndcg)

            self.records['acc_valid'].append(acc_valid)
            self.records['acc_test'].append(acc_test)
            self.records['acc_eth'].append(eth_acc)
            self.records['epoch'].append(self.start_epoch + t)

            if self.threshold < np.mean(np.min(eth_acc, 0)):  # np.mean(acc_valid)
                epoch = 0
                self.threshold = np.mean(np.min(eth_acc, 0))  # np.mean(acc_valid)
                best_acc, best_ndcg, best_eth_acc = acc_test, ndcg_test, eth_acc
                # save the model
                torch.save({'model_state_dict': self.model.state_dict(),
                            'layer_state_dict': self.layer.state_dict(),
                            'records': self.records,
                            'time': time.time() - start},
                           'checkpoints/best_' + args.model + '_fair_' + args.data + '.pth')
            else:
                epoch += 1

            # if acc_valid does not increase, early stop it
            if self.early_max <= epoch:
                break

        print('Stop training Fair!')
        print('Final test_acc:{}, test_ndcg:{},'.format(best_acc, best_ndcg))
        print('{} test_ethnicity_acc:{}'.format('Fair', best_eth_acc))


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

    n, m = -1, -2

    print('Number of each group {}:{}'.format(list(dict(Counter(demo_array[:, n].tolist())).keys())[:m],
                                              list(dict(Counter(demo_array[:, n].tolist())).values())[:m]))

    # randomly divide train/dev/test datasets
    divide_ratio = (0.75, 0.1, 0.15)
    # divide_ratio = (0.4, 0.3, 0.3)
    data_combo = construct_data_combo(data_dia, data_pro, label_seq, real_len, divide_ratio)

    if args.model == 'lstm':
        model = LSTM(icd_size=data_dia.shape[2], pro_size=data_pro.shape[2],
                           hidden_size=args.hidden_size,
                           dropout=args.drop, batch_first=True)
    elif args.model == 'retain':
        model = RETAIN(icd_size=data_dia.shape[2], pro_size=data_pro.shape[2],
                                            hidden_size=args.hidden_size,
                                            dropout=args.drop, batch_first=True)
    elif args.model == 'dipole':
        model = Dipole(attention_type='location_based', icd_size=data_dia.shape[2],
                                            pro_size=data_pro.shape[2], attention_dim=args.hidden_size,
                                            hidden_size=args.hidden_size, dropout=args.drop,
                                            batch_first=True)
    elif args.model == 'stagenet':
        args.hidden_size = 384
        model = StageNet(icd_size=data_dia.shape[2], pro_size=data_pro.shape[2],
                                                hidden_size=args.hidden_size, dropout=args.drop)
    elif args.model == 'setor':
        model = SETOR(alpha=0.5, hidden_size=args.hidden_size,
                                          intermediate_size=args.hidden_size, hidden_act='relu',
                                          icd_size=data_dia.shape[2], pro_size=data_pro.shape[2],
                                          max_position=100, num_attention_heads=2, num_layers=1, dropout=args.drop)
    elif args.model == 'concare':
        model = ConCare(icd_size=data_dia.shape[2], pro_size=data_pro.shape[2],
                                              hidden_dim=args.hidden_size, MHD_num_head=4, drop=args.drop)
    else:
        raise NotImplementedError

    num_params = 0

    # for name in model.state_dict():
    #     print(name)

    for param in model.parameters():
        num_params += param.numel()
    print('num of params', num_params)

    layer = Last_Layer()

    load = False

    adversary_model = Adversary(icd_size=data_dia.shape[2], pro_size=data_pro.shape[2], length=data_dia.shape[1])

    if load:
        checkpoint = torch.load('checkpoints/best_' + args.model + '_fair_' + args.data + '.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        layer.load_state_dict(checkpoint['layer_state_dict'])
        start = time.time() - checkpoint['time']
        records = checkpoint['records']
    else:
        records = {'epoch': [], 'acc_valid': [], 'acc_test': [], 'acc_eth': []}
        start = time.time()

    trainer = Trainer(model, layer, adversary_model, data_combo, records)
    trainer.train()
    # trainer.cluster()

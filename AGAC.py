from __future__ import print_function, division
import argparse
from keras import backend as K
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.nn import Linear
from utils import load_graph, load_data_origin_data, load_data_origin_data1
from GNN import GCNII, GNNLayer
from evaluation import eva
from torch.utils.data import Dataset
from graph_attention_layer import GAE
from preprocess import getGraph
from calcu_graph import construct_graph_kmean

import time
import pandas as pd
import matplotlib.pyplot as plt

torch.set_num_threads(2)
seed = 666
import torch.backends.cudnn as cudnn

cudnn.deterministic = True
cudnn.benchmark = True
import random

random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)


# Loss functions
def mae(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))


def maie_class_loss(y_true, y_pred):
    loss_E = mae(y_true, y_pred)
    return loss_E


class MLP_L(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_L, self).__init__()
        self.wl = Linear(n_mlp, 5)

    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.wl(mlp_in)), dim=1)

        return weight_output


class MLP_1(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_1, self).__init__()
        self.w1 = Linear(n_mlp, 2)

    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.w1(mlp_in)), dim=1)

        return weight_output


class MLP_2(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_2, self).__init__()
        self.w2 = Linear(n_mlp, 2)

    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.w2(mlp_in)), dim=1)

        return weight_output


class MLP_3(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_3, self).__init__()
        self.w3 = Linear(n_mlp, 2)

    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.w3(mlp_in)), dim=1)

        return weight_output


class GraphGAC(nn.Module):

    def __init__(self,n_enc_1, n_enc_2, n_enc_3,dropout_rate, n_heads,
                n_input, n_z, n_clusters, v=1):
        super(GraphGAC, self).__init__()

        self.gac = GAE(
            n_feat=n_input, F1=n_enc_1, F2=n_enc_2, F3=n_enc_3, n_z=n_z, dropout=dropout_rate, n_heads=n_heads
        )

        self.agcn_0 = GNNLayer(n_input, n_enc_1)
        self.agcn_1 = GNNLayer(n_enc_1, n_enc_2)
        self.agcn_2 = GNNLayer(n_enc_2, n_enc_3)
        self.agcn_3 = GNNLayer(n_enc_3, n_z)
        self.agcn_z = GNNLayer(256, n_clusters)

        self.mlp = MLP_L(256)

        # attention on [Z_i || H_i]
        self.mlp1 = MLP_1(2 * n_enc_1)
        self.mlp2 = MLP_2(2 * n_enc_2)
        self.mlp3 = MLP_3(2 * n_enc_3)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def pretrain_ae(self, dataset, A):
        # train_loader = DataLoader(dataset, batch_size=args.pre_batch_size, shuffle=True)

        optimizer = Adam(self.gac.parameters(), lr=args.pre_lr)
        for epoch in range(args.pre_epoch):
            X = torch.Tensor(dataset.x).to(device)
            x_bar, h1, h2, h3, z = self.gac(X, A)  # X~，，，，H

            loss = F.mse_loss(x_bar, X)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    def forward(self, x, A, adj):
        # DNN Module
        # X = np.array(x)
        x_bar, h1, h2, h3, z = self.gac(x, A)

        x_array = list(np.shape(x))
        n_x = x_array[0]

        # # AGCN-H
        z1 = self.agcn_0(x, adj)
        # z2
        m1 = self.mlp1(torch.cat((h1, z1), 1))
        m1 = F.normalize(m1, p=2)
        m11 = torch.reshape(m1[:, 0], [n_x, 1])
        m12 = torch.reshape(m1[:, 1], [n_x, 1])
        m11_broadcast = m11.repeat(1, 128)
        m12_broadcast = m12.repeat(1, 128)
        z2 = self.agcn_1(m11_broadcast.mul(z1) + m12_broadcast.mul(h1), adj)
        # z3
        m2 = self.mlp2(torch.cat((h2, z2), 1))
        m2 = F.normalize(m2, p=2)
        m21 = torch.reshape(m2[:, 0], [n_x, 1])
        m22 = torch.reshape(m2[:, 1], [n_x, 1])
        m21_broadcast = m21.repeat(1, 64)
        m22_broadcast = m22.repeat(1, 64)
        z3 = self.agcn_2(m21_broadcast.mul(z2) + m22_broadcast.mul(h2), adj)
        # z4
        m3 = self.mlp3(torch.cat((h3, z3), 1))  # self.mlp3(h2)
        m3 = F.normalize(m3, p=2)
        m31 = torch.reshape(m3[:, 0], [n_x, 1])
        m32 = torch.reshape(m3[:, 1], [n_x, 1])
        m31_broadcast = m31.repeat(1, 32)
        m32_broadcast = m32.repeat(1, 32)
        z4 = self.agcn_3(m31_broadcast.mul(z3) + m32_broadcast.mul(h3), adj)


        # # AGCN-S
        u = self.mlp(torch.cat((z1, z2, z3, z4, z), 1))
        u = F.normalize(u, p=2)
        u0 = torch.reshape(u[:, 0], [n_x, 1])
        u1 = torch.reshape(u[:, 1], [n_x, 1])
        u2 = torch.reshape(u[:, 2], [n_x, 1])
        u3 = torch.reshape(u[:, 3], [n_x, 1])
        u4 = torch.reshape(u[:, 4], [n_x, 1])

        tile_u0 = u0.repeat(1, 128)
        tile_u1 = u1.repeat(1, 64)
        tile_u2 = u2.repeat(1, 32)
        tile_u3 = u3.repeat(1, 16)
        tile_u4 = u4.repeat(1, 16)

        net_output = torch.cat((tile_u0.mul(z1), tile_u1.mul(z2), tile_u2.mul(z3), tile_u3.mul(z4), tile_u4.mul(z)), 1)
        net_output = self.agcn_z(net_output, adj, active=False)
        predict = F.softmax(net_output, dim=1)

        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


class LoadDataset(Dataset):
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))


def train_graphscc(dataset, A):
    model = GraphGAC(n_enc_1=args.n_enc_1,
                     n_enc_2=args.n_enc_2,
                     n_enc_3=args.n_enc_3,
                     dropout_rate=dropout_rate,
                     n_heads=args.n_attn_heads,
                     n_input=args.n_input,
                     n_z=args.n_z,
                     n_clusters=args.n_clusters,
                     v=1.0).to(device)
    print(model)

    model.pretrain_ae(dataset, A)

    optimizer = Adam(model.parameters(), lr=args.lr)
    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y

    A1 = A.float()

    with torch.no_grad():
        xbar, _, _, z = model(data, A, A1)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20, random_state=666)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    y_pred_last = y_pred

    pae_acc, pae_nmi, pae_ari = eva(y, y_pred, 'pae', pp=False)
    print(':pae_acc {:.4f}'.format(pae_acc), ', pae_nmi {:.4f}'.format(pae_nmi), ', pae_ari {:.4f}'.format(pae_ari))

    features = z.data.cpu().numpy()
    # 利用KNN构造细胞图
    error_rate = construct_graph_kmean(args.name, features.copy(), y, y,
                                       load_type='csv', topk=args.k, method='ncos')
    adj = load_graph(args.name, k=args.k, n=dataset.x.shape[0])
    adj = adj.to(device)

    patient = 0
    series = False
    sil_logs = []
    final_pred = None
    max_sil = 0
    for epoch in range(args.train_epoch):
        if epoch % 1 == 0:
            # update_interval
            xbar, tmp_q, pred, z = model(data, A, adj)

            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)
            res1 = tmp_q.cpu().numpy().argmax(1)  # Q
            res2 = pred.data.cpu().numpy().argmax(1)  # Z
            res3 = p.data.cpu().numpy().argmax(1)  # P
            Q_acc, Q_nmi, Q_ari = eva(y, res1, str(epoch) + 'Q', pp=False)
            Z_acc, Z_nmi, Z_ari = eva(y, res2, str(epoch) + 'Z', pp=False)
            P_acc, P_nmi, p_ari = eva(y, res3, str(epoch) + 'P', pp=False)
            # G_acc, G_nmi, G_ari = eva(y, pred_label, str(epoch) + 'G', pp=False)
            print(epoch, ':Q_acc {:.5f}'.format(Q_acc), ', Q_nmi {:.5f}'.format(Q_nmi), ', Q_ari {:.5f}'.format(Q_ari))
            print(epoch, ':Z_acc {:.5f}'.format(Z_acc), ', Z_nmi {:.5f}'.format(Z_nmi), ', Z_ari {:.5f}'.format(Z_ari))
            print(epoch, ':P_acc {:.5f}'.format(P_acc), ', P_nmi {:.5f}'.format(P_nmi), ', p_ari {:.5f}'.format(p_ari))
            # print(epoch, ':G_acc {:.5f}'.format(G_acc), ', G_nmi {:.5f}'.format(G_nmi), ', G_ari {:.5f}'.format(G_ari))
            delta_label = np.sum(res2 != y_pred_last).astype(np.float32) / res2.shape[0]
            y_pred_last = res2
            if epoch > 0 and delta_label < 0.0001:
                if series:
                    patient += 1
                else:
                    patient = 0
                series = True
                if patient == 100:
                    print('Reached tolerance threshold. Stopping training.')
                    print("Z_acc: {}".format(Z_acc), "Z_nmi: {}".format(Z_nmi),
                          "Z_ari: {}".format(Z_ari))
                    break
            else:
                series = False

        x_bar, q, pred, _ = model(data, A, adj)
        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, data)

        loss = args.kl_loss * kl_loss + args.ce_loss * ce_loss + re_loss
        print(epoch, ':kl_loss {:.5f}'.format(kl_loss), ', ce_loss {:.5f}'.format(ce_loss),
              ', re_loss {:.5f}'.format(re_loss), ', total_loss {:.5f}'.format(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    Q_acc, Q_nmi, Q_ari = eva(y, res1, str(epoch) + 'Q', pp=False)
    Z_acc, Z_nmi, Z_ari = eva(y, res2, str(epoch) + 'Z', pp=False)
    P_acc, P_nmi, p_ari = eva(y, res3, str(epoch) + 'P', pp=False)
    pd.DataFrame(res2).to_csv('result/pred_' + args.name + '.csv', index=False)
    print(epoch, ':Q_acc {:.4f}'.format(Q_acc), ', Q_nmi {:.4f}'.format(Q_nmi), ', Q_ari {:.4f}'.format(Q_ari))
    print(epoch, ':Z_acc {:.4f}'.format(Z_acc), ', Z_nmi {:.4f}'.format(Z_nmi), ', Z_ari {:.4f}'.format(Z_ari))
    print(epoch, ':P_acc {:.4f}'.format(P_acc), ', P_nmi {:.4f}'.format(P_nmi), ', p_ari {:.4f}'.format(p_ari))
    print('predict_y ', res2)
    print(args)
    return Z_acc, Z_nmi, Z_ari


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='Chung')
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--pre_lr', type=float, default=1e-4)
    parser.add_argument('--n_clusters', default=5, type=int)
    parser.add_argument('--load_type', type=str, default='csv')
    parser.add_argument('--kl_loss', type=float, default=0.1)
    parser.add_argument('--ce_loss', type=float, default=0.01)
    parser.add_argument('--similar_method', type=str, default='ncos')
    parser.add_argument('--pre_batch_size', type=int, default=32)
    parser.add_argument('--pre_epoch', type=int, default=400)
    parser.add_argument('--train_epoch', type=int, default=8000)
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--n_enc_1', default=128, type=int, help='number of neurons in the 1-st layer of encoder')
    parser.add_argument('--n_enc_2', default=64, type=int, help='number of neurons in the 2-nd layer of encoder')
    parser.add_argument('--n_enc_3', default=32, type=int, help='number of neurons in the 1-st layer of encoder')
    parser.add_argument('--n_z', default=16, type=int, help='number of neurons in the 2-nd layer of encoder')
    parser.add_argument('--dropout_rate', default=0.4, type=float, help='dropout rate of neurons in autoencoder')
    parser.add_argument('--l2_reg', default=0, type=float, help='coefficient for L2 regularizition')
    parser.add_argument('--n_attn_heads', default=4, type=int, help='number of heads for attention')
    parser.add_argument('--method', default='NE', type=str, help='number of heads for attention')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    # torch.cuda.set_device(args.device)
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    n_clusters = args.n_clusters
    if args.k == 1:
        dropout_rate = 0.  # To avoid absurd results
    else:
        dropout_rate = args.dropout_rate

    file_path1 = "data/" + args.name + "/data.tsv"
    file_path2 = "data/" + args.name + "/label.ann"
    dataset = load_data_origin_data1(file_path1, file_path2, args.load_type, scaling=True)

    GAT_autoencoder_path = 'logs/GATae_' + args.name + '.h5'

    print(args.name)
    print(dataset.x.shape)
    print(dataset.y.shape)
    np.seterr(divide='ignore', invalid='ignore')

    args.k = int(len(dataset.y) / 100)
    if args.k < 5:
        args.k = 5
    if args.k > 20:
        args.k = 20

    args.n_clusters = len(np.unique(dataset.y))
    args.n_input = dataset.x.shape[1]
    A = getGraph(args.name, dataset.x, 0, args.k, args.method)
    A = torch.tensor(A).to(device)
    train_graphscc(dataset, A)

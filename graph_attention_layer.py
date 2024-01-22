"""
The implementation follows https://github.com/danielegrattarola/keras-gat/tree/master/keras_gat,
which is released under MIT License.
"""

from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttention(nn.Module):
    def __init__(self, f_in, f_out,concat: bool, dropout:float, leaky_relu_slope:float=0.2):
        super(GraphAttention, self).__init__()

        self.concat = concat
        self.dropout = dropout
        self.w = nn.Parameter(torch.empty(size=(f_in, f_out)))
        self.a = nn.Parameter(torch.empty(size=(2*f_out, 1)))

        self.leakyrelu = nn.LeakyReLU(leaky_relu_slope)
        self.reset_parameters()

    # 初始化参数
    def reset_parameters(self):
        nn.init.xavier_normal_(self.w)
        nn.init.xavier_normal_(self.a)

    def forward(self, x, adj):
        N = x.size()[0]
        out_feature = self.w.size()[1]
        h = torch.mm(x, self.w)
        # h = F.dropout(h, self.dropout, training=self.training)
        # calculating the dot product of all node embeddings
        # and first half the attention vector parameters (corresponding to neighbor messages)
        source_scores = torch.matmul(h, self.a[:out_feature, :])

        # calculating the dot product of all node embeddings
        # and second half the attention vector parameters (corresponding to target node)
        target_scores = torch.matmul(h, self.a[out_feature:, :])

        # broadcast add
        e = source_scores + target_scores.T
        e = self.leakyrelu(e)
        connectivity_mask = -9e16 * torch.ones_like(e)
        # adj_mat is the N by N adjacency matrix
        # adj = adj.to_dense()  #将稀疏矩阵转换为稠密矩阵
        e = torch.where(adj > 0, e, connectivity_mask)  # masked attention scores

        # attention coefficients are computed as a softmax over the rows
        # for each column j in the attention score matrix e
        attention = F.softmax(e, dim=-1)

        # final node embeddings are computed as a weighted average of the features of its neighbors
        h_prime = torch.matmul(attention, h)

        return attention, h_prime  # 返回每层注意力

# class MultiHeadGraphAttention(nn.Module):
#     def __init__(self, n_head, f_in, f_out,concat: bool, dropout:float, leaky_relu_slope:float=0.2):
#         super(MultiHeadGraphAttention, self).__init__()
#
#         self.n_head = n_head
#         self.concat = concat
#         self.dropout = dropout
#         self.w = nn.Parameter(torch.randn(n_head, f_in, f_out))
#         self.a = nn.Parameter(torch.randn(n_head, f_out, 1))
#
#         self.leakyrelu = nn.LeakyReLU(leaky_relu_slope)
#         self.reset_parameters()
#
#     # 初始化参数
#     def reset_parameters(self):
#         gain = nn.init.calculate_gain('relu')
#         nn.init.xavier_uniform_(self.w, gain=gain)
#         nn.init.xavier_uniform_(self.a, gain=gain)
#
#     def forward(self, x, adj):
#         N = x.size()[0]
#         outputs = []
#         attns = []
#         for i in range(self.n_head):
#             h = torch.matmul(x, self.w[i])
#             # h = F.dropout(h, self.dropout, training=self.training)
#
#             attn_m = torch.matmul(h, self.a[i])
#             attn_m = attn_m.repeat(1, N)
#             attn_m = torch.cat([attn_m.view(-1, 1), attn_m.view(-1, 1)], dim=1)
#
#             attn_m = self.leakyrelu(attn_m)
#             # 这段代码主要是构造一个稀疏张量v,用于进行图注意力的计算。
#             # 具体来看:
#             # adj.nonzero()获取邻接矩阵adj中非零元素的索引坐标,即边的索引。
#             # t()对坐标进行转置,方便后续计算。
#             # attn_m.view(-1) 将attention矩阵展平为一维向量。
#             # attn_adj_m[0]和attn_adj_m[1]分别取出坐标的行列索引。
#             # 将行列索引转换为序号索引,用于在attn_m.view(-1)中取值。
#             # 将坐标attn_adj_m和对应的值组合起来,传给torch.sparse.FloatTensor。
#             # 设置稀疏矩阵v的大小为[N, N]。
#
#             attn_adj_m = adj.nonzero(as_tuple=False).t()
#             v = torch.sparse.FloatTensor(attn_adj_m, attn_m.view(-1)[attn_adj_m[0] * N + attn_adj_m[1]],
#                                          torch.Size([N, N]))
#
#             attention = self.leakyrelu(v.to_dense())
#             # attention = F.dropout(attention, self.dropout, training=self.training)
#
#             output = torch.matmul(attention, h)
#
#             outputs.append(output)
#             attns.append(attention)
#         if self.concat:
#             h = torch.cat(outputs, dim=-1)
#         else:
#             h = torch.mean(torch.stack(outputs), dim=0)
#             attention = torch.mean(torch.stack(attns), dim=0)
#
#         return attention, h  # 返回每层注意力


# GAT层堆叠
class GAE(nn.Module):
    def __init__(self, n_feat, F1, F2, F3, n_z, dropout, n_heads):
        super(GAE, self).__init__()
        self.dropout = dropout
        self.n_heads = n_heads

        self.layer1 = GraphAttention(f_in=n_feat, f_out=F1, concat=True, dropout=dropout, leaky_relu_slope=0.2)
        self.layer2 = GraphAttention(f_in=F1, f_out=F2, concat=True, dropout=dropout, leaky_relu_slope=0.2)
        self.layer3 = GraphAttention(f_in=F2, f_out=F3, concat=True, dropout=dropout, leaky_relu_slope=0.2)
        self.layer4 = GraphAttention(f_in=F3, f_out=n_z, concat=True, dropout=dropout, leaky_relu_slope=0.2)

        self.layer5 = GraphAttention(f_in=n_z, f_out=F3, concat=True, dropout=dropout, leaky_relu_slope=0.2)
        self.layer6 = GraphAttention(f_in=F3, f_out=F2, concat=True, dropout=dropout, leaky_relu_slope=0.2)
        self.layer7 = GraphAttention(f_in=F2, f_out=F1, concat=True, dropout=dropout, leaky_relu_slope=0.2)
        self.layer8 = GraphAttention(f_in=F1, f_out=n_feat, concat=False, dropout=dropout, leaky_relu_slope=0.2)

    def forward(self, x, adj):

        dropout1 = F.dropout(x, self.dropout, training=self.training)

        attn1, h1 = self.layer1(dropout1, adj)
        attn2, h2 = self.layer2(h1, adj)
        attn3, h3 = self.layer3(h2, adj)
        attn4, h4 = self.layer4(h3, adj)
        attn5, h5 = self.layer5(h4, adj)
        attn6, h6 = self.layer6(h5, adj)
        attn7, h7 = self.layer7(h6, adj)
        attn8, x_bar = self.layer8(h7, adj)

        return x_bar, h1, h2, h3, h4  # 返回重构的数据和特征嵌入

class GAE1(nn.Module):
    def __init__(self, n_feat, F1, F2, dropout, n_heads):
        super(GAE1, self).__init__()
        self.dropout = dropout
        self.n_heads = n_heads

        self.layer1 = GraphAttention(f_in=n_feat, f_out=F1, concat=True, dropout=dropout, leaky_relu_slope=0.2)
        self.layer2 = GraphAttention(f_in=F1, f_out=F2, concat=True, dropout=dropout, leaky_relu_slope=0.2)
        self.layer3 = GraphAttention(f_in=F2, f_out=F1, concat=True, dropout=dropout, leaky_relu_slope=0.2)
        self.layer4 = GraphAttention(f_in=F1, f_out=n_feat, concat=False, dropout=dropout, leaky_relu_slope=0.2)


    def forward(self, x, adj):

        dropout1 = F.dropout(x, self.dropout, training=self.training)

        attn1, h1 = self.layer1(dropout1, adj)
        attn2, h2 = self.layer2(h1, adj)
        attn3, h3 = self.layer3(h2, adj)
        attn4, x_bar = self.layer4(h3, adj)


        return x_bar, h2  # 返回重构的数据和特征嵌入

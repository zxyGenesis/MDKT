import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import Normalize

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        # Wh = torch.mm(adj.to(torch.float32).cuda(),Wh)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(adj.to(torch.float32).cuda()*attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=False) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj).unsqueeze(0) for att in self.attentions], dim=0)
        x = x.mean(dim=0)
        x = F.dropout(x, self.dropout, training=self.training)
        return x


class FSLModel(nn.Module):
    def __init__(self, n, reload):
        super(FSLModel, self).__init__()
        self.test_w = []
        self.Word_Vector = torch.FloatTensor(np.load('imagenet1360wordvec.npy')).cuda()
        self.Word_linear_All = nn.Sequential(nn.Linear(1000, 2048, bias=False),
                                             nn.ReLU())
        if reload == False:
            relation = np.load('ori_graph.npy')
        else:
            relation = np.load('graph1.npy')
        self.a = [2,1,1,1,1]
        self.n = n
        self.adj = torch.from_numpy(relation).to(torch.float32).cuda()
        self.W = nn.Parameter(torch.zeros(size=(1360, 2048)))
        self.gat1 = GAT(1000, 2048, 0.5, 0.2, 8)
        self.conv = nn.Conv1d(2, 1, kernel_size=1, stride=1)
        self.l2norm = Normalize(2)

    def forward(self, x):
        self.Word_Vector = self.l2norm(self.Word_Vector)
        t_trans = self.gat1(self.Word_Vector, self.adj)
        t_origin = self.Word_linear_All(self.Word_Vector)
        t_trans = t_trans.unsqueeze(1)
        t_origin = t_origin.unsqueeze(1)
        t_final = self.conv(torch.cat((t_origin, t_trans), 1)).squeeze(1)
        v_final = self.W + torch.mm(self.adj, self.W.clone().detach_())

        scores_v = torch.mm(x, v_final.t())
        scores_t = torch.mm(x, t_final.t())
        self.test_w = v_final + self.a[n-1]*t_final
        scores_all = torch.mm(x, self.test_w.t())
        return scores_v, scores_t, scores_all, t_final, v_final

    def test(self, x):
        scores = torch.mm(x, self.test_w.t())
        return scores

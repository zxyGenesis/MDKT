import torch
import torch.nn as nn
import json
import torch.nn.functional as F
import numpy as np
from datetime import datetime

def getTime():
    now = datetime.now()
    return now.strftime("%d%M%S")


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


def graph_generate(g,k):
    with open('ExperimentSplit/Json/train.json','r') as f:
        exp = json.load(f)
        base_classes = list(set(exp['image_labels']))

    with open('ExperimentSplit/Json/test.json','r') as f:
        exp = json.load(f)
        novel_classes = list(set(exp['image_labels']))

    # g = np.load('w-'+str(split)+'.npy')
    adj = np.zeros((1360,1360))

    def distribution_calibration(query, base_means,k):
        for i in range(len(novel_classes)):
            dist = []
            for j in range(len(base_means)):
                dist.append(np.linalg.norm(query[i]-base_means[j]))
            index = np.argsort(dist)[:k]
            dis = [dist[w] for w in index]
            # d = [2**(-di) for di in dis]
            d = [1/di if di!=0 else 1 for di in dis ]

            for v in range(len(index)):
                # adj[base_classes[index[v]], novel_classes[i]] = 2*d[v]
                adj[novel_classes[i], base_classes[index[v]]] = d[v]

    distribution_calibration(g[novel_classes],g[base_classes],k)
    sum = adj.sum(axis=1)
    sum = np.where(sum>0, sum, 1)
    sum = sum[:, np.newaxis]
    sum = np.repeat(sum, 1360, axis=1)
    adj = adj/sum
    return adj
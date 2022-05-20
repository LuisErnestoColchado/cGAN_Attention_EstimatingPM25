# ************************************************************************************************
# Author: Luis Ernesto Colchado Soncco
# Email: luis.colchado@ucsp.edu.pe
# Description: kNN Attention polling layer 
# Based on paper: https://arxiv.org/abs/1805.08905
# ************************************************************************************************

import numpy as np
import torch
import torch.nn as nn
from torch import dtype
from torch.autograd import Variable


class GraphAttentionLayer(nn.Module):
    r"""Attention layer

    Args:
        in_dim: int, dimension of input
        out_dim: int, dimension of output
        out_indices: torch.LongTensor, the indices of nodes whose representations are
                     to be computed
                     Default None, calculate all node representations
                     If not None, need to reset it every time model is run
        feature_subset: torch.LongTensor. Default None, use all features
        kernel: 'affine' (default), use affine function to calculate attention
                'gaussian', use weighted Gaussian kernel to calculate attention
        k: int, number of nearest-neighbors used for calculate node representation
           Default None, use all nodes
        graph: a list of torch.LongTensor, corresponding to the nearest neighbors of nodes
               whose representations are to be computed
               Make sure graph and out_indices are aligned properly
        use_previous_graph: only used when graph is None
                            if True, to calculate graph use input
                            otherwise, use newly transformed output
        nonlinearity_1: nn.Module, non-linear activations followed by linear layer
        nonlinearity_2: nn.Module, non-linear activations followed after attention operation

    Shape:
        - Input: (N, in_dim) graph node representations
        - Output: (N, out_dim) if out_indices is None
                  else (len(out_indices), out_dim)

    Attributes:
        weight: (out_dim, in_dim)
        a: out_dim if kernel is 'gaussian'
           out_dim*2 if kernel is 'affine'

    Examples:

        >>> m = GraphAttentionLayer(2,2,feature_subset=torch.LongTensor([0,1]),
                        graph=torch.LongTensor([[0,5,1], [3,4,6]]), out_indices=[0,1],
                        kernel='gaussian', nonlinearity_1=None, nonlinearity_2=None)
        >>> x = Variable(torch.randn(10,3))
        >>> m(x)
    """

    def __init__(self, in_dim, out_dim, k=None, graph=None, out_indices=None,
                 feature_subset=None, kernel='affine', nonlinearity_1=nn.Hardtanh(),
                 nonlinearity_2=None, use_previous_graph=True, reset_graph_every_forward=False,
                 no_feature_transformation=False, rescale=True, layer_norm=False, layer_magnitude=100,
                 key_dim=None, feature_selection_only=False):
        super(GraphAttentionLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.graph = graph
        if graph is None:
            self.cal_graph = True
        else:
            self.cal_graph = False
        self.use_previous_graph = use_previous_graph
        self.reset_graph_every_forward = reset_graph_every_forward
        self.no_feature_transformation = no_feature_transformation
        if self.no_feature_transformation:
            assert in_dim == out_dim
        else:
            self.weight = nn.Parameter(torch.Tensor(out_dim, in_dim))
            # initialize parameters
            std = 1. / np.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-std, std)

        self.rescale = rescale
        self.k = k
        self.out_indices = out_indices
        self.feature_subset = feature_subset
        self.kernel = kernel
        self.nonlinearity_1 = nonlinearity_1
        self.nonlinearity_2 = nonlinearity_2
        self.layer_norm = layer_norm
        self.layer_magnitude = layer_magnitude
        self.feature_selection_only = feature_selection_only

        if kernel == 'affine':
            self.a = nn.Parameter(torch.Tensor(out_dim * 2))
        elif kernel == 'gaussian' or kernel == 'inner-product' or kernel == 'avg_pool' or kernel == 'cosine':
            self.a = nn.Parameter(torch.Tensor(out_dim))
        elif kernel == 'key-value':
            if key_dim is None:
                self.key = None
                key_dim = out_dim
            else:
                if self.use_previous_graph:
                    self.key = nn.Linear(in_dim, key_dim)
                else:
                    self.key = nn.Linear(out_dim, key_dim)
            self.key_dim = key_dim
            self.a = nn.Parameter(torch.Tensor(out_dim))
        else:
            raise ValueError('kernel {0} is not supported'.format(kernel))
        value = 1. / len(self.a)
        #!self.a.data = torch.full()
        self.a.data = torch.full([len(self.a)], value)
        #!std = 1. / np.sqrt(self.weight.size(1))
        #!self.weight.data.uniform_(-std, std)
    def reset_graph(self, graph=None):
        self.graph = graph
        self.cal_graph = True if self.graph is None else False

    def reset_out_indices(self, out_indices=None):
        self.out_indices = out_indices

    def forward(self, x):
        # Split auxiliary features and pollutant feature
        pollutant = x[:, -1]
        x = x[:, :-1]
        #print(pollutant)
        if self.reset_graph_every_forward:
            self.reset_graph()
        
        N = x.size(0)
       
        out_indices = torch.range(start=0, end=N - 1).long() if self.out_indices is None else self.out_indices

        if self.feature_subset is not None:
            x = x[:, self.feature_subset]
        assert self.in_dim == x.size(1)

        if self.no_feature_transformation:
            out = x
        else:
            out = nn.functional.linear(x, self.weight)

        feature_weight = nn.functional.softmax(self.a, dim=0)
        if self.rescale and self.kernel != 'affine':
            out = out * feature_weight
            if self.feature_selection_only:
                return out

        if self.nonlinearity_1 is not None:
            out = self.nonlinearity_1(out)
        k = N - 1 if self.k is None else self.k  # min(self.k, out.size(0))

        if self.kernel == 'key-value':
            if self.key is None:
                keys = x if self.use_previous_graph else out
            else:
                keys = self.key(x) if self.use_previous_graph else self.key(out)
            norm = torch.norm(keys, p=2, dim=-1)
            att = (keys[out_indices].unsqueeze(-2) * keys.unsqueeze(-3)).sum(-1) / (
                    norm[out_indices].unsqueeze(-1) * norm)
            att_, idx = att.topk(k, -1)
            a = Variable(torch.zeros(att.size()).fill_(float('-inf')).type(dtype['float']))
            a.scatter_(-1, idx, att_)
            a = nn.functional.softmax(a, dim=-1)
            y = (a.unsqueeze(-1) * out.unsqueeze(-3)).sum(-2)
            if self.nonlinearity_2 is not None:
                y = self.nonlinearity_2(y)
            if self.layer_norm:
                y = nn.functional.relu(y)  # maybe redundant; just play safe
                y = y / y.sum(-1, keepdim=True) * self.layer_magnitude  # <UncheckAssumption> y.sum(-1) > 0
            return y

        # The following line is BUG: self.graph won't update after the first update
        # if self.graph is None
        # replaced with the following line
        if self.cal_graph:
            if self.kernel != 'key-value':
                features = x if self.use_previous_graph else out
                dist = torch.norm(features.unsqueeze(1) - features.unsqueeze(0), p=2, dim=-1)
                dist[dist == 0] = np.inf
                d, dist = dist.sort()
                # print(dist.shape)
                # Except neighbor dist 0 
                self.graph = dist[:, :-1]
                self.graph = self.graph[out_indices]
                
        y = Variable(torch.zeros(len(out_indices), 1 + (
                    self.out_dim * 2)).float())  # !!!(out.size(1)+1)*10 #out.size(1)+ #%%10+self.in_dim
        for i, idx in enumerate(out_indices):
            aux_data = out[idx]
            neighbor_idx = self.graph[i][:k]
            #s!print(k)
            # neighbor_idx = neighbor_idx[neighbor_idx > 0]
            if self.kernel == 'gaussian':
                if self.rescale:  # out has already been rescaled
                    a = -torch.sum((out[idx] - out[neighbor_idx]) ** 2, dim=1)
                else:
                    a = -torch.sum((feature_weight * (out[idx] - out[neighbor_idx])) ** 2, dim=1)
            elif self.kernel == 'inner-product':
                if self.rescale:  # out has already been rescaled
                    a = torch.sum(out[idx] * out[neighbor_idx], dim=1)
                else:
                    a = torch.sum(feature_weight * (out[idx] * out[neighbor_idx]), dim=1)
            elif self.kernel == 'cosine':
                if self.rescale:  # out has already been rescaled
                    norm = torch.norm(out[idx]) * torch.norm(out[neighbor_idx], p=2, dim=-1)
                    a = torch.sum(out[idx] * out[neighbor_idx], dim=1) / norm
                else:
                    norm = torch.norm(feature_weight * out[idx]) * torch.norm(feature_weight * out[neighbor_idx], p=2,
                                                                              dim=-1)
                    a = torch.sum(feature_weight * (out[idx] * out[neighbor_idx]), dim=1) / norm
            elif self.kernel == 'affine':
                a = torch.mv(torch.cat([(out[idx].unsqueeze(0)
                                         * Variable(torch.ones(len(neighbor_idx)).unsqueeze(1)).float()),
                                        out[neighbor_idx]], dim=1), self.a)
            elif self.kernel == 'avg_pool':
                a = Variable(torch.ones(len(neighbor_idx)).float())

            a = nn.functional.softmax(a, dim=0)

            x_pollutant = torch.cat([out[neighbor_idx], pollutant[neighbor_idx].unsqueeze(1)], dim=1)
            mean_data = torch.sum(x_pollutant * a.unsqueeze(1), dim=0)
            data = torch.cat([aux_data, mean_data])
            y[i] = data
        if self.nonlinearity_2 is not None:#-! and self.kernel=='affine':
            y = self.nonlinearity_2(y)
        if self.layer_norm:
            y = nn.functional.relu(y)  # maybe redundant; just play safe
            y = y / y.sum(-1, keepdim=True) * self.layer_magnitude  # <UncheckAssumption> y.sum(-1) > 0
        y_ = y
        return y, y_, a
##
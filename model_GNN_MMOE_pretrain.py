import copy

import numpy as np
import time
import pickle as pkl
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, top_k_accuracy_score, log_loss
from scipy.special import expit
import torch.nn.functional as F
import matplotlib.pyplot as plt
import dgl
import pandas as pd
from dgl.nn.pytorch import NNConv, Set2Set,RelGraphConv
from train_test_util import topk_accuracy_per_class, class_label
import torch
import torch.nn as nn
import math
import networkx as nx
def losscurve(train_losses):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.title('Training and Validation Loss')
    plt.ylabel('Loss')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.legend()
    plt.show()
import random
class GatedAggregator(nn.Module):
    """Gated aggregator for molecular graph features with adaptive thresholding."""
    def __init__(self, readout_feats):
        super(GatedAggregator, self).__init__()
        self.W_updates = nn.Linear(readout_feats, readout_feats)
        self.W_resets = nn.Linear(readout_feats, readout_feats)
        self.threshold = torch.tensor(0.1, dtype=torch.float32)
        
    def forward(self, chunks):
        """
        Aggregate molecular chunks using gated mechanism with L2 norm filtering.
        Args:
            chunks: tuple of tensors [num_mols x batch x embedding_size]
        """
        chunks_tensor = torch.stack(list(chunks), dim=0)
        norms = torch.norm(chunks_tensor, p=2, dim=(1, 2))
        mean_norm = norms.mean()
        if norms.size(0)==1:
            normalized_norms = torch.tensor(1.0,dtype=torch.bool,device=chunks_tensor.device)
        else:
            std_norm = norms.std()
            if std_norm == 0:
                std_norm = 1.0
            normalized_norms = (norms - mean_norm) / std_norm
        mask = (normalized_norms > -self.threshold).float().view(-1, 1, 1)
        masked_chunks = chunks_tensor * mask
        updates = torch.sigmoid(self.W_updates(masked_chunks))
        weighted_chunks = (1 + updates) * masked_chunks
        return weighted_chunks.sum(dim=0)
class GGRU(nn.Module):
    """Graph GRU with gating mechanism for node feature updates."""
    def __init__(self, hidden_dim):
        super(GGRU, self).__init__()
        self.f1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid())
        self.f2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid())
    
    def forward(self, nodefeats):
        theta=nodefeats*self.f2(nodefeats)*(1-self.f1(nodefeats))
        nx_graph = nx.random_regular_graph(d=0, n=nodefeats.shape[0])
        adj=torch.tensor(nx.adjacency_matrix(nx_graph).todense(),dtype=torch.float32,device=nodefeats.device)
        out=torch.matmul(adj,theta*self.f1(nodefeats))+theta
        return out

class GAU(nn.Module):
    """Gated Attention Unit with squared ReLU attention."""
    def __init__(
        self,
        dim,
        query_key_dim = 128,
        expansion_factor = 2.,
        add_residual = True,
        dropout = 0.,
    ):
        super().__init__()
        hidden_dim = int(expansion_factor * dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.to_hidden = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2),
            nn.SiLU()
        )
        self.to_qk = nn.Sequential(
            nn.Linear(dim, query_key_dim),
            nn.SiLU()
        )
        self.gamma = nn.Parameter(torch.ones(2, query_key_dim))
        self.beta = nn.Parameter(torch.zeros(2, query_key_dim))
        nn.init.normal_(self.gamma, std=0.02)
        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.add_residual = add_residual

    def forward(self, x):
        seq_len = x.shape[-2]
        normed_x = self.norm(x)
        v, gate = self.to_hidden(normed_x).chunk(2, dim = -1)
        Z = self.to_qk(normed_x)
        QK = torch.einsum('... d, h d -> ... h d', Z, self.gamma) + self.beta
        q, k = QK.unbind(dim=-2)
        sim = torch.einsum('b i d, b j d -> b i j', q, k) / seq_len
        A = F.relu(sim) ** 2
        A = self.dropout(A)
        V = torch.einsum('b i j, b j d -> b i d', A, v)
        V = V * gate
        out = self.to_out(V)
        if self.add_residual:
            out = out+x
        return out


import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from argparse import Namespace
from dgl.nn import GatedGraphConv


class MPNNLayer(nn.Module):
    """Message Passing Neural Network Layer for graph convolution."""
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super(MPNNLayer, self).__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.edge_update = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_dim)
        )
        self.node_update = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )

    def message_func(self, edges):
        concat_features = torch.cat([edges.src['h'], edges.data['e'], edges.dst['h']], dim=1)
        e_updated = self.edge_update(concat_features)
        return {'e_updated': e_updated}
    
    def reduce_func(self, nodes):
        e_sum = torch.sum(nodes.mailbox['e_updated'], dim=1)
        node_input = torch.cat([nodes.data['h'], e_sum], dim=1)
        h_updated = self.node_update(node_input)
        return {'h': h_updated}

    def forward(self, g, node_features, edge_features):
        g.ndata['h'] = node_features
        g.edata['e'] = edge_features
        g.update_all(self.message_func, self.reduce_func)
        return g.ndata['h'], g.edata['e']



class MPNN(nn.Module):
    """Message Passing Neural Network for molecular graph encoding."""
    def __init__(self, node_dim, edge_dim, hidden_dim = 1024, num_layers=3, num_step_set2set = 1, num_layer_set2set = 1,readout_feats = 1024):
        super(MPNN, self).__init__()
        self.layers = nn.ModuleList([
            MPNNLayer(node_dim, edge_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.readout = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.bn_n=nn.BatchNorm1d(node_dim)
        self.bn_e = nn.BatchNorm1d(edge_dim)
        self.gru = nn.GRU(node_dim, hidden_dim)
    
    def forward(self, g):
        node_features = g.ndata['node_attr']
        edge_features = g.edata['edge_attr']

        for layer in self.layers:
            node_features_l, edge_features_l = layer(g, node_features, edge_features)
            node_features=node_features+node_features_l
            edge_features=edge_features+edge_features_l
        feat, _ = self.gru(node_features.unsqueeze(0))
        feat=feat.squeeze(0)
        feat=torch.sigmoid(feat)*feat

        batch_num_nodes = g.batch_num_nodes()
        subgraph_indices = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(batch_num_nodes)])
        subgraph_indices = subgraph_indices.to(feat.device)
        sum_feats = torch.zeros(len(batch_num_nodes), feat.size(1), device=feat.device)
        sum_feats.index_add_(0, subgraph_indices, feat)
        return sum_feats

import math
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-3, elementwise_affine=True, memory_efficient=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'

from copy import deepcopy

class UserAttention(nn.Module):
    """Cross-attention mechanism between reaction and condition embeddings."""
    def __init__(self, user_dim, item_dim, hidden_dim,outdim):
        super(UserAttention, self).__init__()
        self.output = nn.Linear(user_dim, hidden_dim)
        self.Q=nn.Linear(user_dim, hidden_dim)
        self.K=nn.Linear(item_dim+outdim, hidden_dim)
        self.V=nn.Linear(item_dim+outdim, 2048)
        self.dropout = nn.Dropout(0.2)
        self.groupnorm=RMSNorm(2048)


    def attention(self, user_vec, item_vec):
        posi = torch.eye(item_vec.shape[0], device=user_vec.device)
        item_vec = torch.cat((item_vec, posi), 1)
        Q = self.Q(user_vec)
        K = self.K(item_vec)
        V = self.V(item_vec)
        T=2
        attention_scores = torch.matmul(Q, K.T) / (math.sqrt(K.shape[1])*T)
        attention_scores = nn.Softmax(dim=-1)(attention_scores)
        user_vec = torch.matmul(attention_scores, V)
        return user_vec

    def forward(self, user_vec, item_vec):
        user_vec=self.attention(user_vec, item_vec)+user_vec
        return user_vec



class TurAttention(nn.Module):
    """Differential attention mechanism with learnable lambda parameters."""
    def __init__(self, user_dim, item_dim, hidden_dim,outdim):
        super(TurAttention, self).__init__()
        self.userattr1=UserAttention(user_dim, item_dim, hidden_dim,outdim)
        self.userattr2=UserAttention(user_dim, item_dim, hidden_dim,outdim)
        self.userattr3=UserAttention(user_dim, item_dim, hidden_dim,outdim)
        self.userattr4=UserAttention(user_dim, item_dim, hidden_dim,outdim)
        self.line1=nn.Linear(hidden_dim,hidden_dim)
        self.groupnorm=RMSNorm(2048)
        self.lambda_q1 = nn.Parameter(torch.zeros(hidden_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(hidden_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(hidden_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(hidden_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.weights = nn.Parameter(torch.tensor(0.0), requires_grad=True)
    
    def forward(self, user_vec, item_vec):
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(user_vec)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(user_vec)
        lambda_init = 0.8 - 0.6 * math.exp(-6)
        lambda_full = lambda_1 - lambda_2 + lambda_init
        user_attr1=self.userattr1(user_vec, item_vec)
        user_attr2=self.userattr2(user_vec, item_vec)
        user_vec =self.groupnorm(user_attr1-user_attr2*lambda_full)*self.weights+user_vec
        return user_vec,item_vec



import torch
import torch.nn as nn
import torch.nn.functional as F





class MPNN_mix(nn.Module):
    """MPNN with attention-based mixing for reaction condition prediction."""
    def __init__(self, node_in_feats, edge_in_feats, n_classes,
                 readout_feats = 1024,
                 predict_hidden_feats = 512):
        super(MPNN_mix, self).__init__()
        self.mpnn = MPNN(node_in_feats, edge_in_feats)
        self.attr=TurAttention(readout_feats*2,readout_feats,512,n_classes)
        self.sum1=GatedAggregator(readout_feats)
        self.sum2 = GatedAggregator(readout_feats)
        self.sum3 = GatedAggregator(readout_feats)
        self.sum4 = GatedAggregator(readout_feats)
        self.predict = nn.Sequential(
            nn.Linear(readout_feats * 2, predict_hidden_feats), nn.PReLU(),
            nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(),
            nn.Linear(predict_hidden_feats, n_classes)
        )
    
    def forward(self, rmols, pmols,rmols2, pmols2,cmols,lam,rmol_max_cnt,pmol_max_cnt):
        r_graph_feats = self.mpnn(rmols)
        p_graph_feats = self.mpnn(pmols)
        batch_size =r_graph_feats.shape[0] // rmol_max_cnt
        pbatch_size = p_graph_feats.shape[0] // pmol_max_cnt

        r_graph_feats = self.sum1(torch.split(r_graph_feats, batch_size))
        p_graph_feats = self.sum1(torch.split(p_graph_feats, pbatch_size))

        cmols=[cmols]
        c_graph_feats = torch.sum(torch.stack([self.mpnn(mol) for mol in cmols]), 0)
        concat_feats = torch.cat([r_graph_feats, p_graph_feats], 1)
        out,_= self.attr(concat_feats, c_graph_feats)
        logit=self.predict(out)
        out = torch.sigmoid(logit)
        return out,logit




class reactionMPNN_mix(nn.Module):
    """Teacher model wrapper for reaction condition prediction."""
    def __init__(self, node_in_feats, edge_in_feats, n_classes,
                 readout_feats=1024,
                 predict_hidden_feats=512):
        super(reactionMPNN_mix, self).__init__()
        self.experts1 = MPNN_mix( node_in_feats, edge_in_feats, n_classes,readout_feats=1024,predict_hidden_feats=512)

    def forward(self, rmols, pmols, rmols2, pmols2, cmols, lam, rmol_max_cnt,pmol_max_cnt):
        out1,logit=self.experts1(rmols, pmols, rmols2, pmols2, cmols, lam, rmol_max_cnt,pmol_max_cnt)
        return out1,out1,logit

class baseMPNN(nn.Module):
    """Baseline MPNN model without attention mechanism."""
    def __init__(self, node_in_feats, edge_in_feats, n_classes,
                 readout_feats=1024,
                 predict_hidden_feats=512):
        super(baseMPNN, self).__init__()
        self.mpnn = MPNN(node_in_feats, edge_in_feats)
        self.sum1=GatedAggregator(readout_feats)
        self.sum2 = GatedAggregator(readout_feats)
        self.predict = nn.Sequential(
            nn.Linear(readout_feats * 2, predict_hidden_feats), nn.PReLU(),
            nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(),
            nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(),
            nn.Linear(predict_hidden_feats, n_classes)
        )
        self.itemtrans = nn.Sequential(
            nn.Linear(readout_feats, predict_hidden_feats), nn.PReLU(),
            nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(),
            nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(),
            nn.Linear(predict_hidden_feats, n_classes))
        self.bn = nn.BatchNorm1d(n_classes)
        self.attr = TurAttention(readout_feats * 2, readout_feats, 512, n_classes)

    def forward(self, rmols, pmols, rmols2, pmols2, cmols, lam, rmol_max_cnt,pmol_max_cnt):
        r_graph_feats = self.mpnn(rmols)
        p_graph_feats = self.mpnn(pmols)
        batch_size =r_graph_feats.shape[0] // rmol_max_cnt
        pbatch_size = p_graph_feats.shape[0] // pmol_max_cnt
        r_graph_feats = self.sum1(torch.split(r_graph_feats, batch_size))
        p_graph_feats = self.sum2(torch.split(p_graph_feats, pbatch_size))
        concat_feats = torch.cat([r_graph_feats, p_graph_feats], 1)
        out=self.predict(concat_feats)
        out =torch.sigmoid(out)
        return out,out,out
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss, jaccard_score, average_precision_score, ndcg_score, label_ranking_average_precision_score

def to_onehot(sequences, vector_length):
    onehot_encoded = []
    for seq in sequences:
        onehot_seq = np.zeros(vector_length)
        for index in seq:
            if index < vector_length:
                onehot_seq[index] = 1
        onehot_encoded.append(onehot_seq)
    return np.array(onehot_encoded)
def calculate(y_true_onehot, y_pred_onehot):
    precision = precision_score(y_true_onehot, y_pred_onehot, average='samples',zero_division=0)
    recall = recall_score(y_true_onehot, y_pred_onehot, average='samples',zero_division=0)
    ndcg = ndcg_score(y_true_onehot, y_pred_onehot)
    return precision,recall,ndcg
def test_metrics(y_true, y_pred):
    """Calculate precision, recall, and NDCG metrics for validation and test sets."""
    vector_length =239
    y_true_flattened = [sublist[0] for sublist in y_true]
    y_pred_flattened = [sublist[0] for sublist in y_pred]
    y_true_onehot = to_onehot(y_true_flattened, vector_length)
    y_pred_onehot = to_onehot(y_pred_flattened, vector_length)
    num_val=y_pred_onehot.shape[0]
    split_points = [int(num_val * 0.5)]
    y_pred_onehot_val, y_pred_onehot_test = np.split(y_pred_onehot, split_points)
    y_true_onehot_val, y_true_onehot_test = np.split(y_true_onehot, split_points )
    precision,recall,ndcg=calculate(y_true_onehot_val, y_pred_onehot_val)
    print("VAL.","Precision:", precision,"Recall:", recall,"NDCG:", ndcg)
    precision1,recall1,ndcg1=calculate(y_true_onehot_test, y_pred_onehot_test)
    print("TEST.","Precision:", precision1,"Recall:", recall1,"NDCG:", ndcg1)
    return precision,recall,ndcg

from sklearn.metrics import f1_score, recall_score, precision_score


class Trainer:
    """Trainer for teacher model with label mixing and co-occurrence regularization."""
    def __init__(self, net, n_classes, rmol_max_cnt, pmol_max_cnt, batch_size, model_path, cuda):
        self.net = net.to(cuda)
        self.n_classes = n_classes
        self.rmol_max_cnt = rmol_max_cnt
        self.pmol_max_cnt = pmol_max_cnt
        self.batch_size = batch_size
        self.model_path = model_path
        self.cuda = cuda
        print(self.cuda)
    
    def load(self):
        self.net.load_state_dict(torch.load(self.model_path))

    def CooccurrenceRegularizationLoss(self,input, cooccurrence_matrix, lambda_reg):
        """Calculate co-occurrence matrix regularization loss."""
        cooccurrence_matrix[cooccurrence_matrix>0]=1
        prob_diff_matrix = torch.sigmoid(input.unsqueeze(2) - input.unsqueeze(1))
        prob_diff_squared = torch.log1p(prob_diff_matrix)
        weighted_diff = prob_diff_squared * cooccurrence_matrix
        reg_loss = weighted_diff.sum(dim=(1, 2))
        reg_loss = reg_loss.mean() * lambda_reg
        return reg_loss

    def compute_loss(self, logit, y_true):
        """Calculate ranking loss by sampling negative samples for each positive sample."""
        batch_size, num_classes = logit.shape
        num_neg_samples=50
        index_pos = (y_true == 1).nonzero(as_tuple=False)
        index_neg = (y_true == 0).nonzero(as_tuple=False)
        N_pos = index_pos.size(0)
        N_neg = index_neg.size(0)
        sampled_neg_indices = torch.randint(0, N_neg, (N_pos * num_neg_samples,))
        cat_pos_neg = torch.cat([index_pos.repeat_interleave(num_neg_samples, dim=0),
                                 index_neg[sampled_neg_indices, 1].view(-1, 1)], dim=1)
        pos_logit = logit[cat_pos_neg[:, 0], cat_pos_neg[:, 1]]
        neg_logit = logit[cat_pos_neg[:, 0], cat_pos_neg[:, 2]]
        left = torch.cat((pos_logit, neg_logit), dim=0)
        right = torch.cat((neg_logit, pos_logit), dim=0)
        cmp_true = torch.cat((
            torch.ones(len(pos_logit), 1),
            torch.zeros(len(pos_logit), 1)
        ), dim=0).to(logit.device)
        diff = torch.sigmoid(left - right)
        pairwise_loss = F.binary_cross_entropy(diff, cmp_true.squeeze(), reduction='mean')
        return pairwise_loss
    def select_positive_negative_samples(self,logits, labels):
        """Select positive samples and random negative samples from logits."""
        positive_mask = labels == 1
        negative_mask = labels == 0
        positive_logits = []
        negative_logits = []
        batch_size = logits.shape[0]
        for i in range(batch_size):
            pos_indices = positive_mask[i].nonzero(as_tuple=True)[0]
            neg_indices = negative_mask[i].nonzero(as_tuple=True)[0]
            num_pos = pos_indices.size(0)
            if neg_indices.size(0) > num_pos:
                neg_indices = neg_indices[torch.randperm(neg_indices.size(0))[:num_pos]]
            positive_logits.append(logits[i, pos_indices])
            negative_logits.append(logits[i, neg_indices])
        positive_logits = torch.cat(positive_logits, dim=0)
        negative_logits = torch.cat(negative_logits, dim=0)
        right = torch.cat((negative_logits,positive_logits),dim=0)
        left = torch.cat((positive_logits,negative_logits),dim=0)
        cmp_true = torch.cat((torch.ones(len(positive_logits), 1), torch.zeros(len(positive_logits), 1)), dim=0).to(logits.device)
        torch.nn.functional.binary_cross_entropy(torch.sigmoid(left - right), cmp_true.squeeze(), reduction='mean')
        return left, right,cmp_true


    def training(self, train_loader, val_loader, rmol_max_cnt, pmol_max_cnt,max_epochs = 1000):

        category_to_index1 = {
             'solvent1':1,'solvent2':2,'catalyst1': 0
        }
        category_to_index2 = {'reagent1':3,'reagent2':4
        }
        y1=class_label(category_to_index1).to(self.cuda)
        y2=class_label(category_to_index2).to(self.cuda)

        train_losscur=[]
        print("Model path:",self.model_path)


        loss_fn = torch.nn.BCELoss(reduction = 'none')
        optimizer = torch.optim.Adam(self.net.parameters(), lr = 1e-3, weight_decay = 3e-5)
        scheduler = MultiStepLR(optimizer, milestones=[50,70,250,300], gamma=0.5)

        # lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,min_lr=1e-6, verbose=True)

        train_size = train_loader.dataset.__len__()
        val_y = val_loader.dataset.y
        val_log = np.zeros(max_epochs)
        with open('./data_dgl_item.pkl', 'rb') as f:
            self.cmol_graphs = pkl.load(f)
        inputs_cmol = self.cmol_graphs
        inputs_cmol = [b[0] for b in inputs_cmol[0]]
        inputs_cmol = dgl.batch(inputs_cmol).to(self.cuda)
        inputs_cmol.ndata['node_attr'] = inputs_cmol.ndata['node_attr'].float()
        inputs_cmol.edata['edge_attr'] = inputs_cmol.edata['edge_attr'].float()
        alp = 2

        for epoch in range(max_epochs):
            self.net.train()
            start_time = time.time()
            grad_norm_list = []
            total_loss = []
            ord_loss=[]
            Co_loss=[]
            train_loader_iter = iter(train_loader)
            for batchidx, batchdata in enumerate(train_loader):
                lam=np.random.beta(alp,alp)
                try:
                    batchdata2 = next(train_loader_iter)
                except StopIteration:
                    train_loader_iter = iter(train_loader)
                    batchdata2 = next(train_loader_iter)
                inputs_rmol = batchdata[0].to(self.cuda)
                inputs_pmol = batchdata[1].to(self.cuda)
                labels2 = batchdata2[-1].to(self.cuda)
                labels = batchdata[-1].to(self.cuda)

                preds,out1,logit = self.net(inputs_rmol, inputs_pmol,inputs_rmol,inputs_pmol,inputs_cmol,lam,rmol_max_cnt,pmol_max_cnt)
                labels=labels*lam+(1-lam)*labels2
                la1=labels*y1
                la2=labels*y2
                ooo= loss_fn(out1, la1).sum(axis=1).mean()+loss_fn(out1, la2).sum(axis=1).mean()
                loss_ord=ooo
                loss =loss_fn(preds,labels).sum(axis=1).mean()+loss_ord
                optimizer.zero_grad()
                loss.backward()
                assert not torch.isnan(loss)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1e3)
                grad_norm_list.append(grad_norm.cpu().numpy())
                optimizer.step()
                train_loss = loss.detach().item()
                train_loss_odr=loss_ord.detach().item()
                total_loss.append(train_loss)
                ord_loss.append(train_loss_odr)
            print('--- training epoch %d, lr %f, processed %d/%d, loss %.3f, ord_loss %.3f,time elapsed(min) %.2f'
                  %(epoch, optimizer.param_groups[-1]['lr'], train_size, train_size, np.mean(total_loss), np.mean(ord_loss),(time.time()-start_time)/60), np.max(grad_norm_list))
    
            start_time = time.time()
            topk=[6,10,20]
            val_y_preds,val_y_scores = self.inference(val_loader,inputs_cmol,topk,rmol_max_cnt, pmol_max_cnt)

            y_true_flattened = [sublist[0] for sublist in val_y]
            y_true_onehot = to_onehot(y_true_flattened, 239)
            val_y_scores, y_true_onehot=torch.tensor(val_y_scores), torch.tensor(y_true_onehot)
            num_val=val_y_scores.shape[0]
            val_y_scores,test_y_scores = torch.split(val_y_scores,  [int(num_val*0.5), num_val-int(num_val*0.5)])
            y_true_onehot,test_true_onehot=torch.split(y_true_onehot,  [int(num_val*0.5), num_val-int(num_val*0.5)])

            k=3
            topk_acc=topk_accuracy_per_class(val_y_scores,y_true_onehot,k)
            print("Validation set:",topk_acc)

            topk_acc1 = topk_accuracy_per_class(test_y_scores, test_true_onehot, 1)
            topk_acc3 = topk_accuracy_per_class(test_y_scores, test_true_onehot, 3)
            topk_acc5 = topk_accuracy_per_class(test_y_scores, test_true_onehot, 5)
            topk_result = pd.concat([topk_acc1.set_index('Class'),
                                topk_acc3.set_index('Class'),
                                topk_acc5.set_index('Class')], axis=1).reset_index()
            print("Test set:",topk_result)
            valpr=[]
            valrc=[]
            ndcgc=[]
            for i,pred in enumerate(val_y_preds):
                print("top",topk[i])
                prc,rec,ndcg=test_metrics(val_y, pred)
                valpr.append(prc)
                valrc.append(rec)
                ndcgc.append(ndcg)

            val_loss = 1-sum(topk_acc['Top3_Accuracy'])/4
            scheduler.step()
            val_log[epoch]=val_loss
            if np.argmin(val_log[:epoch + 1]) == epoch:
                torch.save(self.net.state_dict(),  self.model_path)
                print("Model saved to:", self.model_path)
        print('training terminated at epoch %d' %epoch)


    def topk_indices(self,tst_y_scores, k):
        return [[np.argsort(x)[-k:][::-1].tolist()] for x in tst_y_scores]

    def inference(self, tst_loader,c_mol,topk,rmol_max_cnt, pmol_max_cnt):
        self.net.eval()    
        tst_y_scores = []
        with torch.no_grad():
            for batchidx, batchdata in enumerate(tst_loader):
                inputs_rmol = batchdata[0].to(self.cuda)
                inputs_pmol = batchdata[1].to(self.cuda)
                preds_list,_,_ = self.net(inputs_rmol, inputs_pmol,inputs_rmol, inputs_pmol,c_mol,1,rmol_max_cnt, pmol_max_cnt)
                preds_list=preds_list.cpu().numpy()
                tst_y_scores.append(preds_list)
        tst_y_scores = (np.vstack(tst_y_scores))

        tst_y_preds=[]
        for k in topk:
            tst_y_preds.append(self.topk_indices(tst_y_scores,k))

        return tst_y_preds,tst_y_scores

import time
import pickle as pkl
from torch.optim.lr_scheduler import MultiStepLR
import pandas as pd
from train_test_util import topk_accuracy_per_class,class_label,test_metrics,to_onehot,score_dff_sample
import dgl
import math
import torch.nn as nn
import torch
import torch.nn.functional as F
import random
import numpy as np
from dgl.nn import Set2Set
from model_GNN_MMOE_pretrain import reactionMPNN_mix as teacher_model





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
        nodes.data['h'] =   nodes.data['h']
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
        
        # Aggregate atom-level features to molecule-level
        batch_num_nodes = g.batch_num_nodes()
        subgraph_indices = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(batch_num_nodes)])
        subgraph_indices = subgraph_indices.to(feat.device)
        sum_feats = torch.zeros(len(batch_num_nodes), feat.size(1), device=feat.device)
        sum_feats.index_add_(0, subgraph_indices, feat)
        return sum_feats

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
            nn.Linear(predict_hidden_feats, n_classes))
    
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
        concat_feats,_= self.attr(concat_feats, c_graph_feats)
        logit=self.predict(concat_feats)
        out = torch.sigmoid(logit)
        return out,logit,concat_feats




class reactionMPNN_mix(nn.Module):
    """Student model wrapper for reaction condition prediction."""
    def __init__(self, node_in_feats, edge_in_feats, n_classes,
                 readout_feats=1024,
                 predict_hidden_feats=512):
        super(reactionMPNN_mix, self).__init__()
        self.experts1 = MPNN_mix( node_in_feats, edge_in_feats, n_classes,readout_feats=1024,predict_hidden_feats=512)
    
    def forward(self, rmols, pmols, rmols2, pmols2, cmols, lam, rmol_max_cnt,pmol_max_cnt):
        out,logit,concat_feats=self.experts1(rmols, pmols, rmols2, pmols2, cmols, lam, rmol_max_cnt,pmol_max_cnt)
        return out,logit,concat_feats

class Trainer:
    """Trainer for student model with knowledge distillation from teacher model."""
    def __init__(self, net, n_classes, rmol_max_cnt, pmol_max_cnt, batch_size, model_path, cuda):
        self.net = net.to(cuda)
        self.teacher =  reactionMPNN_mix(169, 9, n_classes).to(cuda)
        self.n_classes = n_classes
        self.rmol_max_cnt = rmol_max_cnt
        self.pmol_max_cnt = pmol_max_cnt
        self.batch_size = batch_size
        self.model_path = model_path
        self.cuda = cuda
        self.distillation_weight = 1600  
        print(self.cuda)

    def CooccurrenceRegularizationLoss(self,input, cooccurrence_matrix, lambda_reg):
        """Calculate co-occurrence matrix regularization loss."""
        cooccurrence_matrix[cooccurrence_matrix>0]=1
        prob_diff_matrix = torch.sigmoid(-input.unsqueeze(2) + input.unsqueeze(1))
        weighted_diff = -torch.log(1-torch.clamp(prob_diff_matrix * cooccurrence_matrix,max=0.8))
        reg_loss = weighted_diff.sum(dim=(1, 2))
        reg_loss = reg_loss.mean() * lambda_reg
        return reg_loss

    def compute_loss(self, logit, y_true):
        """Calculate ranking loss by sampling negative samples for each positive sample."""
        num_neg_samples=100
        index_pos = (y_true == 1).nonzero(as_tuple=False)
        index_neg = (y_true == 0).nonzero(as_tuple=False)
        
        if index_pos.size(0) == 0 or index_neg.size(0) == 0:
            return torch.tensor(0.0, device=logit.device, requires_grad=True)
            
        N_pos = index_pos.size(0)
        N_neg = index_neg.size(0)
        
        sampled_neg_indices = torch.randint(0, N_neg, (N_pos * num_neg_samples,))
        
        try:
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
            diff = torch.clamp(diff, 1e-7, 1.0 - 1e-7)
            
            pairwise_loss = F.binary_cross_entropy(diff, cmp_true.squeeze(), reduction='mean')
            return pairwise_loss
        except Exception as e:
            print(f"Error computing rank loss: {e}")
            return torch.tensor(0.0, device=logit.device, requires_grad=True)

    def distillation_loss(self,student_probs, teacher_probs, temperature=2.0):
        """
        Calculate knowledge distillation loss for multi-label classification.
        Args:
            student_probs: Student model probability output [batch_size, num_classes]
            teacher_probs: Teacher model probability output [batch_size, num_classes]
            temperature: Temperature parameter for soft label smoothing
        """
        try:
            student_probs = torch.clamp(student_probs, 1e-7, 1.0 - 1e-7)
            teacher_probs = torch.clamp(teacher_probs, 1e-7, 1.0 - 1e-7)
            class_losses = -(teacher_probs * torch.log(student_probs) + 
                            (1 - teacher_probs) * torch.log(1 - student_probs))
            
            confidence_weights = 2.0 * torch.abs(teacher_probs - 0.5)
            weighted_losses = class_losses * confidence_weights
            loss = weighted_losses.mean()
            return loss
        except Exception as e:
            print(f"Error computing distillation loss: {e}")
            return torch.tensor(0.1, device=student_probs.device, requires_grad=True)

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
        print("Model path:",self.model_path)
        state_dict = torch.load('./model/model_attentionmix.pt')
        self.teacher.load_state_dict(state_dict)

        # Freeze teacher model parameters
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        loss_fn = torch.nn.BCELoss(reduction = 'none')
        optimizer = torch.optim.Adam(self.net.parameters(), lr = 1e-4, weight_decay = 3e-5)
        scheduler = MultiStepLR(optimizer, milestones=[100,150,300], gamma=0.5)

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
        co_occurrence_matrix=torch.load('data/co_occurrence_matrix.pt')
        for epoch in range(max_epochs):
            self.net.train()
            start_time = time.time()
            grad_norm_list = []
            total_loss = []
            ord_loss=[]
            Co_loss=[]
            category_to_index1 = {
                'solvent1': 1, 'solvent2': 2, 'catalyst1': 0
            }

            category_to_index2 = {'reagent1': 3, 'reagent2': 4
                                  }
            y1 = class_label(category_to_index1).to(self.cuda)
            y2 = class_label(category_to_index2).to(self.cuda)
            train_loader_iter = iter(train_loader)  # 创建 train_loader 的迭代器
            for batchidx, batchdata in enumerate(train_loader):
                lam=np.random.beta(alp,alp)
                inputs_rmol = batchdata[0].to(self.cuda)
                inputs_pmol = batchdata[1].to(self.cuda)
                labels = batchdata[-1].to(self.cuda)
                preds,logit,_ = self.net(inputs_rmol, inputs_pmol,inputs_rmol,inputs_pmol,inputs_cmol,lam,rmol_max_cnt,pmol_max_cnt)
                preds_tea,logit_tea,_ = self.teacher(inputs_rmol, inputs_pmol,inputs_rmol,inputs_pmol,inputs_cmol,lam,rmol_max_cnt,pmol_max_cnt)
                distillation=[]
                distillation_loss = self.distillation_loss(preds, preds_tea) * self.distillation_weight
                distillation.append(distillation_loss.item())
                co_loss=self.CooccurrenceRegularizationLoss(logit, co_occurrence_matrix,1e-3)
                rank_loss=self.compute_loss(logit,labels)

                loss =loss_fn(preds,labels).sum(axis=1).mean()+co_loss+rank_loss+distillation_loss

                optimizer.zero_grad()
                loss.backward()
                assert not torch.isnan(loss)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1e3)
                grad_norm_list.append(grad_norm.cpu().numpy())
                optimizer.step()
                train_loss = loss.detach().item()

                Co_loss.append(rank_loss.detach().item()+co_loss.detach().item())
                total_loss.append(train_loss)
            print('--- training epoch %d, lr %f, processed %d/%d, loss %.3f,co_loss %.3f,distillation_loss %.3f,time elapsed(min) %.2f'
                  %(epoch, optimizer.param_groups[-1]['lr'], train_size, train_size, np.mean(total_loss), np.mean(Co_loss),np.mean(distillation),(time.time()-start_time)/60), np.max(grad_norm_list))
    
            # validation
            topk=[6,10,20]
            val_y_preds,val_y_scores = self.inference(val_loader,inputs_cmol,topk,rmol_max_cnt, pmol_max_cnt)
            y_true_flattened = [sublist[0] for sublist in val_y]
            y_true_onehot = to_onehot(y_true_flattened, 231)
            val_y_scores, y_true_onehot=torch.tensor(val_y_scores), torch.tensor(y_true_onehot)
            num_val=val_y_scores.shape[0]
            val_y_scores,test_y_scores = torch.split(val_y_scores,  [int(num_val*0.5), num_val-int(num_val*0.5)])
            y_true_onehot,test_true_onehot=torch.split(y_true_onehot,  [int(num_val*0.5), num_val-int(num_val*0.5)])
            k=3
            topk_acc=topk_accuracy_per_class(val_y_scores,y_true_onehot,k)
            score_dff_sample(val_y_scores,y_true_onehot,co_occurrence_matrix)
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
                torch.save(self.net.state_dict(),  './model/model_%s_%s_%d_attentionmix_fine_jmc.pt')
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

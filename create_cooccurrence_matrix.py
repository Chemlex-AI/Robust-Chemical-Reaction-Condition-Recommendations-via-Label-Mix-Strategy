from torch.utils.data import DataLoader
from dgl.data.utils import split_dataset, Subset
from dataset import GraphDataset
import dgl
import numpy
from argparse import ArgumentParser
import os
import sys
import os
import pandas as pd
import numpy as np
from collections import defaultdict
import pickle
import torch
sys.stdout.flush()  # 确保输出立即刷新
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from tqdm import tqdm



def build_cooccurrence_matrix_vectorized(train_loader, num_items=231, device='cpu'):

    cooccurrence_matrix = torch.zeros((num_items, num_items), dtype=torch.long, device=device)

    for batchidx, batchdata in enumerate(train_loader):
        # labels: [batch_size, num_items]
        labels = batchdata[-1].to(device)
        binary_matrix = labels.float()
        batch_co = torch.matmul(binary_matrix.t(), binary_matrix)
        # 将对角线设置为0
        batch_co.fill_diagonal_(0)
        cooccurrence_matrix += batch_co.long()
    return cooccurrence_matrix
def save_co_occurrence_matrix(co_occurrence_matrix, filepath):
    torch.save(co_occurrence_matrix, filepath)
    print(f'共现矩阵已保存至 {filepath}')

# 3. 加载共现矩阵
def load_co_occurrence_matrix(filepath):
    if os.path.exists(filepath):
        print(f'从 {filepath} 加载共现矩阵')
        return torch.load(filepath)
    else:
        return None





if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--rtype', '-t', type=str, choices=['example', 'suzuki', 'cn', 'negishi', 'pkr'],
                        default='example')
    parser.add_argument('--method', '-m', type=str, choices=['Attention', 'AttentionMIX', 'MPNN', 'VAE'],
                        default='Attention')
    parser.add_argument('--iterid', '-i', type=int, default=2)
    parser.add_argument('--mode', '-o', type=str, choices=['trn', 'tst'], default='trn')
    args = parser.parse_args()

    rtype = args.rtype
    method = args.method
    iterid = args.iterid
    mode = args.mode

    import random
    import numpy as np
    import torch


    def collate_fn(rmol_max_cnt, pmol_max_cnt):
        def collate_reaction_graphs(batch):
            batchdata = list(map(list, zip(*batch)))
            gs = [dgl.batch(s) for s in batchdata[:-1]]
            # labels = torch.FloatTensor(batchdata[-1])
            labels = numpy.array(batchdata[-1])  # 先将列表转换为 numpy.ndarray
            labels = torch.FloatTensor(labels)  # 再将 numpy.ndarray 转换为 PyTorch 的 FloatTensor
            inputs_rmol = dgl.batch([b for b in gs[:rmol_max_cnt]])
            inputs_pmol = dgl.batch([b for b in gs[rmol_max_cnt:rmol_max_cnt + pmol_max_cnt]])
            return *(inputs_rmol, inputs_pmol), labels

        return collate_reaction_graphs


    if not os.path.exists('./model/'): os.makedirs('./model/')

    if method == 'Attention':
        from model_GNN_MMOE import reactionMPNN_mix as Model
        from model_GNN_MMOE import Trainer

        model_path = './model/model_%s_%s_%d_attention.pt' % (method, rtype, iterid)
        batch_size = 128
        use_rxnfp = False
    elif method == 'AttentionMIX':
        from model_GNN_MMOE import reactionMPNN_mix as Model
        from model_GNN_MMOE import Trainer

        model_path = './model/model_%s_%s_%d_attentionmix.pt' % (method, rtype, iterid)
        batch_size = 128
        use_rxnfp = False
    elif method == 'MPNN':
        from model_MPNN import baseMPNN as Model
        from model_MPNN import Trainer

        model_path = './model/model_%s_%s_%d_MPNN.pt' % (method, rtype, iterid)
        batch_size = 128
        use_rxnfp = False
    elif method == 'VAE':
        from VAE import VAE as Model
        from VAE import Trainer

        model_path = './model/model_%s_%s_%d_VAE.pt' % (method, rtype, iterid)
        batch_size = 128
        use_rxnfp = False

    random_state = 35

    cuda = torch.device('cuda:2')

    print("batchsize****************", batch_size)
    print("Using method!!!!!!!", method)
    trndata = GraphDataset(rtype, split='trn', use_rxnfp=use_rxnfp, seed=random_state)
    valdata = GraphDataset(rtype, split='tst', use_rxnfp=use_rxnfp, seed=random_state)

    rmol_max_cnt = trndata.rmol_max_cnt
    pmol_max_cnt = trndata.pmol_max_cnt

    trn_loader = DataLoader(dataset=trndata, batch_size=batch_size, shuffle=True,collate_fn=collate_fn(rmol_max_cnt, pmol_max_cnt), num_workers=1, drop_last=False)
    val_loader = DataLoader(dataset=valdata, batch_size=batch_size, shuffle=False,collate_fn=collate_fn(rmol_max_cnt, pmol_max_cnt), num_workers=1, drop_last=False)
    n_classes = trndata.n_classes
    # 构建共现矩阵
    print("构建共现矩阵")
    filepath='co_occurrence_matrix.pt'
    co_occurrence_matrix = load_co_occurrence_matrix(filepath)

    co_occurrence_matrix=None
    if co_occurrence_matrix is None:
        # 如果文件不存在，构建共现矩阵并保存
        co_occurrence_matrix = build_cooccurrence_matrix_vectorized(trn_loader, num_items=n_classes, device='cuda:2')
        save_co_occurrence_matrix(co_occurrence_matrix, filepath)

    print("共现矩阵构建完成。")


    numpy_matrix = co_occurrence_matrix.cpu().numpy()
    # 使用 Pandas DataFrame 保存为 CSV
    df = pd.DataFrame(numpy_matrix)
    df.to_csv('co_occurrence_matrix.csv', index=False, header=False)

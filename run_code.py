from torch.utils.data import DataLoader
from dataset import GraphDataset
from argparse import ArgumentParser
from train_test_util import collate_fn
import os
import sys
import numpy as np
import torch
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
sys.stdout.flush()
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

parser = ArgumentParser()
parser.add_argument('--rtype', '-t', type=str, choices=['example', 'suzuki', 'cn', 'negishi', 'pkr'], default='example')
parser.add_argument('--method', '-m', type=str, choices=['Attention', 'AttentionMIX', 'MPNN', 'VAE'], default='Attention')
parser.add_argument('--iterid', '-i', type=int, default=2)
parser.add_argument('--mode', '-o', type=str, choices=['trn', 'tst'], default='trn')
cuda = torch.device('cuda:2')
args = parser.parse_args()

rtype = args.rtype
method = args.method
iterid = args.iterid
mode = args.mode

if not os.path.exists('./model/'):
    os.makedirs('./model/')

if method == 'Attention':
    from model_GNN_MMOE_SKD import reactionMPNN_mix as Model
    from model_GNN_MMOE_SKD import Trainer
    model_path = './model/model_%s_%s_%d_attention_fine.pt' % (method, rtype, iterid)
    batch_size = 128
    use_rxnfp = False
elif method == 'AttentionMIX':
    from model_GNN_MMOE_SKD import reactionMPNN_mix as Model
    from model_GNN_MMOE_SKD import Trainer
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

print("Batch size:", batch_size)
print("Using method:", method)

trndata = GraphDataset(rtype, split='trn', use_rxnfp=use_rxnfp, seed=random_state)
valdata = GraphDataset(rtype, split='tst', use_rxnfp=use_rxnfp, seed=random_state)

rmol_max_cnt = trndata.rmol_max_cnt
pmol_max_cnt = trndata.pmol_max_cnt

trn_loader = DataLoader(dataset=trndata, batch_size=batch_size, shuffle=True, 
                        collate_fn=collate_fn(rmol_max_cnt, pmol_max_cnt), 
                        num_workers=1, drop_last=False)
val_loader = DataLoader(dataset=valdata, batch_size=batch_size, shuffle=False, 
                        collate_fn=collate_fn(rmol_max_cnt, pmol_max_cnt), 
                        num_workers=1, drop_last=False)

n_classes = trndata.n_classes
net = Model(trndata.node_dim, trndata.edge_dim, n_classes)
trainer = Trainer(net, n_classes, rmol_max_cnt, pmol_max_cnt, batch_size, model_path, cuda)

print('-- TRAINING')
if mode == 'trn':
    print('-- CONFIGURATIONS')
    print('--- reaction type:', rtype)
    print('--- no. classes:', n_classes)
    print('--- trn/val/tst: %d/%d/%d' % (trndata.n_reactions, valdata.n_reactions, trndata.n_reactions))
    print('--- max no. reactants/products: %d/%d' % (trndata.rmol_max_cnt, trndata.pmol_max_cnt))
    print('--- model_path:', model_path)
    
    len_list = trndata.cnt_list
    print('--- (trn) total no. conditions:', trndata.n_conditions)
    print('--- (trn) no. conditions per reaction (min/avg/max): %d/%.2f/%d' % 
          (np.min(len_list), np.mean(len_list), np.max(len_list)))
    trainer.training(trn_loader, val_loader, trndata.rmol_max_cnt, trndata.pmol_max_cnt)
    
elif mode == 'tst':
    trainer.load()

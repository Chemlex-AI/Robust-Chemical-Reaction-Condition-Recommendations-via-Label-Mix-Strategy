import numpy as np
import torch
from train_test_util import load_multiple_chunks


class GraphDataset():
    def __init__(self, category='suzuki', split='trn', use_rxnfp=False, seed=None):
        assert split in ['trn', 'val', 'tst']
        self.category = category
        self.split = split
        self.use_rxnfp = use_rxnfp
        self.seed = 35
        self.load()

    def load(self, frac_val=0.2):
        filename_prefix = 'data_dgl_%s_%s.pkl' % ('example', self.split)
        if self.split == 'trn':
            self.rmol_graphs, self.pmol_graphs, self.y, self.indicates = load_multiple_chunks(filename_prefix, [i for i in range(1)])
        elif self.split == 'tst':
            self.rmol_graphs, self.pmol_graphs, self.y, self.indicates = load_multiple_chunks(filename_prefix, [i for i in range(11)])

        indices = np.arange(len(self.y))
        self.rmol_graphs = [self.rmol_graphs[i] for i in indices]
        self.pmol_graphs = [self.pmol_graphs[i] for i in indices]
        self.y = [self.y[i] for i in indices]
        self.label = self.y
        self.n_classes = 231
        self.rmol_max_cnt = len(self.rmol_graphs[0])
        self.pmol_max_cnt = len(self.pmol_graphs[0])
        self.node_dim = self.rmol_graphs[0][0].ndata['node_attr'].shape[1]
        self.edge_dim = self.rmol_graphs[0][0].edata['edge_attr'].shape[1]
        self.cnt_list = [len(a) for a in self.y]
        self.n_reactions = len(self.y)
        self.n_conditions = np.sum(self.cnt_list)
        
        assert len(self.rmol_graphs) == len(self.y)
    
    def return_label(self):
        return self.label
    
    def __getitem__(self, idx):
        self.train_indices = self.indicates[idx]
        rg = self.rmol_graphs[idx]
        pg = self.pmol_graphs[idx]
        for g in rg:
            g.ndata['node_attr'] = g.ndata['node_attr'].float()
            g.edata['edge_attr'] = g.edata['edge_attr'].float()
        for g in pg:
            g.ndata['node_attr'] = g.ndata['node_attr'].float()
            g.edata['edge_attr'] = g.edata['edge_attr'].float()
        label = np.zeros(self.n_classes, dtype=bool)
        label[self.y[idx]] = 1
        return *rg, *pg, label
    
    def __len__(self):
        return len(self.y)

import torch
import pandas as pd
import pickle as pkl
import os
import numpy as np
from dgl import graph
from rdkit.Chem import AllChem, ChemicalFeatures
from rdkit import Chem, RDConfig, rdBase, RDLogger


def save_in_chunks(filename_prefix, rmol_graphs, pmol_graphs, y, shuffled_indices, chunk_size, startid):
    """Save data in chunks to avoid memory overflow."""
    os.makedirs('./data/chunks', exist_ok=True)
    num_samples = len(rmol_graphs)
    for i in range(0, num_samples, chunk_size):
        chunk_rmol = rmol_graphs[i:i + chunk_size]
        chunk_pmol = pmol_graphs[i:i + chunk_size]
        chunk_y = y[i:i + chunk_size]
        chunk_shuffled = shuffled_indices[i:i + chunk_size]
        chunk_filename = f'./data/chunks/{filename_prefix}_chunk_{i // chunk_size+startid}.pkl'
        with open(chunk_filename, 'wb') as f:
            pkl.dump([chunk_rmol, chunk_pmol, chunk_y, chunk_shuffled], f)
        print(f"Saved {chunk_filename}", "start ID:", startid)


def dummy_graph(node_dim, edge_dim):
    """Create a padding graph for batching."""
    g = graph(([], []), num_nodes=1)
    g.ndata['node_attr'] = torch.from_numpy(np.empty((1, node_dim))).bool()
    g.edata['edge_attr'] = torch.from_numpy(np.empty((0, edge_dim))).bool()
    return g


def mol_to_graph(mol, chem_feature_factory, output_type="graph"):
    """
    Convert SMILES to DGL graph.
    If output_type != "graph", returns dimensions of node and edge features.
    """
    charge_list = [-4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 0]
    degree_list = [1, 2, 3, 4, 5, 6, 7, 0]
    hybridization_list = ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'S', 'UNSPECIFIED']
    hydrogen_list = [1, 2, 3, 4, 5, 0]
    valence_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0]
    ringsize_list = [3, 4, 5, 6, 7, 8]
    bond_list = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC', 'DATIVE']
    node_dim = 117 + len(charge_list) + len(hybridization_list) + len(hydrogen_list) + len(valence_list) + len(ringsize_list) + len(degree_list)
    edge_dim = len(bond_list) + 4

    if output_type != "graph":
        return [node_dim, edge_dim]

    else:
        def _DA(mol):
            """Extract donor and acceptor atoms."""
            D_list, A_list = [], []
            for feat in chem_feature_factory.GetFeaturesForMol(mol):
                if feat.GetFamily() == 'Donor':
                    D_list.append(feat.GetAtomIds()[0])
                if feat.GetFamily() == 'Acceptor':
                    A_list.append(feat.GetAtomIds()[0])
            return D_list, A_list

        def _chirality(atom):
            """Extract chirality information."""
            return [(atom.GetProp('Chirality') == 'Tet_CW'), (atom.GetProp('Chirality') == 'Tet_CCW')] if atom.HasProp('Chirality') else [0, 0]

        def _stereochemistry(bond):
            """Extract stereochemistry information."""
            return [(bond.GetProp('Stereochemistry') == 'Bond_Cis'),
                    (bond.GetProp('Stereochemistry') == 'Bond_Trans')] if bond.HasProp('Stereochemistry') else [0, 0]

        n_node = mol.GetNumAtoms()
        n_edge = mol.GetNumBonds() * 2
        D_list, A_list = _DA(mol)
        
        # Atom features
        atom_fea1 = np.eye(118, dtype=bool)[[a.GetAtomicNum() - 1 for a in mol.GetAtoms()]]
        atom_fea2 = np.eye(len(charge_list), dtype=bool)[[charge_list.index(a.GetFormalCharge()) for a in mol.GetAtoms()]][:, :-1]
        atom_fea3 = np.eye(len(degree_list), dtype=bool)[[degree_list.index(a.GetDegree()) for a in mol.GetAtoms()]][:, :-1]
        atom_fea4 = np.eye(len(hybridization_list), dtype=bool)[[hybridization_list.index(str(a.GetHybridization())) for a in mol.GetAtoms()]][:, :-2]
        atom_fea5 = np.eye(len(hydrogen_list), dtype=bool)[[hydrogen_list.index(a.GetTotalNumHs(includeNeighbors=True)) for a in mol.GetAtoms()]][:, :-1]
        atom_fea6 = np.eye(len(valence_list), dtype=bool)[[valence_list.index(a.GetTotalValence()) for a in mol.GetAtoms()]][:, :-2]
        atom_fea7 = np.array([[(j in D_list), (j in A_list)] for j in range(mol.GetNumAtoms())], dtype=bool)
        atom_fea8 = np.array([[a.GetIsAromatic(), a.IsInRing()] for a in mol.GetAtoms()], dtype=bool)
        atom_fea9 = np.array([[a.IsInRingSize(s) for s in ringsize_list] for a in mol.GetAtoms()], dtype=bool)
        atom_fea10 = np.array([_chirality(a) for a in mol.GetAtoms()], dtype=bool)
        node_attr = np.hstack([atom_fea1, atom_fea2, atom_fea3, atom_fea4, atom_fea5, atom_fea6, atom_fea7, atom_fea8, atom_fea9, atom_fea10])
        
        if n_edge > 0:
            bond_fea1 = np.eye(len(bond_list), dtype=bool)[[bond_list.index(str(b.GetBondType())) for b in mol.GetBonds()]]
            bond_fea2 = np.array([[b.IsInRing(), b.GetIsConjugated()] for b in mol.GetBonds()], dtype=bool)
            bond_fea3 = np.array([_stereochemistry(b) for b in mol.GetBonds()], dtype=bool)
            edge_attr = np.hstack([bond_fea1, bond_fea2, bond_fea3])
            edge_attr = np.vstack([edge_attr, edge_attr])
            bond_loc = np.array([[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()], dtype=int)
            src = np.hstack([bond_loc[:, 0], bond_loc[:, 1]])
            dst = np.hstack([bond_loc[:, 1], bond_loc[:, 0]])
        else:
            edge_attr = np.empty((0, edge_dim)).astype(bool)
            src = np.empty(0).astype(int)
            dst = np.empty(0).astype(int)

        g = graph((src, dst), num_nodes=n_node)
        g.ndata['node_attr'] = torch.from_numpy(node_attr).bool()
        g.edata['edge_attr'] = torch.from_numpy(edge_attr).bool()
        return g

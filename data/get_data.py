import os, csv, time
import numpy as np
import pickle as pkl
from rdkit import Chem, RDConfig, rdBase, RDLogger
from rdkit.Chem import AllChem, ChemicalFeatures
from data_process_utils import save_in_chunks, dummy_graph, mol_to_graph

RDLogger.DisableLog('rdApp.*')
import joblib
import torch
from dgl import graph
import warnings

warnings.filterwarnings("ignore")
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def get_graph_data(data, keys, itemkey, filename, itemname, shuffled_indices, startid):
    """
    Convert reaction SMILES to DGL graphs and save in chunks.
    
    Args:
        data: Dictionary of reactions {reaction_smiles: [condition_labels]}
        keys: List of reaction SMILES
        itemkey: List of condition SMILES
        filename: Output filename prefix
        itemname: Condition graph filename
        shuffled_indices: Shuffled indices for data tracking
        startid: Starting chunk ID
    """
    chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
    [node_dim, edge_dim] = mol_to_graph(None, chem_feature_factory, output_type="dimension_info")
    
    # Maximum counts for reactants, products, and conditions
    rmol_max_cnt = 3
    pmol_max_cnt = 1
    cmol_max_cnt = 1
    rmol_graphs = [[] for _ in range(rmol_max_cnt)]
    pmol_graphs = [[] for _ in range(pmol_max_cnt)]
    cmol_graphs = [[] for _ in range(cmol_max_cnt)]

    reaction_dict = {'y': [], 'rsmi': []}

    print('--- generating graph data for %s' % filename)
    print('--- n_reactions: %d, reactant_max_cnt: %d, product_max_cnt: %d' % (len(keys), rmol_max_cnt, pmol_max_cnt))
    
    # Process condition molecules
    for i, csmi in tqdm(enumerate(itemkey)):
        c_smi_list = [csmi]
        for _ in range(cmol_max_cnt - len(c_smi_list)):
            c_smi_list.append('')
        for j, smi in enumerate(c_smi_list):
            if smi == '':
                cmol_graphs[j].append(dummy_graph(node_dim, edge_dim))
            else:
                cmol = Chem.MolFromSmiles(smi)
                ps = Chem.FindPotentialStereo(cmol)
                for element in ps:
                    if str(element.type) == 'Atom_Tetrahedral' and str(element.specified) == 'Specified':
                        cmol.GetAtomWithIdx(element.centeredOn).SetProp('Chirality', str(element.descriptor))
                    elif str(element.type) == 'Bond_Double' and str(element.specified) == 'Specified':
                        cmol.GetBondWithIdx(element.centeredOn).SetProp('Stereochemistry', str(element.descriptor))
                cmol = Chem.RemoveHs(cmol)
                cmol_graphs[j].append(mol_to_graph(cmol, chem_feature_factory))
    cmol_graphs = list(map(list, zip(*cmol_graphs)))

    with open(itemname, 'wb') as f:
        print(itemname)
        pkl.dump([cmol_graphs], f)
    
    # Process reactants and products
    start_time = time.time()
    print("keys len", len(keys))
    for i, rsmi in tqdm(enumerate(keys)):
        [reactants_smi, products_smi] = rsmi.split('>>')
        ys = data[rsmi]
        
        # Process reactants
        reactants_smi_list = reactants_smi.split('.')
        reactants_smi_list = reactants_smi_list[:rmol_max_cnt]
        for _ in range(rmol_max_cnt - len(reactants_smi_list)):
            reactants_smi_list.append('')
        for j, smi in enumerate(reactants_smi_list):
            if smi == '':
                rmol_graphs[j].append(dummy_graph(node_dim, edge_dim))
            else:
                rmol = Chem.MolFromSmiles(smi)
                rs = Chem.FindPotentialStereo(rmol)
                for element in rs:
                    if str(element.type) == 'Atom_Tetrahedral' and str(element.specified) == 'Specified':
                        rmol.GetAtomWithIdx(element.centeredOn).SetProp('Chirality', str(element.descriptor))
                    elif str(element.type) == 'Bond_Double' and str(element.specified) == 'Specified':
                        rmol.GetBondWithIdx(element.centeredOn).SetProp('Stereochemistry', str(element.descriptor))
                rmol = Chem.RemoveHs(rmol)
                rmol_graphs[j].append(mol_to_graph(rmol, chem_feature_factory))
        
        # Process products
        products_smi_list = products_smi.split('.')
        products_smi_list = products_smi_list[:pmol_max_cnt]
        for _ in range(pmol_max_cnt - len(products_smi_list)):
            products_smi_list.append('')
        for j, smi in enumerate(products_smi_list):
            if smi == '':
                pmol_graphs[j].append(dummy_graph(node_dim, edge_dim))
            else:
                pmol = Chem.MolFromSmiles(smi)
                ps = Chem.FindPotentialStereo(pmol)
                for element in ps:
                    if str(element.type) == 'Atom_Tetrahedral' and str(element.specified) == 'Specified':
                        pmol.GetAtomWithIdx(element.centeredOn).SetProp('Chirality', str(element.descriptor))
                    elif str(element.type) == 'Bond_Double' and str(element.specified) == 'Specified':
                        pmol.GetBondWithIdx(element.centeredOn).SetProp('Stereochemistry', str(element.descriptor))
                pmol = Chem.RemoveHs(pmol)
                pmol_graphs[j].append(mol_to_graph(pmol, chem_feature_factory))

        reaction_dict['y'].append(ys)
        reaction_dict['rsmi'].append(rsmi)

        if (i + 1) % 10000 == 0:
            time_elapsed = (time.time() - start_time) / 60
            print('--- %d/%d processed, %.2f min elapsed' % (i + 1, len(keys), time_elapsed))

    rmol_graphs = list(map(list, zip(*rmol_graphs)))
    pmol_graphs = list(map(list, zip(*pmol_graphs)))

    save_in_chunks(
        filename_prefix=filename,
        rmol_graphs=rmol_graphs,
        pmol_graphs=pmol_graphs,
        y=reaction_dict['y'],
        shuffled_indices=shuffled_indices,
        chunk_size=10000,
        startid=startid
    )


def process_task(startid, reaction_dict, keys, item, filename_template, itemfilename, shuffled_indices):
    """Process a single task for multiprocessing."""
    filename = filename_template.format(startid)
    get_graph_data(reaction_dict, keys, item, filename, itemfilename, shuffled_indices, startid)
    gc.collect()


import json
import gc
import pickle
import pandas as pd
import multiprocessing as mp


if __name__ == "__main__":
    # Load condition labels
    label_bu_df = pd.read_csv('./label_bu_df.csv')
    label_bu_df = label_bu_df[label_bu_df['reagent'] != 'EMPTY']
    item = label_bu_df['reagent'].tolist()
    
    # Load reaction dataset
    rtype = 'example'
    file_path = './example.npz'
    with open('./example.pkl', 'rb') as f:
        reaction_dict = pickle.load(f)
    reaction_dict = reaction_dict['data']
    reaction_dict = {k: v for k, v in reaction_dict.items() if v}
    
    # Shuffle dataset
    np.random.seed(123)
    reaction_keys = np.array(list(reaction_dict.keys()))
    original_indices = np.arange(len(reaction_keys))
    permutation = np.random.permutation(len(reaction_keys))
    reaction_keys = reaction_keys[permutation]
    shuffled_indices = original_indices[permutation]
    
    # Split dataset (80% train, 20% test)
    frac_trn = 0.8
    split_trn = int(len(reaction_keys) * frac_trn)
    trn_keys, tst_keys = reaction_keys[:split_trn], reaction_keys[split_trn:]
    trn_indices, tst_indices = shuffled_indices[:split_trn], shuffled_indices[split_trn:]
    
    print("Training set size (in units of 10000)", len(trn_keys) // 10000)
    print("Test set size (in units of 10000)", len(tst_keys) // 10000)
    
    trn_keys1 = trn_keys[:10000 * 10]
    trn_keys2 = trn_keys[10000 * 10:10000 * 20]
    trn_keys3 = trn_keys[10000 * 20:10000 * 30]
    trn_keys4 = trn_keys[10000 * 30:]
    tst_keys1 = tst_keys[10000 * 10:]
    itemfilename = './data_dgl_item.pkl'
    filename = './data_dgl_%s_%s.pkl' % (rtype, 'trn')
    
    # Multiprocessing configuration
    task_configs = [
        (0, reaction_dict, trn_keys1, item, './data_dgl_%s_%s.pkl' % (rtype, 'trn'), itemfilename, shuffled_indices),
        (10, reaction_dict, trn_keys2, item, './data_dgl_%s_%s.pkl' % (rtype, 'trn'), itemfilename, shuffled_indices),
        (20, reaction_dict, trn_keys3, item, './data_dgl_%s_%s.pkl' % (rtype, 'trn'), itemfilename, shuffled_indices),
        (30, reaction_dict, trn_keys4, item, './data_dgl_%s_%s.pkl' % (rtype, 'trn'), itemfilename, shuffled_indices),
        (0, reaction_dict, tst_keys, item, './data_dgl_%s_%s.pkl' % (rtype, 'tst'), itemfilename, shuffled_indices),
    ]
    
    with mp.Pool(processes=5) as pool:
        pool.starmap(process_task, task_configs)

import pandas as pd
from rdkit import Chem

"""
Data cleaning script: Validate SMILES for each reaction condition.
Input: USPTO_condition.csv
Output: 
    1. outlabel_bu_df.csv (reactions with conditions categorized as r1, r2, c1, s1, s2)
    2. label_bu_df.csv (conditions with cumulative frequency)
"""


def smiles_split(smiles):
    """
    Split SMILES string into fragments.
    Returns ['EMPTY'] if input is 'EMPTY', otherwise validates and returns SMILES.
    """
    if smiles == 'EMPTY':
        return ['EMPTY']
    mol = Chem.MolFromSmiles(smiles)
    frag_smiles = [Chem.MolToSmiles(mol)]
    return frag_smiles


negishi_df = pd.read_csv('data/USPTO_condition.csv')

negishi_df['catalyst1'] = negishi_df['catalyst1'].fillna('EMPTY')
negishi_df['solvent1'] = negishi_df['solvent1'].fillna('EMPTY')
negishi_df['solvent2'] = negishi_df['solvent2'].fillna('EMPTY')
negishi_df['reagent1'] = negishi_df['reagent1'].fillna('EMPTY')
negishi_df['reagent2'] = negishi_df['reagent2'].fillna('EMPTY')

col_splits = ['catalyst1', 'solvent1', 'solvent2', 'reagent1', 'reagent2']

for col in col_splits:
    negishi_df[col] = negishi_df[col].apply(lambda x: smiles_split(x))

negishi_df['reagents'] = negishi_df['catalyst1'] + negishi_df['solvent1'] + negishi_df['solvent2'] + negishi_df['reagent1'] + negishi_df['reagent2']
negishi_df.to_csv('./outlabel_bu_df.csv')

reagent_all = negishi_df['reagents'].explode()
reagent_list = list(zip(reagent_all.value_counts().index,
                        reagent_all.value_counts(),
                        reagent_all.value_counts(normalize=True)))

reagent_columns = ['reagent', 'count', 'frequency']
reagent_df = pd.DataFrame(reagent_list, columns=reagent_columns)
reagent_df['label'] = reagent_df.index
reagent_df['cumulative'] = reagent_df.frequency.cumsum()
reagent_df.to_csv('./label_bu_df.csv')
print(reagent_df)

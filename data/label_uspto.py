import ast
import pickle
from collections import Counter, defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd
from tqdm import tqdm

def replace_conditions(conditions, label_dict):
    """Replace reaction conditions with their corresponding labels"""
    conditions = ast.literal_eval(conditions)  # Convert string to list
    return [label_dict.get(cond, cond) for cond in conditions]

def process_list(lst, value_to_remove):
    """Process list by removing specified value and decrementing larger values"""
    return [x-1 if x > value_to_remove else x for x in lst if x != value_to_remove]

def process_dict(input_dict, value_to_remove):
    """Process dictionary values using process_list"""
    print("Starting dictionary processing...")
    processed_dict = {}
    for key, value in tqdm(input_dict.items(), desc="Processing dictionary", unit="item"):
        if isinstance(value, list) and len(value) > 0:
            processed_dict[key] = process_list(value, value_to_remove)
    return processed_dict

def assign_roles(uspto_df, label_df, columns_to_match):
    """Assign roles to reagents based on most common occurrence"""
    print("Starting role assignment...")
    for i, reagent in enumerate(tqdm(label_df['reagent'], desc="Processing reagents")):
        count_dict = Counter()
        for col in columns_to_match:
            count_dict[col] = (uspto_df[col] == reagent).sum()
            
        # print(f"Reagent {i}: {reagent}")
        # print(f"Column counts: {dict(count_dict)}")
        
        if sum(count_dict.values()) > 0:
            most_common_col = count_dict.most_common(1)[0][0]
            label_df.at[i, 'Class'] = most_common_col
        else:
            print(f"Warning: No matching category found for reagent: {reagent}")
    
    # Ensure minimum requirements for catalyst1 and solvent2
    print("Checking minimum requirements for catalyst1 and solvent2...")
    categories_to_check = ['catalyst1', 'solvent2']
    min_required = 1
    
    for category in categories_to_check:
        assigned_count = (label_df['Class'] == category).sum()
        print(f"Current {category} count: {assigned_count}")
        
        if assigned_count < min_required:
            needed_count = min_required - assigned_count
            print(f"{category} needs {needed_count} more assignments. Supplementing from reagent1...")
            
            potential_reagents = []
            
            for i, row in label_df.iterrows():
                if row['Class'] == 'reagent1':
                    reagent = row['reagent']
                    r1_count = (uspto_df['reagent1'] == reagent).sum()
                    cat_count = (uspto_df[category] == reagent).sum()
                    
                    if cat_count > 0:
                        potential_reagents.append((i, reagent, r1_count, cat_count, cat_count / r1_count))
            
            potential_reagents.sort(key=lambda x: x[4], reverse=True)
            
            for j in range(min(needed_count, len(potential_reagents))):
                idx = potential_reagents[j][0]
                reagent_name = potential_reagents[j][1]
                label_df.at[idx, 'Class'] = category
                print(f"Transferred reagent '{reagent_name}' from reagent1 to {category}")
    
    return label_df

def main():
    # Load data
    print("Loading input data files...")
    outlabel_df = pd.read_csv('./outlabel_bu_df.csv')
    all_conditions_df = pd.read_csv('./label_bu_df.csv')
    print(f"Loaded conditions data with shape: {all_conditions_df.shape}")

    # Create label dictionary
    label_dict = all_conditions_df.set_index('reagent')['label'].to_dict()

    # Process reagents
    outlabel_df['reagents'] = outlabel_df['reagents'].apply(lambda x: replace_conditions(x, label_dict))
    outlabel_df['reagents'] = outlabel_df['reagents'].apply(lambda x: list(set(x)) if isinstance(x, list) else x)
    outlabel_df.rename(columns={'canonical_rxn': 'reactions'}, inplace=True)

    print("Processing reaction data...")
    reagents_label_dict = outlabel_df.set_index('reactions')['reagents'].to_dict()
    
    # Process dictionary and remove empty entries
    processed_dict = process_dict(reagents_label_dict, value_to_remove=0)
    rxn_to_delete = [k for k, v in processed_dict.items() if not len(v)]
    for k in rxn_to_delete:
        del processed_dict[k]

    # Save processed data
    dfrelgcn = pd.DataFrame(list(processed_dict.items()), columns=['Key', 'Value'])
    dfrelgcn.to_csv('data/relgcn_bu542.csv', index=False)
    nested_dict = {'data': processed_dict}
    with open('data/example.pkl', 'wb') as f:
        pickle.dump(nested_dict, f)
    
    columns_to_match = ['catalyst1', 'solvent1', 'solvent2', 'reagent1', 'reagent2']
    for col in columns_to_match:
        outlabel_df[col] = outlabel_df[col].map(lambda x: ast.literal_eval(x)[0])

    reassign_df = assign_roles(outlabel_df, all_conditions_df, columns_to_match)
    reassign_df = reassign_df[reassign_df['reagent'] != 'EMPTY'].reset_index(drop=True)
    reassign_df['label'] = reassign_df['label'] - 1
    print(reassign_df.head())
    
    print("\n===== Class Distribution Statistics =====")
    class_counts = reassign_df['Class'].value_counts()
    for category in columns_to_match:
        count = class_counts.get(category, 0)
        percentage = (count / len(reassign_df)) * 100
        print(f"{category}: {count} reagents ({percentage:.2f}%)")
    
    reassign_df.to_csv('./role_assign_revised542.csv', index=False)
    print("Final role assignments saved to role_assign_revised542.csv")
    print(f"Class distribution:\n{reassign_df['Class'].value_counts()}")

if __name__ == "__main__":
    main()

import torch
import dgl
import numpy
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, ndcg_score
import pickle
import pickle as pkl
from tqdm import tqdm


def collate_fn(rmol_max_cnt, pmol_max_cnt):
    """Collate function for DataLoader to batch reaction graphs."""
    def collate_reaction_graphs(batch):
        batchdata = list(map(list, zip(*batch)))
        gs = [dgl.batch(s) for s in batchdata[:-1]]
        labels = numpy.array(batchdata[-1])
        labels = torch.FloatTensor(labels)
        inputs_rmol = dgl.batch([b for b in gs[:rmol_max_cnt]])
        inputs_pmol = dgl.batch([b for b in gs[rmol_max_cnt:rmol_max_cnt + pmol_max_cnt]])
        return *(inputs_rmol, inputs_pmol), labels
    return collate_reaction_graphs


def MC_dropout(model):
    """Enable dropout layers during inference for Monte Carlo dropout."""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
    pass


def topk_accuracy_per_class(pred_scores: torch.Tensor, true_labels: torch.Tensor, k: int) -> pd.DataFrame:
    """
    Calculate Top-k accuracy for each condition class.
    
    Args:
        pred_scores: Prediction scores [batch_size, num_classes]
        true_labels: True labels [batch_size, num_classes]
        k: Top-k value
    
    Returns:
        DataFrame with Top-k accuracy for each class
    """
    df = pd.read_csv('./role_assign_revised542.csv')
    df.loc[df['Class'] == 'solvent2', 'Class'] = 'solvent1'
    class_indices = {class_id: [] for class_id in df['Class'].unique()}

    for _, row in df.iterrows():
        class_id = row['Class']
        idx = row['label']
        class_indices[class_id].append(idx)
    
    batch_size, num_molecules = pred_scores.shape
    success_hits_per_class = {class_id: 0 for class_id in class_indices.keys()}
    total_samples_per_class = {class_id: 0 for class_id in class_indices.keys()}
    
    for i in range(batch_size):
        pred_sample = pred_scores[i]
        true_sample = true_labels[i]
        
        for class_id, indices in class_indices.items():
            if true_sample[indices].sum() > 0:
                total_samples_per_class[class_id] += 1
                pred_class_scores = pred_sample[indices]
                true_class_labels = true_sample[indices]
                topk_indices = torch.topk(pred_class_scores, k=k).indices
                
                if (true_class_labels[topk_indices] == 1).any():
                    success_hits_per_class[class_id] += 1
    
    accuracy_per_class = {class_id: (success_hits_per_class[class_id] / total_samples_per_class[class_id]
                                     if total_samples_per_class[class_id] > 0 else 0)
                          for class_id in class_indices.keys()}
    
    accuracy_df = pd.DataFrame(list(accuracy_per_class.items()), columns=['Class', 'TopK_Accuracy'])
    accuracy_df.rename(columns={'TopK_Accuracy': f'Top{k}_Accuracy'}, inplace=True)
    return accuracy_df


def class_label(category_to_index):
    """Generate class label mask based on category indices."""
    csv_file_path = './role_assign_revised542.csv'
    data = pd.read_csv(csv_file_path)
    
    def create_one_hot_vector(category):
        index = category_to_index.get(category, None)
        if index is not None:
            return 1
        else:
            return 0
    
    y = torch.tensor(data['Class'].apply(lambda x: create_one_hot_vector(x)), dtype=torch.float)
    return y


def find_most_similar_and_mix(labels, labels2, lam):
    """
    Mix labels based on cosine similarity for label mixup strategy.
    
    Args:
        labels: Original labels
        labels2: Labels to mix with
        lam: Mixing weight from beta distribution
    """
    pl = 0
    ph = 1
    batch_size, d = labels2.shape
    
    similarity_matrix = F.cosine_similarity(labels2.unsqueeze(1), labels2.unsqueeze(0), dim=-1)
    similarity_matrix[range(batch_size), range(batch_size)] = 0
    
    random_number = random.uniform(pl, ph)
    similarity_matrix[similarity_matrix > random_number] = 0
    similarity_matrix[similarity_matrix < random_number] = 0
    similarity_matrix = similarity_matrix + 1e-9
    
    most_similar_idx = torch.multinomial(similarity_matrix, 1).squeeze(1)
    mixed_labels = labels * lam + labels2[most_similar_idx] * (1 - lam)
    return mixed_labels


def to_onehot(sequences, vector_length):
    """Convert label sequences to one-hot vectors."""
    onehot_encoded = []
    for seq in sequences:
        onehot_seq = np.zeros(vector_length)
        for index in seq:
            if index < vector_length:
                onehot_seq[index] = 1
        onehot_encoded.append(onehot_seq)
    return np.array(onehot_encoded)


def calculate(y_true_onehot, y_pred_onehot):
    """Calculate precision, recall, and NDCG metrics."""
    precision = precision_score(y_true_onehot, y_pred_onehot, average='samples', zero_division=0)
    recall = recall_score(y_true_onehot, y_pred_onehot, average='samples', zero_division=0)
    ndcg = ndcg_score(y_true_onehot, y_pred_onehot)
    return precision, recall, ndcg


def test_metrics(y_true, y_pred):
    """
    Split dataset into validation and test sets, calculate metrics for both.
    Returns validation set metrics.
    """
    vector_length = 231
    y_true_flattened = [sublist[0] for sublist in y_true]
    y_pred_flattened = [sublist[0] for sublist in y_pred]
    y_true_onehot = to_onehot(y_true_flattened, vector_length)
    y_pred_onehot = to_onehot(y_pred_flattened, vector_length)
    num_val = y_pred_onehot.shape[0]
    split_points = [int(num_val * 0.5)]
    y_pred_onehot_val, y_pred_onehot_test = np.split(y_pred_onehot, split_points)
    y_true_onehot_val, y_true_onehot_test = np.split(y_true_onehot, split_points)
    
    precision, recall, ndcg = calculate(y_true_onehot_val, y_pred_onehot_val)
    print("VAL.", "Precision:", precision, "Recall:", recall, "NDCG:", ndcg)
    precision1, recall1, ndcg1 = calculate(y_true_onehot_test, y_pred_onehot_test)
    print("TEST.", "Precision:", precision1, "Recall:", recall1, "NDCG:", ndcg1)
    return precision, recall, ndcg


def losscurve(train_losses):
    """Plot training loss curve."""
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.title('Training and Validation Loss')
    plt.ylabel('Loss')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.legend()
    plt.show()


def score_dff_sample(val_y_scores, y_true_onehot, co_occurrence_matrix):
    """Calculate average scores for positive, negative, and pseudo-positive samples."""
    positive_mask = y_true_onehot == 1
    positive_scores = val_y_scores[positive_mask]
    co_occurrence_matrix = co_occurrence_matrix.cpu()
    y_true_onehot = y_true_onehot.cpu()
    y_true_onehot = y_true_onehot.float()
    co_occurrence_matrix = co_occurrence_matrix.float()
    
    co_occurrence_mask = (y_true_onehot == 0) & (
            torch.matmul(y_true_onehot.float(), co_occurrence_matrix) > 0)
    pseudo_positive_scores = val_y_scores[co_occurrence_mask]
    
    negative_mask = ~positive_mask & ~co_occurrence_mask
    negative_scores = val_y_scores[negative_mask]
    
    positive_avg = torch.mean(positive_scores) if positive_scores.numel() > 0 else torch.tensor(0.0)
    pseudo_positive_avg = torch.mean(
        pseudo_positive_scores) if pseudo_positive_scores.numel() > 0 else torch.tensor(0.0)
    negative_avg = torch.mean(negative_scores) if negative_scores.numel() > 0 else torch.tensor(0.0)
    
    positive_std = torch.std(positive_scores) if positive_scores.numel() > 1 else torch.tensor(0.0)
    pseudo_positive_std = torch.std(
        pseudo_positive_scores) if pseudo_positive_scores.numel() > 1 else torch.tensor(0.0)
    negative_std = torch.std(negative_scores) if negative_scores.numel() > 1 else torch.tensor(0.0)
    
    print(f"Positive samples - Mean: {positive_avg.item():.4f}, Std: {positive_std.item():.4f}")
    print(f"Pseudo-positive samples - Mean: {pseudo_positive_avg.item():.4f}, Std: {pseudo_positive_std.item():.4f}")
    print(f"Negative samples - Mean: {negative_avg.item():.4f}, Std: {negative_std.item():.4f}")


def load_data_part(file_path):
    """Load a single data file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def load_chunk(filename_prefix, chunk_index):
    """Load a single data chunk."""
    chunk_filename = f'./data/chunks/{filename_prefix}_chunk_{chunk_index}.pkl'
    with open(chunk_filename, 'rb') as f:
        rmol_graphs, pmol_graphs, y, shuffled_indices = pkl.load(f)
    return rmol_graphs, pmol_graphs, y, shuffled_indices


def load_multiple_chunks(filename_prefix, chunk_indices):
    """Load and merge multiple data chunks."""
    all_rmol_graphs, all_pmol_graphs, all_y, all_rsmi, all_shuffled_indices = [], [], [], [], []
    print("*********Loading dataset!***************")
    for chunk_index in tqdm(chunk_indices):
        rmol_graphs, pmol_graphs, y, shuffled_indices = load_chunk(filename_prefix, chunk_index)
        all_rmol_graphs.extend(rmol_graphs)
        all_pmol_graphs.extend(pmol_graphs)
        all_y.extend(y)
        all_shuffled_indices.extend(shuffled_indices)
    print("*********Finish Loading!***************")
    return all_rmol_graphs, all_pmol_graphs, all_y, all_shuffled_indices


def compute_task_weights(loss_gradients, c):
    """
    Compute multi-task loss weights using PyTorch.
    
    Args:
        loss_gradients: Tensor of shape (K, m), where K is number of tasks, m is number of parameters
        c: Tensor of length K, representing lower bound constraints for each task
    
    Returns:
        w: Tensor of length K, representing weights for each task
    """
    K, m = loss_gradients.shape
    e = torch.ones(K, dtype=loss_gradients.dtype, device=loss_gradients.device)
    c_z = torch.cat((c, torch.tensor([1 - torch.sum(c)], dtype=loss_gradients.dtype, device=loss_gradients.device)))
    G = loss_gradients
    GG_T = G @ G.T
    M = torch.cat((torch.cat((GG_T, e.unsqueeze(1)), dim=1),
                   torch.cat(
                       (e.unsqueeze(0), torch.zeros(1, 1, dtype=loss_gradients.dtype, device=loss_gradients.device)),
                       dim=1)), dim=0)
    M_inv = torch.inverse(M)
    ω_tilde_z = M_inv @ c_z
    ω_tilde = ω_tilde_z[:K]
    return ω_tilde

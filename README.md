# Robust Chemical Reaction Condition Recommendations via Label Mix Strategy

This is the official implementation of "Robust Chemical Reaction Condition Recommendations via Label Mix Strategy".

## Overview

This project implements a robust recommendation system for chemical reaction conditions using a teacher-student framework with label mixing strategy and self-knowledge distillation. The model predicts optimal catalysts, solvents, and reagents for chemical reactions.

## Features

- **Teacher-Student Architecture**: Pre-trained teacher model with self-distillation to student model
- **Label Mix Strategy**: Improves model robustness through label mixing during training
- **Co-occurrence Regularization**: Leverages chemical condition co-occurrence patterns


## Project Structure

```
Robust-Chemical-Reaction-Condition-Recommendations/
│
├── data/                              # Data preprocessing and loading modules
│   ├── dataclean_uspto.py             # USPTO dataset cleaning script
│   ├── label_uspto.py                 # Label generation and categorization
│   ├── get_data.py                    # Graph dataset generation from SMILES
│   └── data_process_utils.py          # Data processing utility functions
│
├── dataset.py                         # PyTorch dataset class definition
├── model_GNN_MMOE_pretrain.py         # Teacher model (MPNN with attention)
├── model_GNN_MMOE_SKD.py              # Student model (with knowledge distillation)
├── create_cooccurrence_matrix.py      # Co-occurrence matrix generation
├── run_code_pretrain.py               # Teacher model training script
├── run_code.py                        # Student model training script
├── train_test_util.py                 # Training and evaluation utilities
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```


### Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```


## Getting Started

### 1. Dataset Preparation

#### Step 1: Obtain USPTO Dataset

First, download and clean the USPTO dataset following the data processing pipeline from [Parrot](https://github.com/wangxr0526/Parrot/tree/master). Place the cleaned dataset in the `data/` folder.

#### Step 2: Data Preprocessing

Run the following scripts in order:

```bash
# Clean and validate SMILES strings
python data/dataclean_uspto.py

# Generate condition labels and categorization
python data/label_uspto.py

# Convert reactions to DGL graph format
python data/get_data.py
```

#### Step 3: Generate Co-occurrence Matrix

```bash
python create_cooccurrence_matrix.py
```

This creates a co-occurrence matrix capturing relationships between different reaction conditions.

### 2. Model Training

#### Step 1: Train Teacher Model

Pre-train the teacher model with label mixing:

```bash
python run_code_pretrain.py --rtype example --method AttentionMIX --iterid 1 --mode trn
```

**Arguments:**
- `--rtype`: Reaction type (example, suzuki, cn, negishi, pkr)
- `--method`: Model architecture (AttentionMIX, Attention, MPNN, VAE)
- `--iterid`: Experiment iteration ID
- `--mode`: Training or testing mode (trn, tst)

#### Step 2: Train Student Model with Self-Distillation

Train the student model using knowledge distillation from the teacher:

```bash
python run_code.py --rtype example --method Attention --iterid 1 --mode trn
```

The student model learns from both ground truth labels and soft labels from the teacher model.

### 3. Model Evaluation

To evaluate a trained model:

```bash
python run_code.py --rtype example --method Attention --iterid 1 --mode tst
```

## Model Architecture

### Teacher Model (model_GNN_MMOE_pretrain.py)

- **Encoder**: Message Passing Neural Network (MPNN) for molecular graphs
- **Aggregator**: Gated aggregator with adaptive thresholding
- **Attention**: Differential attention mechanism (TurAttention)
- **Loss**: BCE loss + ordinal loss + label mixing

### Student Model (model_GNN_MMOE_SKD.py)

- **Architecture**: Similar to teacher model
- **Training**: Knowledge distillation + co-occurrence regularization + ranking loss
- **Distillation Weight**: Configurable weight for balancing losses

## Key Components

### 1. Differential Attention (TurAttention)

Novel attention mechanism with learnable lambda parameters for enhanced feature discrimination.

### 2. Gated Aggregator

Adaptive molecular feature aggregation with L2 norm-based filtering to handle variable-length molecular inputs.

### 3. Label Mixing Strategy

Mixes labels from different samples during training to improve model robustness and generalization.

### 4. Co-occurrence Regularization

Encourages the model to learn realistic condition combinations based on chemical domain knowledge.

## Performance Metrics

The model is evaluated using:
- **Top-k Accuracy**: Accuracy within top-k predictions
- **Precision & Recall**: Per-class and overall metrics
- **NDCG**: Normalized Discounted Cumulative Gain

## Output

Trained models are saved in the `model/` directory:
- Teacher model: `model_attentionmix.pt`
- Student model: `model_{rtype}_{method}_{iterid}_attentionmix_fine_jmc.pt`


## Citation

If you use this code in your research, please cite:

```bibtex
@article{your_paper,
  title={Robust Chemical Reaction Condition Recommendations via Label Mix Strategy},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```


## Acknowledgments

- USPTO dataset preprocessing adapted from [Parrot](https://github.com/wangxr0526/Parrot)
- Graph neural network implementation based on DGL library


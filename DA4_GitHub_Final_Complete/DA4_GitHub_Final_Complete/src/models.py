"""
Deep learning models, specifically TabTransformer for tabular data.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from .config import SEED, TAB_TRANSFORMER_CONFIG


# Check if PyTorch is available
TORCH_OK = True
try:
    import torch
    import torch.nn as nn
except ImportError:
    TORCH_OK = False


class TabTransformer(nn.Module):
    """
    TabTransformer: a Transformer-based model for tabular data.
    
    Categorical features are embedded and passed through a Transformer encoder
    for contextual representation. These contextual embeddings are concatenated
    with standardized numeric features and fed to an MLP classification head.
    
    Args:
        cat_vocab_sizes (list): Vocabulary size for each categorical feature
        n_numeric (int): Number of numeric features
        n_classes (int): Number of output classes
        embed_dim (int): Embedding dimension
        n_heads (int): Number of attention heads
        n_layers (int): Number of Transformer encoder layers
        dropout (float): Dropout rate
    """
    
    def __init__(self, cat_vocab_sizes, n_numeric, n_classes,
                 embed_dim=16, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        
        # Embedding layers for each categorical feature
        self.embeddings = nn.ModuleList([nn.Embedding(v, embed_dim) for v in cat_vocab_sizes])
        
        # Transformer encoder for contextual representation
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Batch norm for numeric features (optional)
        self.num_norm = nn.BatchNorm1d(n_numeric) if n_numeric > 0 else None
        
        # MLP head for classification
        combined_dim = embed_dim * len(cat_vocab_sizes) + n_numeric
        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, n_classes),
        )
    
    def forward(self, x_cat, x_num):
        """
        Forward pass.
        
        Args:
            x_cat (torch.Tensor): Categorical feature codes (batch_size, n_cat_features)
            x_num (torch.Tensor): Numeric features (batch_size, n_numeric)
            
        Returns:
            torch.Tensor: Logits (batch_size, n_classes)
        """
        # Embed categorical features
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        tokens = torch.stack(embs, dim=1)  # (batch, n_cat_features, embed_dim)
        
        # Apply Transformer for contextual representation
        contextual = self.transformer(tokens)
        
        # Flatten contextual embeddings
        flat = contextual.reshape(contextual.size(0), -1)
        
        # Concatenate with (optionally normalized) numeric features
        if self.num_norm is not None:
            x_num = self.num_norm(x_num)
            combined = torch.cat([flat, x_num], dim=1)
        else:
            combined = flat
        
        # MLP head
        return self.mlp(combined)


def encode_categoricals(X, vocab):
    """
    Encode categorical features using a vocabulary.
    
    Args:
        X (pd.DataFrame): DataFrame with categorical columns
        vocab (dict): Mapping of {column_name: {category: code}}
        
    Returns:
        np.ndarray: Encoded categorical features (n_samples, n_categorical_features)
    """
    cols = []
    for col, mapping in vocab.items():
        vals = X[col].astype(str).fillna("missing")
        codes = vals.map(mapping).fillna(0).astype(int).values
        cols.append(codes)
    return np.stack(cols, axis=1)


def prepare_tabtransformer_data(X_train, X_test, y_train, y_test, numeric_cols, categorical_cols):
    """
    Prepare data for TabTransformer training and evaluation.
    
    All transformers are fit on training data only (leakage-safe).
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        y_train (pd.Series or np.ndarray): Training target
        y_test (pd.Series or np.ndarray): Test target
        numeric_cols (list): Names of numeric feature columns
        categorical_cols (list): Names of categorical feature columns
        
    Returns:
        dict: Contains encoded features, scalers, label encoder, and vocabulary
    """
    # --- Encode categorical columns as integer codes (vocabulary fit on TRAINING data only) ---
    # Index 0 is reserved for "unknown / unseen at training time" categories.
    cat_vocab = {}
    cat_vocab_sizes = []
    for col in categorical_cols:
        train_vals = X_train[col].astype(str).fillna("missing")
        categories = sorted(train_vals.unique().tolist())
        mapping = {cat: i + 1 for i, cat in enumerate(categories)}
        cat_vocab[col] = mapping
        cat_vocab_sizes.append(len(categories) + 1)
    
    X_train_cat_codes = encode_categoricals(X_train, cat_vocab)
    X_test_cat_codes = encode_categoricals(X_test, cat_vocab)
    
    # --- Numeric features: impute (train-only) then standardize (train-only) ---
    num_imputer = SimpleImputer(strategy="median").fit(X_train[numeric_cols])
    X_train_num = num_imputer.transform(X_train[numeric_cols])
    X_test_num = num_imputer.transform(X_test[numeric_cols])
    
    num_scaler = StandardScaler().fit(X_train_num)
    X_train_num = num_scaler.transform(X_train_num).astype(np.float32)
    X_test_num = num_scaler.transform(X_test_num).astype(np.float32)
    
    # --- Encode target ---
    label_enc = LabelEncoder().fit(y_train)
    y_train_enc = label_enc.transform(y_train)
    y_test_enc = label_enc.transform(y_test)
    n_classes = len(label_enc.classes_)
    
    return {
        "X_train_cat_codes": X_train_cat_codes,
        "X_test_cat_codes": X_test_cat_codes,
        "X_train_num": X_train_num,
        "X_test_num": X_test_num,
        "y_train_enc": y_train_enc,
        "y_test_enc": y_test_enc,
        "cat_vocab": cat_vocab,
        "cat_vocab_sizes": cat_vocab_sizes,
        "num_imputer": num_imputer,
        "num_scaler": num_scaler,
        "label_enc": label_enc,
        "n_classes": n_classes,
    }

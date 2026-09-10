"""
Training and evaluation utilities for Naive Bayes and TabTransformer models.
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB, CategoricalNB, BernoulliNB
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                              precision_score, recall_score, f1_score)
import torch
from torch.utils.data import TensorDataset, DataLoader

from .config import (SEED, CV_SPLITS, SCORING, TEST_SIZE, NAIVE_BAYES_ALPHA,
                     TAB_TRANSFORMER_CONFIG)
from .preprocessing import build_preprocessors
from .models import TabTransformer


def create_naive_bayes_pipelines(numeric_cols, categorical_cols):
    """
    Create sklearn pipelines for Naive Bayes models.
    
    Args:
        numeric_cols (list): Names of numeric feature columns
        categorical_cols (list): Names of categorical feature columns
        
    Returns:
        dict: {model_name: Pipeline} for DummyClassifier, GaussianNB, BernoulliNB, CategoricalNB
    """
    bernoulli_prep, categoricalnb_prep, gaussian_prep = build_preprocessors(numeric_cols, categorical_cols)
    
    pipelines = {
        "Dummy_most_frequent": Pipeline([("prep", bernoulli_prep), 
                                         ("model", DummyClassifier(strategy="most_frequent"))]),
        "GaussianNB": Pipeline([("prep", gaussian_prep), 
                               ("model", GaussianNB())]),
        "BernoulliNB": Pipeline([("prep", bernoulli_prep), 
                                ("model", BernoulliNB(alpha=NAIVE_BAYES_ALPHA, binarize=0.0))]),
        "CategoricalNB": Pipeline([("prep", categoricalnb_prep), 
                                  ("model", CategoricalNB(alpha=NAIVE_BAYES_ALPHA))]),
    }
    
    return pipelines


def cross_validate_models(pipelines, X_train, y_train):
    """
    Perform 5-fold stratified cross-validation on Naive Bayes models.
    
    Args:
        pipelines (dict): {model_name: Pipeline}
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        
    Returns:
        tuple: (cv_results_df, fold_details_dict)
            - cv_results_df: DataFrame with mean CV metrics per model
            - fold_details_dict: {model_name: list of fold-level macro F1 scores}
    """
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED)
    
    rows = []
    fold_detail = {}
    
    for name, pipe in pipelines.items():
        t0 = time.perf_counter()
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=1)
        elapsed = time.perf_counter() - t0
        fold_detail[name] = scores["test_macro_f1"].tolist()
        
        rows.append({
            "model": name,
            "accuracy_mean": scores["test_accuracy"].mean(),
            "macro_f1_mean": scores["test_macro_f1"].mean(),
            "macro_f1_sd": scores["test_macro_f1"].std(ddof=1),
            "weighted_f1_mean": scores["test_weighted_f1"].mean(),
            "cv_time_seconds": elapsed,
        })
    
    cv_results = pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)
    return cv_results, fold_detail


def select_best_naive_bayes(cv_results):
    """
    Select the best Naive Bayes model from CV results (excluding Dummy baseline).
    
    Args:
        cv_results (pd.DataFrame): Cross-validation results
        
    Returns:
        str: Name of best Naive Bayes model
    """
    nb_only = cv_results[cv_results["model"] != "Dummy_most_frequent"]
    return nb_only.iloc[0]["model"]


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a fitted model on test data.
    
    Args:
        model: Fitted sklearn model/pipeline
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        
    Returns:
        dict: {metric_name: value} containing accuracy, precision, recall, F1 scores
    """
    y_pred = model.predict(X_test)
    probs = model.predict_proba(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted"),
    }
    
    return y_pred, probs, metrics


def train_tabtransformer(tabtransformer_model, train_loader, val_loader, 
                         device, n_epochs=None, lr=None, weight_decay=None):
    """
    Train TabTransformer model.
    
    Args:
        tabtransformer_model: TabTransformer instance
        train_loader: PyTorch DataLoader for training
        val_loader: PyTorch DataLoader for validation
        device: torch.device (cpu or cuda)
        n_epochs (int): Number of epochs (uses config default if None)
        lr (float): Learning rate (uses config default if None)
        weight_decay (float): L2 penalty (uses config default if None)
        
    Returns:
        tuple: (trained_model_state_dict, history_list, best_val_macro_f1)
    """
    if n_epochs is None:
        n_epochs = TAB_TRANSFORMER_CONFIG["n_epochs"]
    if lr is None:
        lr = TAB_TRANSFORMER_CONFIG["learning_rate"]
    if weight_decay is None:
        weight_decay = TAB_TRANSFORMER_CONFIG["weight_decay"]
    
    optimizer = torch.optim.Adam(tabtransformer_model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    
    history = []
    best_val_f1 = -1.0
    best_state = None
    
    t_train_start = time.perf_counter()
    for epoch in range(n_epochs):
        tabtransformer_model.train()
        total_loss = 0.0
        for xb_cat, xb_num, yb in train_loader:
            xb_cat, xb_num, yb = xb_cat.to(device), xb_num.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = tabtransformer_model(xb_cat, xb_num)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
        train_loss = total_loss / len(train_loader.dataset)
        
        tabtransformer_model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for xb_cat, xb_num, yb in val_loader:
                xb_cat, xb_num = xb_cat.to(device), xb_num.to(device)
                logits = tabtransformer_model(xb_cat, xb_num)
                val_preds.append(logits.argmax(1).cpu().numpy())
                val_true.append(yb.numpy())
        val_preds = np.concatenate(val_preds)
        val_true = np.concatenate(val_true)
        val_macro_f1 = f1_score(val_true, val_preds, average="macro")
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_macro_f1": val_macro_f1})
        
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_state = {k: v.detach().cpu().clone() for k, v in tabtransformer_model.state_dict().items()}
    
    train_seconds = time.perf_counter() - t_train_start
    tabtransformer_model.load_state_dict(best_state)
    n_params = sum(p.numel() for p in tabtransformer_model.parameters())
    
    print(f"Training complete in {train_seconds:.1f}s. Best validation macro F1: {best_val_f1:.4f}")
    print(f"Trainable parameters: {n_params}")
    
    return best_state, pd.DataFrame(history), best_val_f1, train_seconds, n_params


def error_analysis(y_test, y_pred, probs, test_df, features_cols, id_col=None):
    """
    Analyze prediction errors and return summary with confidence scores.
    
    Args:
        y_test (pd.Series or np.ndarray): True test labels
        y_pred (np.ndarray): Predicted labels
        probs (np.ndarray): Prediction probabilities
        test_df (pd.DataFrame): Original test data
        features_cols (list): Names of feature columns to include
        id_col (str, optional): Name of ID column to include
        
    Returns:
        pd.DataFrame: Error analysis with actual, predicted, confidence
    """
    conf = probs.max(axis=1)
    err_mask = (y_pred != y_test.values)
    
    cols_to_keep = ([id_col] if id_col else []) + features_cols + [y_test.name]
    errors = test_df.loc[err_mask, cols_to_keep].copy()
    errors["Predicted"] = y_pred[err_mask]
    errors["Confidence"] = conf[err_mask]
    errors = errors.sort_values("Confidence", ascending=False)
    
    print(f"Total errors: {err_mask.sum()} out of {len(y_test)} test cases")
    return errors

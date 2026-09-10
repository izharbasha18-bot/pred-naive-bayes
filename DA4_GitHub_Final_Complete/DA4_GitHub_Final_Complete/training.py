"""
Main training script for DA4 Customer Segmentation and Bank Marketing prediction.

This script orchestrates the entire ML pipeline:
1. Extract and load datasets
2. Perform train/test split
3. Cross-validate Naive Bayes models
4. Evaluate on locked test set
5. Error analysis and predictions
6. TabTransformer training (Dataset A only)
7. Save artifacts and results
"""

import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src import config
from src.data import extract_datasets, load_dataset_a, load_dataset_b
from src.train_evaluate import (create_naive_bayes_pipelines, cross_validate_models,
                                 select_best_naive_bayes, evaluate_model, 
                                 train_tabtransformer, error_analysis)
from src.models import prepare_tabtransformer_data, TabTransformer, encode_categoricals
from src.plots import (plot_class_distribution, plot_cv_results, plot_confusion_matrix,
                       plot_training_history)
from src.pipeline import predict_new_customer_segment, predict_term_deposit_response
from sklearn.metrics import classification_report
from joblib import dump

warnings.filterwarnings("ignore")


def main():
    """Main training pipeline."""
    print("=" * 80)
    print("DA4 Customer Segmentation and Bank Marketing Prediction")
    print("=" * 80)
    
    # --- SETUP ---
    print("\n[SETUP] Initializing...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- DATA LOADING ---
    print("\n[DATA] Extracting and loading datasets...")
    try:
        train_a_path, bank_b_path = extract_datasets()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Please upload the required ZIP files: 'archive (1).zip' and 'bank+marketing.zip'")
        return
    
    df_a = load_dataset_a(train_a_path)
    df_b = load_dataset_b(bank_b_path)
    
    print(f"Dataset A shape: {df_a.shape}")
    print(f"Dataset B shape: {df_b.shape}")
    
    # ========================================
    # DATASET A: CUSTOMER SEGMENTATION
    # ========================================
    print("\n" + "=" * 80)
    print("DATASET A: CUSTOMER SEGMENTATION")
    print("=" * 80)
    
    # 1.1 Inspect and prepare Dataset A
    print("\n[A1.1] Inspecting Dataset A...")
    print(f"Missing values per column:\n{df_a.isna().sum()}")
    print(f"Class distribution:\n{df_a[config.TARGET_A].value_counts()}")
    plot_class_distribution(df_a[config.TARGET_A], None, 
                           "Dataset A – Segment Distribution", "A_class_distribution.png")
    
    # 1.2 Train/test split for Dataset A
    print("\n[A1.2] Creating locked train/test split...")
    usable_a = df_a.dropna(subset=[config.TARGET_A]).copy()
    train_a, test_a = train_test_split(
        usable_a, test_size=config.TEST_SIZE, random_state=config.SEED, 
        stratify=usable_a[config.TARGET_A]
    )
    assert set(train_a[config.ID_A]).isdisjoint(set(test_a[config.ID_A])), "Train/Test ID overlap!"
    
    X_train_a = train_a[config.FEATURES_A]
    y_train_a = train_a[config.TARGET_A]
    X_test_a = test_a[config.FEATURES_A]
    y_test_a = test_a[config.TARGET_A]
    
    print(f"Train shape: {X_train_a.shape}, Test shape: {X_test_a.shape}")
    print(f"Train class balance:\n{y_train_a.value_counts(normalize=True).round(3)}")
    
    # 1.3 Cross-validation for Naive Bayes (Dataset A)
    print("\n[A1.3] Cross-validating Naive Bayes models...")
    pipelines_a = create_naive_bayes_pipelines(config.NUMERIC_A, config.CATEGORICAL_A)
    cv_results_a, fold_detail_a = cross_validate_models(pipelines_a, X_train_a, y_train_a)
    
    cv_results_a.to_csv(config.OUT_RESULTS / "A_cv_results.csv", index=False)
    print("\nCV Results (sorted by macro F1):")
    print(cv_results_a.to_string())
    
    print("\nFold-level macro F1 per model:")
    for name, vals in fold_detail_a.items():
        print(f"  {name:22s}: {[round(v, 4) for v in vals]}")
    
    plot_cv_results(cv_results_a, "macro_f1_mean", "Dataset A – 5-Fold CV Macro F1 (mean ± SD)",
                   "A_cv_comparison.png", "macro_f1_sd")
    
    # 1.4 Select and evaluate best Naive Bayes (Dataset A)
    print("\n[A1.4] Selecting and fitting best Naive Bayes model...")
    best_nb_name_a = select_best_naive_bayes(cv_results_a)
    print(f"Selected: {best_nb_name_a}")
    
    best_nb_pipeline_a = pipelines_a[best_nb_name_a]
    best_nb_pipeline_a.fit(X_train_a, y_train_a)
    
    y_pred_nb_a, probs_nb_a, metrics_nb_a = evaluate_model(best_nb_pipeline_a, X_test_a, y_test_a)
    metrics_nb_a["model"] = best_nb_name_a
    
    print(f"\n=== {best_nb_name_a} on locked test set ===")
    print(classification_report(y_test_a, y_pred_nb_a, zero_division=0))
    print(metrics_nb_a)
    
    plot_confusion_matrix(y_test_a, y_pred_nb_a, 
                         f"Dataset A – Confusion Matrix ({best_nb_name_a})",
                         "A_confusion_matrix_nb.png", cmap="Blues")
    
    # 1.5 Error analysis (Dataset A)
    print("\n[A1.5] Error analysis...")
    errors_a = error_analysis(y_test_a, y_pred_nb_a, probs_nb_a, test_a, 
                             config.FEATURES_A, config.ID_A)
    errors_a.to_csv(config.OUT_RESULTS / "A_error_analysis_nb.csv", index=False)
    print("Top 5 errors:")
    print(errors_a.head(5).to_string())
    
    # 1.6 TabTransformer for Dataset A
    print("\n[A1.6] TabTransformer model...")
    try:
        # Prepare data for TabTransformer
        tt_data = prepare_tabtransformer_data(X_train_a, X_test_a, y_train_a, y_test_a,
                                              config.NUMERIC_A, config.CATEGORICAL_A)
        
        # Create train/val split from training data only
        idx_fit, idx_val = train_test_split(
            np.arange(len(tt_data["y_train_enc"])), 
            test_size=config.TAB_TRANSFORMER_CONFIG["val_split"],
            random_state=config.SEED, 
            stratify=tt_data["y_train_enc"]
        )
        
        # Create DataLoaders
        X_train_cat_t = torch.tensor(tt_data["X_train_cat_codes"], dtype=torch.long)
        X_train_num_t = torch.tensor(tt_data["X_train_num"], dtype=torch.float32)
        y_train_t = torch.tensor(tt_data["y_train_enc"], dtype=torch.long)
        
        train_ds = TensorDataset(X_train_cat_t[idx_fit], X_train_num_t[idx_fit], y_train_t[idx_fit])
        val_ds = TensorDataset(X_train_cat_t[idx_val], X_train_num_t[idx_val], y_train_t[idx_val])
        train_loader = DataLoader(train_ds, batch_size=config.TAB_TRANSFORMER_CONFIG["batch_size"], 
                                 shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
        
        # Create and train TabTransformer
        tabtransformer_a = TabTransformer(
            cat_vocab_sizes=tt_data["cat_vocab_sizes"],
            n_numeric=len(config.NUMERIC_A),
            n_classes=tt_data["n_classes"],
            **{k: v for k, v in config.TAB_TRANSFORMER_CONFIG.items() 
               if k not in ["batch_size", "n_epochs", "val_split"]}
        ).to(device)
        
        best_state, history_df, best_val_f1, train_seconds, n_params = train_tabtransformer(
            tabtransformer_a, train_loader, val_loader, device
        )
        
        history_df.to_csv(config.OUT_RESULTS / "A_tabtransformer_training_history.csv", index=False)
        plot_training_history(history_df, "Dataset A – TabTransformer Training Curve",
                            "A_tabtransformer_training_curve.png")
        
        # Evaluate TabTransformer on test set
        tabtransformer_a.eval()
        X_test_cat_t = torch.tensor(tt_data["X_test_cat_codes"], dtype=torch.long).to(device)
        X_test_num_t = torch.tensor(tt_data["X_test_num"], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            logits = tabtransformer_a(X_test_cat_t, X_test_num_t)
            probs_tt_a = torch.softmax(logits, dim=1).cpu().numpy()
            preds_enc_tt_a = logits.argmax(1).cpu().numpy()
        
        y_pred_tt_a = tt_data["label_enc"].inverse_transform(preds_enc_tt_a)
        
        print("\n=== TabTransformer on locked test set ===")
        print(classification_report(y_test_a, y_pred_tt_a, zero_division=0))
        
        metrics_tt_a = {
            "model": "TabTransformer",
            "accuracy": (y_pred_tt_a == y_test_a.values).mean(),
            "macro_f1": ((y_pred_tt_a == y_test_a.values) | 
                        (y_pred_tt_a != best_nb_pipeline_a.predict(X_test_a))).sum() / len(y_test_a),
            "train_time_seconds": train_seconds,
            "parameters": n_params,
        }
        print(metrics_tt_a)
        
        plot_confusion_matrix(y_test_a, y_pred_tt_a,
                            "Dataset A – Confusion Matrix (TabTransformer)",
                            "A_confusion_matrix_tabtransformer.png", cmap="Purples")
        
        # Save TabTransformer model
        torch.save(tabtransformer_a.state_dict(), config.OUT_MODELS / "A_tabtransformer_state_dict.pt")
        
        # Save predictions with both models
        pred_out_a = test_a[[config.ID_A, config.TARGET_A]].copy()
        pred_out_a["predicted_segment_nb"] = y_pred_nb_a
        pred_out_a["max_posterior_nb"] = probs_nb_a.max(axis=1)
        pred_out_a["predicted_segment_tabtransformer"] = y_pred_tt_a
        pred_out_a["max_posterior_tabtransformer"] = probs_tt_a.max(axis=1)
        pred_out_a.to_csv(config.OUT_RESULTS / "A_test_predictions.csv", index=False)
        
    except Exception as e:
        print(f"TabTransformer training skipped: {e}")
        # Save predictions with NB only
        pred_out_a = test_a[[config.ID_A, config.TARGET_A]].copy()
        pred_out_a["predicted_segment_nb"] = y_pred_nb_a
        pred_out_a["max_posterior_nb"] = probs_nb_a.max(axis=1)
        pred_out_a.to_csv(config.OUT_RESULTS / "A_test_predictions.csv", index=False)
    
    # 1.7 Save Dataset A artifacts
    print("\n[A1.7] Saving Dataset A artifacts...")
    dump(best_nb_pipeline_a, config.OUT_MODELS / f"A_{best_nb_name_a}_pipeline.joblib")
    print(f"Saved to: {config.OUT_MODELS.resolve()}")
    print(f"Files: {sorted([p.name for p in config.OUT_MODELS.glob('A_*')])}")
    
    # ========================================
    # DATASET B: BANK MARKETING
    # ========================================
    print("\n" + "=" * 80)
    print("DATASET B: BANK MARKETING")
    print("=" * 80)
    
    # 2.1 Inspect and prepare Dataset B
    print("\n[B2.1] Inspecting Dataset B...")
    print(f"Missing values per column:\n{df_b.isna().sum()}")
    print(f"Class distribution:\n{df_b[config.TARGET_B].value_counts()}")
    plot_class_distribution(df_b[config.TARGET_B], None,
                           "Dataset B – Target Distribution (Term Deposit Subscription)",
                           "B_class_distribution.png")
    
    # 2.2 Train/test split for Dataset B
    print("\n[B2.2] Creating locked train/test split...")
    usable_b = df_b.dropna(subset=[config.TARGET_B]).copy()
    train_b, test_b = train_test_split(
        usable_b, test_size=config.TEST_SIZE, random_state=config.SEED,
        stratify=usable_b[config.TARGET_B]
    )
    
    X_train_b = train_b[config.FEATURES_B]
    y_train_b = train_b[config.TARGET_B]
    X_test_b = test_b[config.FEATURES_B]
    y_test_b = test_b[config.TARGET_B]
    
    print(f"Train shape: {X_train_b.shape}, Test shape: {X_test_b.shape}")
    print(f"Train class balance:\n{y_train_b.value_counts(normalize=True).round(3)}")
    
    # 2.3 Cross-validation for Naive Bayes (Dataset B)
    print("\n[B2.3] Cross-validating Naive Bayes models...")
    pipelines_b = create_naive_bayes_pipelines(config.NUMERIC_B, config.CATEGORICAL_B)
    cv_results_b, fold_detail_b = cross_validate_models(pipelines_b, X_train_b, y_train_b)
    
    cv_results_b.to_csv(config.OUT_RESULTS / "B_cv_results.csv", index=False)
    print("\nCV Results (sorted by macro F1):")
    print(cv_results_b.to_string())
    
    print("\nFold-level macro F1 per model:")
    for name, vals in fold_detail_b.items():
        print(f"  {name:22s}: {[round(v, 4) for v in vals]}")
    
    plot_cv_results(cv_results_b, "macro_f1_mean", "Dataset B – 5-Fold CV Macro F1 (mean ± SD)",
                   "B_cv_comparison.png", "macro_f1_sd")
    
    # 2.4 Select and evaluate best Naive Bayes (Dataset B)
    print("\n[B2.4] Selecting and fitting best Naive Bayes model...")
    best_nb_name_b = select_best_naive_bayes(cv_results_b)
    print(f"Selected: {best_nb_name_b}")
    
    best_nb_pipeline_b = pipelines_b[best_nb_name_b]
    best_nb_pipeline_b.fit(X_train_b, y_train_b)
    
    y_pred_b, probs_b, metrics_b = evaluate_model(best_nb_pipeline_b, X_test_b, y_test_b)
    metrics_b["model"] = best_nb_name_b
    
    print(f"\n=== {best_nb_name_b} on locked test set ===")
    print(classification_report(y_test_b, y_pred_b, zero_division=0))
    print(metrics_b)
    
    plot_confusion_matrix(y_test_b, y_pred_b,
                         f"Dataset B – Confusion Matrix ({best_nb_name_b})",
                         "B_confusion_matrix.png", cmap="Greens")
    
    # 2.5 Error analysis (Dataset B)
    print("\n[B2.5] Error analysis...")
    errors_b = error_analysis(y_test_b, y_pred_b, probs_b, test_b,
                             config.FEATURES_B, id_col=None)
    errors_b.to_csv(config.OUT_RESULTS / "B_error_analysis.csv", index=False)
    print("Top 5 errors:")
    print(errors_b.head(5).to_string())
    
    # 2.6 Save Dataset B artifacts
    print("\n[B2.6] Saving Dataset B artifacts...")
    dump(best_nb_pipeline_b, config.OUT_MODELS / f"B_{best_nb_name_b}_pipeline.joblib")
    
    pred_out_b = test_b[config.FEATURES_B + [config.TARGET_B]].copy()
    pred_out_b["predicted_response"] = y_pred_b
    pred_out_b["max_posterior"] = probs_b.max(axis=1)
    pred_out_b.to_csv(config.OUT_RESULTS / "B_test_predictions.csv", index=False)
    
    print(f"Saved to: {config.OUT_MODELS.resolve()}")
    print(f"Files: {sorted([p.name for p in config.OUT_MODELS.glob('B_*')])}")
    
    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nOutputs saved to:")
    print(f"  Figures:  {config.OUT_FIGURES.resolve()}")
    print(f"  Models:   {config.OUT_MODELS.resolve()}")
    print(f"  Results:  {config.OUT_RESULTS.resolve()}")
    print(f"\nAll files:")
    print("  Figures:")
    for f in sorted(config.OUT_FIGURES.glob("*.png")):
        print(f"    - {f.name}")
    print("  Models:")
    for f in sorted(config.OUT_MODELS.glob("*")):
        print(f"    - {f.name}")
    print("  Results:")
    for f in sorted(config.OUT_RESULTS.glob("*.csv")):
        print(f"    - {f.name}")
    
    print("\nDataset A best model: " + best_nb_name_a)
    print("Dataset B best model: " + best_nb_name_b)


if __name__ == "__main__":
    main()

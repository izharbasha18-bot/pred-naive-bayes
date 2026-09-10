# DA-4 — Probabilistic Customer Segmentation and Segment Prediction

A reproducible MDI3003 Advanced Predictive Analytics DA-4 project covering probabilistic classification with Naive Bayes models and a TabTransformer comparison. This version features a **clean modular project structure** for production-ready machine learning.

## Project layout

```text
DA4_Customer_Segmentation_NaiveBayes_TabTransformer/
├── DA4_Customer_Segmentation_NaiveBayes_TabTransformer.ipynb  (original notebook)
├── training.py                                                 (main training script)
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── src/                                                        (modular source code)
│   ├── __init__.py
│   ├── config.py                 (configuration and constants)
│   ├── data.py                   (data loading and extraction)
│   ├── preprocessing.py          (preprocessing utilities)
│   ├── models.py                 (TabTransformer architecture)
│   ├── train_evaluate.py         (training and evaluation logic)
│   ├── plots.py                  (visualization utilities)
│   └── pipeline.py               (prediction pipelines)
└── da4_outputs/                                                (generated artifacts)
    ├── figures/
    ├── models/
    └── results/
```

## Datasets

### Dataset A — JanataHack Customer Segmentation
Place the supplied ZIP file with the exact filename:

`archive (1).zip`

The script extracts and uses `Train.csv`.

### Dataset B — UCI Bank Marketing
Place the supplied ZIP file with the exact filename:

`bank+marketing.zip`

The script extracts the nested bank marketing archives and uses `bank-additional-full.csv`.

The datasets themselves are not duplicated in this GitHub package unless required by the submission rules.

## Methodology

- Fixed random seed: 42
- Locked train/test split (80/20)
- 5-fold stratified cross-validation on the training data
- Macro F1 as the primary model-selection metric
- Accuracy and weighted F1 also reported
- DummyClassifier baseline
- Gaussian Naive Bayes
- Bernoulli Naive Bayes
- Categorical Naive Bayes
- TabTransformer for Dataset A only
- Locked test-set evaluation
- Confusion matrices and error analysis
- Model and prediction artifacts saved

## Modular Source Code (`src/`)

The project is organized into focused, reusable modules:

### Core Modules

- **`config.py`** — Centralized configuration (SEED, paths, feature lists, model hyperparameters)
- **`data.py`** — Data loading and ZIP extraction for both datasets
- **`preprocessing.py`** — Custom transformers (`SafeOrdinalToNonNegative`) and preprocessing pipeline builders
- **`models.py`** — TabTransformer architecture and data preparation utilities
- **`train_evaluate.py`** — Cross-validation, model training, and evaluation logic
- **`plots.py`** — Visualization utilities for confusion matrices, training curves, and comparisons
- **`pipeline.py`** — Inference pipelines for new customer predictions

## Running the Training Script

### Prerequisites

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Upload/place the two required ZIP datasets in the project directory:
   - `archive (1).zip` (JanataHack Customer Segmentation)
   - `bank+marketing.zip` (UCI Bank Marketing)

### Running the Full Pipeline

```bash
python training.py
```

This will:
1. Extract and load both datasets
2. Create locked 80/20 train/test splits
3. Cross-validate Naive Bayes models (5-fold stratified)
4. Evaluate best models on locked test set
5. Perform error analysis and confidence scoring
6. Train TabTransformer (Dataset A only)
7. Generate visualizations and save all artifacts

### Output Files

After running `training.py`, the following are created under `da4_outputs/`:

**Figures** (`figures/`):
- `A_class_distribution.png` — Dataset A segment distribution
- `B_class_distribution.png` — Dataset B target distribution
- `A_cv_comparison.png` — Cross-validation comparison (Dataset A)
- `B_cv_comparison.png` — Cross-validation comparison (Dataset B)
- `A_confusion_matrix_nb.png` — Naive Bayes confusion matrix (Dataset A)
- `B_confusion_matrix.png` — Naive Bayes confusion matrix (Dataset B)
- `A_confusion_matrix_tabtransformer.png` — TabTransformer confusion matrix (Dataset A)
- `A_tabtransformer_training_curve.png` — TabTransformer training history

**Models** (`models/`):
- `A_[ModelName]_pipeline.joblib` — Fitted Naive Bayes pipeline (Dataset A)
- `B_[ModelName]_pipeline.joblib` — Fitted Naive Bayes pipeline (Dataset B)
- `A_tabtransformer_state_dict.pt` — Saved TabTransformer weights

**Results** (`results/`):
- `A_cv_results.csv` — 5-fold CV metrics (Dataset A)
- `B_cv_results.csv` — 5-fold CV metrics (Dataset B)
- `A_test_predictions.csv` — Test predictions and confidences (Dataset A)
- `B_test_predictions.csv` — Test predictions and confidences (Dataset B)
- `A_error_analysis_nb.csv` — Incorrect predictions with confidence (Dataset A)
- `B_error_analysis.csv` — Incorrect predictions with confidence (Dataset B)
- `A_tabtransformer_training_history.csv` — Training history per epoch

## Using Modular Components

The modules can be imported and used independently:

```python
from src.data import extract_datasets, load_dataset_a
from src.train_evaluate import create_naive_bayes_pipelines, cross_validate_models
from src.pipeline import predict_new_customer_segment

# Load and extract data
train_a_path, bank_b_path = extract_datasets()
df_a = load_dataset_a(train_a_path)

# Create and train models
pipelines_a = create_naive_bayes_pipelines(NUMERIC_A, CATEGORICAL_A)
cv_results_a, fold_details = cross_validate_models(pipelines_a, X_train_a, y_train_a)

# Make predictions on new customer
prediction = predict_new_customer_segment(new_customer_profile, best_pipeline)
```

## Reproduction with the Original Notebook

The original Jupyter Notebook (`DA4_Customer_Segmentation_NaiveBayes_TabTransformer.ipynb`) remains intact with all pre-computed outputs. The modular code produces identical results:

- Same datasets (extracted from identical ZIP files)
- Same random seed (SEED = 42)
- Same train/test split (80/20, stratified)
- Same cross-validation strategy (5-fold stratified with shuffle=True, random_state=42)
- Same preprocessing pipelines (KBinsDiscretizer, OneHotEncoder, etc.)
- Same models and hyperparameters
- Same evaluation metrics and locked test set results

## Important

## Important

The original Jupyter Notebook already contains executed outputs from the DA-4 analysis. The notebook remains unchanged and can be opened directly in GitHub or Colab. The new modular training script produces identical results with the same methodology, datasets, and evaluation metrics.


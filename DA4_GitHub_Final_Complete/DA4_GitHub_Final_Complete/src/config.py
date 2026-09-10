"""
Configuration and constants for DA4 project.
"""

from pathlib import Path
import numpy as np

# Random seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Output directories
OUT = Path("da4_outputs")
OUT_FIGURES = OUT / "figures"
OUT_MODELS = OUT / "models"
OUT_RESULTS = OUT / "results"

# Create output directories
for d in [OUT_FIGURES, OUT_MODELS, OUT_RESULTS]:
    d.mkdir(parents=True, exist_ok=True)

# Data extraction directory
DATA_EXTRACT_DIR = Path("da4_data")
DATA_EXTRACT_DIR.mkdir(exist_ok=True)

# ZIP file search locations
CANDIDATE_DIRS = [".", "/content", "/mnt/user-data/uploads", "/mnt/user-data/outputs"]

# --- Dataset A: Customer Segmentation ---
TARGET_A = "Segmentation"
ID_A = "ID"
NUMERIC_A = ["Age", "Work_Experience", "Family_Size"]
CATEGORICAL_A = ["Gender", "Ever_Married", "Graduated", "Profession", "Spending_Score", "Var_1"]
FEATURES_A = NUMERIC_A + CATEGORICAL_A

# --- Dataset B: Bank Marketing ---
TARGET_B = "y"
NUMERIC_B = ["age", "campaign", "pdays", "previous", "emp.var.rate", "cons.price.idx", 
             "cons.conf.idx", "euribor3m", "nr.employed"]
CATEGORICAL_B = ["job", "marital", "education", "default", "housing", "loan", "contact", 
                 "month", "day_of_week", "poutcome"]
FEATURES_B = NUMERIC_B + CATEGORICAL_B
EXCLUDED_LEAKAGE_B = ["duration"]  # Only known after the call; leaks outcome information

# Cross-validation and model parameters
CV_SPLITS = 5
TEST_SIZE = 0.20

# Scoring metrics for model selection
SCORING = {
    "accuracy": "accuracy",
    "macro_f1": "f1_macro",
    "weighted_f1": "f1_weighted"
}

# TabTransformer hyperparameters
TAB_TRANSFORMER_CONFIG = {
    "embed_dim": 16,
    "n_heads": 4,
    "n_layers": 2,
    "dropout": 0.1,
    "batch_size": 64,
    "n_epochs": 40,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "val_split": 0.15,
}

# Preprocessing parameters
KBINS_N_BINS = 5
IMPUTER_STRATEGY_NUMERIC = "median"
IMPUTER_STRATEGY_CATEGORICAL = "most_frequent"
NAIVE_BAYES_ALPHA = 1.0

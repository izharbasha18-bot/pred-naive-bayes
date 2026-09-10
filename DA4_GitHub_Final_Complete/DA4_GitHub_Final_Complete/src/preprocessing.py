"""
Preprocessing utilities and custom transformers.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, KBinsDiscretizer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from .config import KBINS_N_BINS, IMPUTER_STRATEGY_NUMERIC, IMPUTER_STRATEGY_CATEGORICAL, NAIVE_BAYES_ALPHA


class SafeOrdinalToNonNegative(BaseEstimator, TransformerMixin):
    """
    Encodes categorical features as non-negative integer codes (1..K).
    Unseen categories encountered during transform are mapped to 0.
    This is safe for use with CategoricalNB which requires non-negative integer inputs.
    
    Fit on training data only; unseen categories in test data map to 0.
    """
    
    def fit(self, X, y=None):
        """Fit the encoder on training data."""
        self.enc_ = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.enc_.fit(X)
        self.n_features_in_ = X.shape[1]
        return self
    
    def transform(self, X):
        """Transform data; unseen categories become 0."""
        return self.enc_.transform(X).astype(int) + 1


def build_preprocessors(numeric_cols, categorical_cols):
    """
    Build three preprocessing pipelines for different Naive Bayes variants.
    
    Args:
        numeric_cols (list): Names of numeric feature columns
        categorical_cols (list): Names of categorical feature columns
        
    Returns:
        tuple: (bernoulli_prep, categoricalnb_prep, gaussian_prep)
            - bernoulli_prep: Binarized numeric bins + one-hot categoricals
            - categoricalnb_prep: Ordinal-binned numeric + safe non-negative categorical codes
            - gaussian_prep: Continuous numeric features only (no categoricals)
    """
    
    # --- BernoulliNB: Binarized numeric bins + one-hot categoricals ---
    numeric_bins_onehot = Pipeline([
        ("imputer", SimpleImputer(strategy=IMPUTER_STRATEGY_NUMERIC)),
        ("bins", KBinsDiscretizer(n_bins=KBINS_N_BINS, encode="onehot", strategy="quantile")),
    ])
    category_ohe = Pipeline([
        ("imputer", SimpleImputer(strategy=IMPUTER_STRATEGY_CATEGORICAL)),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    bernoulli_prep = ColumnTransformer([
        ("num_bins", numeric_bins_onehot, numeric_cols),
        ("cat", category_ohe, categorical_cols),
    ])
    
    # --- CategoricalNB: Ordinal-binned numeric + safe non-negative categorical codes ---
    cat_num = Pipeline([
        ("imputer", SimpleImputer(strategy=IMPUTER_STRATEGY_NUMERIC)),
        ("bins", KBinsDiscretizer(n_bins=KBINS_N_BINS, encode="ordinal", strategy="quantile")),
    ])
    cat_cat = Pipeline([
        ("imputer", SimpleImputer(strategy=IMPUTER_STRATEGY_CATEGORICAL)),
        ("safe_ordinal", SafeOrdinalToNonNegative()),
    ])
    categoricalnb_prep = ColumnTransformer([
        ("num", cat_num, numeric_cols),
        ("cat", cat_cat, categorical_cols),
    ])
    
    # --- GaussianNB: Continuous numeric features only ---
    gaussian_prep = ColumnTransformer([
        ("num", SimpleImputer(strategy=IMPUTER_STRATEGY_NUMERIC), numeric_cols),
    ])
    
    return bernoulli_prep, categoricalnb_prep, gaussian_prep

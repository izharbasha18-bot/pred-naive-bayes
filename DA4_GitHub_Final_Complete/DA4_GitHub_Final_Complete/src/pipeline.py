"""
Pipeline utility functions for prediction and model management.
"""

import pandas as pd
import numpy as np
from .config import NUMERIC_A, CATEGORICAL_A, NUMERIC_B, CATEGORICAL_B


def predict_new_customer_segment(customer_profile, best_nb_pipeline, 
                                  label_enc=None, use_tabtransformer=False, 
                                  tabtransformer_model=None, device=None,
                                  cat_vocab=None, num_scaler=None, num_imputer=None):
    """
    Predict customer segment for a new profile (Dataset A).
    
    Args:
        customer_profile (dict): Customer feature dictionary (must include all FEATURES_A)
        best_nb_pipeline: Fitted Naive Bayes pipeline
        label_enc: LabelEncoder for classes (used if not None)
        use_tabtransformer (bool): Use TabTransformer instead of Naive Bayes
        tabtransformer_model: Fitted TabTransformer model
        device: torch.device for TabTransformer
        cat_vocab (dict): Categorical vocabulary for TabTransformer
        num_scaler: StandardScaler for numeric features (TabTransformer)
        num_imputer: SimpleImputer for numeric features (TabTransformer)
        
    Returns:
        dict: {
            "model_used": str,
            "predicted_segment": str,
            "posterior_distribution": dict,
            "confidence": float,
            "review_recommendation": str
        }
    """
    missing = [c for c in NUMERIC_A + CATEGORICAL_A if c not in customer_profile]
    if missing:
        raise ValueError(f"Missing mandatory fields: {missing}")
    
    row = pd.DataFrame([customer_profile], columns=NUMERIC_A + CATEGORICAL_A)
    
    if use_tabtransformer and tabtransformer_model is not None:
        # Use TabTransformer
        import torch
        from .models import encode_categoricals
        
        cat_codes = encode_categoricals(row, cat_vocab)
        num_vals = num_scaler.transform(num_imputer.transform(row[NUMERIC_A])).astype(np.float32)
        tabtransformer_model.eval()
        with torch.no_grad():
            logits = tabtransformer_model(
                torch.tensor(cat_codes, dtype=torch.long).to(device),
                torch.tensor(num_vals, dtype=torch.float32).to(device),
            )
            p = torch.softmax(logits, dim=1).cpu().numpy()[0]
        classes = label_enc.classes_
        model_used = "TabTransformer"
    else:
        # Use Naive Bayes
        p = best_nb_pipeline.predict_proba(row)[0]
        classes = best_nb_pipeline.classes_
        model_used = best_nb_pipeline.named_steps["model"].__class__.__name__
    
    distribution = {str(c): float(v) for c, v in zip(classes, p)}
    confidence = float(p.max())
    predicted = str(classes[int(np.argmax(p))])
    
    if confidence >= 0.60:
        review = "high_confidence"
    elif confidence >= 0.40:
        review = "moderate_confidence_review"
    else:
        review = "low_confidence_manual_review"
    
    return {
        "model_used": model_used,
        "predicted_segment": predicted,
        "posterior_distribution": distribution,
        "confidence": confidence,
        "review_recommendation": review,
    }


def predict_term_deposit_response(customer_profile, best_nb_pipeline):
    """
    Predict term deposit subscription for a new customer (Dataset B).
    
    Args:
        customer_profile (dict): Customer feature dictionary (must include all FEATURES_B, excludes 'duration')
        best_nb_pipeline: Fitted Naive Bayes pipeline
        
    Returns:
        dict: {
            "model_used": str,
            "predicted_response": str,
            "posterior_distribution": dict,
            "confidence": float,
            "review_recommendation": str
        }
    """
    missing = [c for c in NUMERIC_B + CATEGORICAL_B if c not in customer_profile]
    if missing:
        raise ValueError(f"Missing mandatory fields: {missing}")
    
    row = pd.DataFrame([customer_profile], columns=NUMERIC_B + CATEGORICAL_B)
    p = best_nb_pipeline.predict_proba(row)[0]
    classes = best_nb_pipeline.classes_
    
    distribution = {str(c): float(v) for c, v in zip(classes, p)}
    confidence = float(p.max())
    predicted = str(classes[int(np.argmax(p))])
    
    if confidence >= 0.60:
        review = "high_confidence"
    elif confidence >= 0.40:
        review = "moderate_confidence_review"
    else:
        review = "low_confidence_manual_review"
    
    return {
        "model_used": best_nb_pipeline.named_steps["model"].__class__.__name__,
        "predicted_response": predicted,
        "posterior_distribution": distribution,
        "confidence": confidence,
        "review_recommendation": review,
    }

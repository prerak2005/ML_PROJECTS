"""
Utility functions for the Titanic Survival Prediction project.
Provides reusable helpers for data loading, preprocessing, feature engineering,
and model evaluation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
from sklearn.model_selection import cross_val_score


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_data(filepath="data/titanic.csv"):
    """Load the Titanic dataset and return a DataFrame."""
    df = pd.read_csv(filepath)
    return df
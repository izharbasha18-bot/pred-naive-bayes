"""
Plotting utilities for visualization of results.
"""

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from .config import OUT_FIGURES


def plot_class_distribution(data, column, title, filename):
    """
    Plot bar chart of class distribution.
    
    Args:
        data (pd.DataFrame or pd.Series): Data to plot
        column (str): Column name (if data is DataFrame)
        title (str): Plot title
        filename (str): Filename to save plot
    """
    if isinstance(data, dict) or hasattr(data, 'value_counts'):
        # It's a Series or dict-like
        if hasattr(data, 'value_counts'):
            plot_data = data.value_counts()
        else:
            plot_data = data
    else:
        plot_data = data[column].value_counts()
    
    ax = plot_data.sort_index().plot(kind="bar", title=title, figsize=(8, 5))
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUT_FIGURES / filename, dpi=150)
    plt.show()


def plot_cv_results(cv_results, metric_col, title, filename, err_col=None):
    """
    Plot cross-validation results bar chart.
    
    Args:
        cv_results (pd.DataFrame): CV results with 'model' column and metric columns
        metric_col (str): Column name for the metric to plot
        title (str): Plot title
        filename (str): Filename to save plot
        err_col (str, optional): Column name for error bars (typically SD)
    """
    ax = cv_results.set_index("model")[[metric_col]].plot(
        kind="bar",
        yerr=cv_results.set_index("model")[[err_col]] if err_col else None,
        legend=False,
        title=title,
        figsize=(10, 5)
    )
    ax.set_ylabel(metric_col)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_FIGURES / filename, dpi=150)
    plt.show()


def plot_confusion_matrix(y_test, y_pred, title, filename, cmap="Blues"):
    """
    Plot confusion matrix.
    
    Args:
        y_test: True test labels
        y_pred: Predicted labels
        title (str): Plot title
        filename (str): Filename to save plot
        cmap (str): Colormap name
    """
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap=cmap, xticks_rotation=45)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUT_FIGURES / filename, dpi=150)
    plt.show()


def plot_training_history(history_df, title, filename):
    """
    Plot TabTransformer training history (train loss and validation macro F1).
    
    Args:
        history_df (pd.DataFrame): DataFrame with 'epoch', 'train_loss', 'val_macro_f1' columns
        title (str): Plot title
        filename (str): Filename to save plot
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.plot(history_df["epoch"], history_df["train_loss"], 
             color="tab:red", label="Train loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss", color="tab:red")
    ax1.tick_params(axis='y', labelcolor='tab:red')
    
    ax2 = ax1.twinx()
    ax2.plot(history_df["epoch"], history_df["val_macro_f1"], 
             color="tab:blue", label="Val macro F1", linewidth=2)
    ax2.set_ylabel("Validation macro F1", color="tab:blue")
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    plt.title(title)
    fig.tight_layout()
    plt.savefig(OUT_FIGURES / filename, dpi=150)
    plt.show()

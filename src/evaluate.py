import matplotlib

matplotlib.use('Agg')  # save plots to files, no display window needed

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src.config import CLASSES


def print_report(y_true, y_pred):
    print(classification_report(y_true, y_pred, zero_division=0))
    # Macro F1 is the average F1 across all classes, where every class gets an equal vote.
    print(f"Test macro F1: {f1_score(y_true, y_pred, average='macro'):.3f}")


def plot_confusion_matrix(y_true, y_pred, path):
    cm_raw = confusion_matrix(y_true, y_pred, labels=CLASSES)
    cm_df = pd.DataFrame(cm_raw, index=CLASSES, columns=CLASSES)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=True, square=True)
    plt.title('Multiclass Confusion Matrix', fontsize=14, pad=15)
    plt.xlabel('Predicted Class', fontsize=12, labelpad=10)
    plt.ylabel('Actual Class', fontsize=12, labelpad=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def feature_importances(model, features):
    """Which features the forest relied on most (shares of total sorting work, sum to 1)."""
    return pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

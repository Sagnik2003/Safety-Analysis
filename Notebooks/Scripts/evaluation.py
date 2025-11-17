import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix
)
import seaborn as sns

class evaluation_methods:
    def __init__(self, y_true, y_pred, y_prob):
        """
        y_true : true labels
        y_pred : predicted class labels
        y_prob : predicted probabilities (n_samples, n_classes)
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob

    def compute_metrics(self):
        metrics = {
            "Accuracy": accuracy_score(self.y_true, self.y_pred),
            "Precision": precision_score(self.y_true, self.y_pred, average='weighted', zero_division=0),
            "Recall": recall_score(self.y_true, self.y_pred, average='weighted', zero_division=0),
            "F1 Score": f1_score(self.y_true, self.y_pred, average='weighted', zero_division=0),
            "ROC-AUC": roc_auc_score(self.y_true, self.y_prob, multi_class='ovr'),
            "Log Loss": log_loss(self.y_true, self.y_prob)
        }
        return pd.DataFrame([metrics])

    def save_confusion_matrix(self, filename="confusion_matrix.png"):
        cm = confusion_matrix(self.y_true, self.y_pred)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

        return filename

    def plot_roc_curve(self, filename="roc_curve.png"):
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc

        classes = np.unique(self.y_true)

        # Binarize labels for multiclass ROC computation
        y_true_bin = label_binarize(self.y_true, classes=classes)

        plt.figure(figsize=(7, 6))

        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], self.y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"Class {cls} (AUC = {roc_auc:.2f})")

        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.title("ROC Curve (Multiclass)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

        return filename

if __name__ == "__main__":
    pass
    # # Example usage
    # y_true = np.array([0, 1, 1, 0, 1])
    # y_pred = np.array([0, 1, 0, 0, 1])
    # y_prob = np.array([0.1, 0.9, 0.4, 0.3, 0.8])

    # evaluator = evaluation_methods(y_true, y_pred, y_prob)
    # metrics_df = evaluator.compute_metrics()
    # print(metrics_df)

    # cm_file = evaluator.save_confusion_matrix()
    # print(f"Confusion matrix saved to {cm_file}")

    # roc_file = evaluator.plot_roc_curve()
    # print(f"ROC curve saved to {roc_file}")
    
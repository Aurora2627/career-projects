import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, average_precision_score

def classification_metrics(y, predictions, probabilities):
    labels = list(range(6))
    one_hot = np.eye(6)[y]
    return {"accuracy": float(accuracy_score(y, predictions)), "macro_f1": float(f1_score(y, predictions, labels=labels, average="macro", zero_division=0)), "macro_average_precision": float(average_precision_score(one_hot, probabilities, average="macro")), "confusion_matrix": confusion_matrix(y, predictions, labels=labels).tolist(), "per_class": classification_report(y, predictions, labels=labels, output_dict=True, zero_division=0)}

from sklearn.metrics import roc_auc_score, f1_score
import numpy as np

def calc_auc_score(y_true, y_prob):
    try:
        return roc_auc_score(y_true, y_prob)
    except ValueError:
        return 0.5

def calc_f1_score(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob > threshold).astype(int)
    return f1_score(y_true, y_pred)

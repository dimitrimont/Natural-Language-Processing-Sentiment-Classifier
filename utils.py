import json
import os
from typing import Dict, List, Iterable, Tuple
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix

# ---------- IO ----------
def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def save_json(d: Dict, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(d, f, indent=2)

# ---------- Metrics ----------
def compute_scores(y_true, y_pred, y_proba) -> Dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "report": classification_report(y_true, y_pred, digits=3)
    }

# ---------- Visualization ----------
def save_confusion_matrix(y_true, y_pred, out_path: str, title: str = "Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    ensure_dir(os.path.dirname(out_path))
    plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    for (i, j), v in [((i, j), cm[i, j]) for i in range(cm.shape[0]) for j in range(cm.shape[1])]:
        plt.text(j, i, str(v), ha='center', va='center')
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

# ---------- Explainability for Track A ----------
def top_ngrams_from_tfidf_lr(pipeline, k: int = 25) -> List[Tuple[str, str, float]]:
    """
    Returns rows of (type, ngram, coef) where type in {"positive","negative"}.
    """
    vec = pipeline.named_steps['tfidf']
    clf = pipeline.named_steps['clf']
    feature_names = np.array(vec.get_feature_names_out())
    coefs = clf.coef_[0]
    top_pos_idx = np.argsort(coefs)[-k:][::-1]
    top_neg_idx = np.argsort(coefs)[:k]
    rows = []
    rows += [("positive", feature_names[i], float(coefs[i])) for i in top_pos_idx]
    rows += [("negative", feature_names[i], float(coefs[i])) for i in top_neg_idx]
    return rows

def save_top_ngrams_csv(rows: Iterable[Tuple[str, str, float]], out_path: str):
    import csv
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("type","ngram","coef"))
        for r in rows:
            w.writerow(r)

# ---------- Error analysis ----------
def save_errors_csv(texts: List[str], y_true, y_pred, y_proba, out_path: str, max_len: int = 500):
    import csv
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("true","pred","prob_pos","text_snippet"))
        for t, yt, yp, p in zip(texts, y_true, y_pred, y_proba):
            if yt != yp:
                w.writerow((int(yt), int(yp), float(p), t.replace("\n"," ")[:max_len]))

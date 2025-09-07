import os
import argparse
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

from data import load_movie_reviews
from utils import compute_scores, save_json, save_confusion_matrix, ensure_dir, save_errors_csv

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                    help="Sentence-Transformers model name")
    ap.add_argument("--batch_size", type=int, default=64, help="Encoding batch size")
    ap.add_argument("--C", type=float, default=2.0, help="LogReg inverse regularization strength")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--outdir", type=str, default="outputs/track_b", help="Where to write artifacts")
    return ap.parse_args()

def main():
    args = get_args()
    ensure_dir(args.outdir)

    # 1) Load the same split as Track A (same seed for fair comparison)
    X_train, X_test, y_train, y_test = load_movie_reviews(seed=args.seed)

    # 2) Encode texts into dense semantic vectors with a pretrained MiniLM encoder
    model = SentenceTransformer(args.encoder)
    Xtr = model.encode(X_train, batch_size=args.batch_size, convert_to_numpy=True, show_progress_bar=True)
    Xte = model.encode(X_test,  batch_size=args.batch_size, convert_to_numpy=True, show_progress_bar=True)
    # Xtr, Xte shapes ~ (N, 384) for all-MiniLM-L6-v2

    # 3) Train a simple, strong linear head on top of embeddings
    clf = LogisticRegression(max_iter=300, C=args.C, solver='liblinear', random_state=args.seed)
    clf.fit(Xtr, y_train)

    # 4) Evaluate
    preds = clf.predict(Xte)
    proba = clf.predict_proba(Xte)[:, 1]
    scores = compute_scores(y_test, preds, proba)

    # 5) Save metrics & plots
    save_json(scores, os.path.join(args.outdir, "metrics.json"))
    print("=== Track B (MiniLM embeddings + LR) ===")
    print(scores["report"])
    print({k: v for k, v in scores.items() if k != "report"})

    save_confusion_matrix(y_test, preds, os.path.join(args.outdir, "confusion_matrix.png"),
                          title="MiniLM + LR")

    # Also save misclassifications for side-by-side error analysis (like Track A)
    save_errors_csv(X_test, y_test, preds, proba, os.path.join(args.outdir, "errors.csv"))

    # 6) Save the classifier and the encoder name used (encoder is downloaded at runtime)
    joblib.dump(clf, os.path.join(args.outdir, "model.joblib"))
    with open(os.path.join(args.outdir, "encoder.txt"), "w") as f:
        f.write(args.encoder + "\n")

if __name__ == "__main__":
    main()

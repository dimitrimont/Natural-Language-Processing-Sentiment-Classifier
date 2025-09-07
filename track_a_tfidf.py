import os
import argparse
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from data import load_movie_reviews
from utils import (
    compute_scores, save_json, save_confusion_matrix,
    top_ngrams_from_tfidf_lr, save_top_ngrams_csv,
    save_errors_csv, ensure_dir
)

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ngram_min", type=int, default=1)
    ap.add_argument("--ngram_max", type=int, default=2)
    ap.add_argument("--min_df", type=int, default=2)
    ap.add_argument("--max_df", type=float, default=0.9)
    ap.add_argument("--C", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="outputs/track_a")
    return ap.parse_args()

def main():
    args = get_args()
    ensure_dir(args.outdir)

    X_train, X_test, y_train, y_test = load_movie_reviews(seed=args.seed)

    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(args.ngram_min, args.ngram_max),
            min_df=args.min_df,
            max_df=args.max_df
        )),
        ('clf', LogisticRegression(max_iter=300, C=args.C, solver='liblinear', random_state=args.seed))
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:,1]

    scores = compute_scores(y_test, preds, proba)
    save_json(scores, os.path.join(args.outdir, "metrics.json"))
    print("=== Track A (TF-IDF + LR) ===")
    print(scores["report"])
    print({k:v for k,v in scores.items() if k != "report"})

    save_confusion_matrix(y_test, preds, os.path.join(args.outdir, "confusion_matrix.png"), title="TF-IDF + LR")

    rows = top_ngrams_from_tfidf_lr(pipe, k=25)
    save_top_ngrams_csv(rows, os.path.join(args.outdir, "top_features.csv"))

    save_errors_csv(X_test, y_test, preds, proba, os.path.join(args.outdir, "errors.csv"))

    joblib.dump(pipe, os.path.join(args.outdir, "model.joblib"))

if __name__ == "__main__":
    main()

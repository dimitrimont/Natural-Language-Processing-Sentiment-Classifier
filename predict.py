#!/usr/bin/env python
import os
import sys
import argparse
import numpy as np

# Output formatting
import pandas as pd

# ---------- Default artifact paths ----------
A_DIR = "outputs/track_a"     # TF-IDF + LR (Pipeline)
B_DIR = "outputs/track_b"     # MiniLM + LR (encoder.txt + joblib)
S_DIR = "outputs/stretch"     # DistilBERT fine-tuned (HF model dir)
S_MODEL_DIR = os.path.join(S_DIR, "model")

# ---------- Utilities ----------
def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def read_texts_from_args_or_stdin(args) -> list[str]:
    if args.text is not None:
        return [args.text]
    if args.file is not None:
        with open(args.file, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        return lines
    # Interactive stdin mode
    print("Enter text (one per line). Press Ctrl-D (mac/linux) or Ctrl-Z (win) to run.\n", file=sys.stderr)
    lines = []
    try:
        for ln in sys.stdin:
            if ln.strip():
                lines.append(ln.strip())
    except KeyboardInterrupt:
        pass
    return lines

def pretty_print_single(texts, probs, labels, threshold):
    for t, p, y in zip(texts, probs, labels):
        print(f"[{int(y)} | prob_pos={p:.3f} | thresh={threshold:.2f}]  {t}")

def snip(s: str, n: int = 180) -> str:
    return s.replace("\n", " ")[:n]

# ---------- Track A (TF-IDF + LR) ----------
def predict_track_a(texts: list[str], threshold: float = 0.5):
    import joblib
    model_path = os.path.join(A_DIR, "model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Track A model not found at {model_path}. Run: python track_a_tfidf.py")
    pipe = joblib.load(model_path)
    proba = pipe.predict_proba(texts)[:, 1]
    labels = (proba >= threshold).astype(int)
    return proba, labels

# ---------- Track B (MiniLM embeddings + LR) ----------
def predict_track_b(texts: list[str], threshold: float = 0.5, batch_size: int = 64):
    from sentence_transformers import SentenceTransformer
    import joblib

    enc_path = os.path.join(B_DIR, "encoder.txt")
    clf_path = os.path.join(B_DIR, "model.joblib")
    if not (os.path.exists(enc_path) and os.path.exists(clf_path)):
        raise FileNotFoundError(f"Track B artifacts missing at {B_DIR}. Run: python track_b_sbert.py")

    with open(enc_path) as f:
        enc_name = f.read().strip()
    enc = SentenceTransformer(enc_name)
    X = enc.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)

    clf = joblib.load(clf_path)
    proba = clf.predict_proba(X)[:, 1]
    labels = (proba >= threshold).astype(int)
    return proba, labels

# ---------- Stretch (DistilBERT fine-tuned) ----------
def predict_stretch(texts: list[str], threshold: float = 0.5, batch_size: int = 32, max_len: int = 256):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    if not os.path.isdir(S_MODEL_DIR):
        raise FileNotFoundError(f"DistilBERT model not found at {S_MODEL_DIR}. Run: python stretch_distilbert.py")

    tok = AutoTokenizer.from_pretrained(S_MODEL_DIR, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(S_MODEL_DIR)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() else "cpu")
    )
    mdl.to(device)
    mdl.eval()

    all_probs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tok(batch, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = mdl(**enc)
            probs = torch.softmax(out.logits, dim=-1)[:, 1].detach().cpu().numpy()
            all_probs.append(probs)

    proba = np.concatenate(all_probs, axis=0)
    labels = (proba >= threshold).astype(int)
    return proba, labels

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Predict sentiment with Track A/B or DistilBERT; or compare all.")
    ap.add_argument("--track", choices=["a", "b", "stretch"], default="stretch",
                    help="Single-track prediction: a=TFIDF+LR, b=MiniLM+LR, stretch=DistilBERT fine-tuned")
    ap.add_argument("--compare", action="store_true",
                    help="Compare multiple tracks side-by-side (overrides --track).")
    ap.add_argument("--tracks", type=str, default="a,b,stretch",
                    help="Comma-separated list when using --compare (e.g., 'a,b' or 'b,stretch').")
    ap.add_argument("--text", type=str, default=None, help="Single input text to classify")
    ap.add_argument("--file", type=str, default=None, help="Path to a file with one text per line")
    ap.add_argument("--threshold", type=float, default=0.5, help="Positive class threshold")
    ap.add_argument("--batch_size", type=int, default=32, help="Batch size for BERT/SBERT")
    ap.add_argument("--max_len", type=int, default=256, help="Max token length for DistilBERT")
    args = ap.parse_args()

    texts = read_texts_from_args_or_stdin(args)
    if not texts:
        print("No input text. Use --text, --file, or pipe lines via stdin.", file=sys.stderr)
        sys.exit(1)

    if not args.compare:
        # Single-track mode
        if args.track == "a":
            proba, labels = predict_track_a(texts, threshold=args.threshold)
            model_name = "Track A (TF-IDF + LR)"
        elif args.track == "b":
            proba, labels = predict_track_b(texts, threshold=args.threshold, batch_size=args.batch_size)
            model_name = "Track B (MiniLM + LR)"
        else:
            proba, labels = predict_stretch(texts, threshold=args.threshold,
                                            batch_size=args.batch_size, max_len=args.max_len)
            model_name = "Stretch (DistilBERT fine-tuned)"

        print(f"=== Predictions — {model_name} ===")
        pretty_print_single(texts, proba, labels, args.threshold)
        return

    # ---------------- Comparison mode ----------------
    wanted = [t.strip().lower() for t in args.tracks.split(",") if t.strip()]
    valid = {"a", "b", "stretch"}
    tracks = [t for t in wanted if t in valid]
    if not tracks:
        print("No valid tracks in --tracks. Use any of: a,b,stretch", file=sys.stderr)
        sys.exit(2)

    # Collect predictions per track, handling missing models gracefully
    results = {"text": [snip(t) for t in texts]}
    errors = []

    if "a" in tracks:
        try:
            p, y = predict_track_a(texts, threshold=args.threshold)
            results["A_prob"] = p
            results["A_pred"] = y
        except Exception as e:
            errors.append(f"Track A unavailable: {e}")
            results["A_prob"] = [np.nan]*len(texts)
            results["A_pred"] = [np.nan]*len(texts)

    if "b" in tracks:
        try:
            p, y = predict_track_b(texts, threshold=args.threshold, batch_size=args.batch_size)
            results["B_prob"] = p
            results["B_pred"] = y
        except Exception as e:
            errors.append(f"Track B unavailable: {e}")
            results["B_prob"] = [np.nan]*len(texts)
            results["B_pred"] = [np.nan]*len(texts)

    if "stretch" in tracks:
        try:
            p, y = predict_stretch(texts, threshold=args.threshold,
                                   batch_size=args.batch_size, max_len=args.max_len)
            results["S_prob"] = p
            results["S_pred"] = y
        except Exception as e:
            errors.append(f"Stretch unavailable: {e}")
            results["S_prob"] = [np.nan]*len(texts)
            results["S_pred"] = [np.nan]*len(texts)

    df = pd.DataFrame(results)
    # Order columns nicely if all three were requested
    col_order = ["text"]
    if "a" in tracks:       col_order += ["A_prob", "A_pred"]
    if "b" in tracks:       col_order += ["B_prob", "B_pred"]
    if "stretch" in tracks: col_order += ["S_prob", "S_pred"]
    df = df[col_order]

    print(f"=== Side-by-side comparison (threshold={args.threshold:.2f}) ===")
    print(df.to_string(index=False))

    if errors:
        print("\nNotes:", *errors, sep="\n- ")

if __name__ == "__main__":
    main()

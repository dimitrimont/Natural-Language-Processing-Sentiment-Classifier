import os
import argparse
import numpy as np
import pandas as pd

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
)
import evaluate

from data import load_movie_reviews
from utils import (
    ensure_dir,
    save_json,
    save_confusion_matrix,
    save_errors_csv,
    compute_scores,
)

def softmax_np(logits: np.ndarray) -> np.ndarray:
    """Stable softmax over last dimension."""
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="distilbert-base-uncased",
                    help="HF model name to fine-tune")
    ap.add_argument("--epochs", type=int, default=2, help="Number of fine-tuning epochs")
    ap.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    ap.add_argument("--train_bs", type=int, default=16, help="Per-device train batch size")
    ap.add_argument("--eval_bs", type=int, default=32, help="Per-device eval batch size")
    ap.add_argument("--max_len", type=int, default=256, help="Max sequence length (truncate)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--outdir", type=str, default="outputs/stretch", help="Where to write artifacts")
    return ap.parse_args()

def main():
    args = get_args()
    set_seed(args.seed)
    ensure_dir(args.outdir)

    # 1) Load same split as previous tracks (for fair comparison)
    X_train, X_test, y_train, y_test = load_movie_reviews(seed=args.seed)

    train_df = pd.DataFrame({"text": X_train, "label": y_train})
    test_df  = pd.DataFrame({"text": X_test,  "label": y_test})
    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    test_ds  = Dataset.from_pandas(test_df,  preserve_index=False)

    # 2) Tokenize
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    def tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_len,
        )

    train_tok = train_ds.map(tok, batched=True, remove_columns=["text"])
    test_tok  = test_ds.map(tok, batched=True, remove_columns=["text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 3) Model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2,
    )

    # 4) Metrics for Trainer (accuracy & F1 during explicit eval)
    metric_acc = evaluate.load("accuracy")
    metric_f1  = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": metric_acc.compute(predictions=preds, references=labels)["accuracy"],
            "f1": metric_f1.compute(predictions=preds, references=labels)["f1"],
        }

    # 5) Training args (back-compat: no evaluation_strategy/save_strategy/report_to)
    out_model_dir = os.path.join(args.outdir, "model")
    training_args = TrainingArguments(
        output_dir=out_model_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=50,
        seed=args.seed,
        # Older versions handle eval when you call trainer.evaluate()/predict() explicitly.
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=test_tok,      # used when we call trainer.evaluate()/predict()
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 6) Train & evaluate (explicit calls for old versions)
    trainer.train()
    eval_res = trainer.evaluate()
    save_json(eval_res, os.path.join(args.outdir, "metrics_trainer.json"))
    print("Trainer eval (accuracy/F1):", eval_res)

    # 7) Predict on test set for full metrics + artifacts
    preds_out = trainer.predict(test_tok)
    logits = preds_out.predictions
    labels = preds_out.label_ids
    preds = np.argmax(logits, axis=-1)
    prob_pos = softmax_np(logits)[:, 1]

    # Full metric bundle (Accuracy, F1, ROC-AUC, report)
    scores = compute_scores(labels, preds, prob_pos)
    save_json(scores, os.path.join(args.outdir, "metrics.json"))
    print("=== DistilBERT (fine-tuned) ===")
    print(scores["report"])
    print({k: v for k, v in scores.items() if k != "report"})

    # 8) Artifacts to match other tracks
    save_confusion_matrix(labels, preds, os.path.join(args.outdir, "confusion_matrix.png"),
                          title="DistilBERT (fine-tuned)")
    save_errors_csv(X_test, labels, preds, prob_pos, os.path.join(args.outdir, "errors.csv"))

    # 9) Save model + tokenizer (HF format)
    ensure_dir(out_model_dir)
    model.save_pretrained(out_model_dir)
    tokenizer.save_pretrained(out_model_dir)

if __name__ == "__main__":
    main()

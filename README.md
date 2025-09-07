# Natural Language Processing Sentiment Classifier

This project implements and compares two approaches to sentiment classification on text data:  
- Track A: Classical statistical pipeline (TF-IDF + Logistic Regression)  
- Track B: Transformer-based embeddings (MiniLM + Logistic Regression)  

The project explores the trade-off between interpretability vs performance in NLP classification and provides reproducible scripts for training, evaluation, and prediction.

---

## Features

- Dual-track pipelines  
  - Track A: TF-IDF + Logistic Regression (interpretable, lightweight)  
  - Track B: MiniLM embeddings + Logistic Regression (context-aware, higher accuracy)  
- Evaluation metrics: Accuracy, F1-score, confusion matrix, error analysis  
- Jupyter Notebook integration for exploration and visualization  
- Optional DistilBERT fine-tuning (`stretch_distilbert.py`) for extension  

---

## Installation

bash
# Clone the repository
git clone https://github.com/yourusername/NLP-Sentiment-Classifier.git
cd NLP-Sentiment-Classifier


## Dependencies include:

scikit-learn
pandas, numpy
matplotlib
transformers
torch





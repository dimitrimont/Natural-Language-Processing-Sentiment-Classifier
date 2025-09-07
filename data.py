import random
from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk

def _ensure_movie_reviews():
    try:
        from nltk.corpus import movie_reviews  # noqa: F401
    except LookupError:
        nltk.download('movie_reviews')
    from nltk.corpus import movie_reviews
    return movie_reviews

def load_movie_reviews(test_size: float = 0.2, seed: int = 42) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
    
    """
    Returns: X_train, X_test, y_train, y_test
    """

    random.seed(seed); np.random.seed(seed)

    mr = _ensure_movie_reviews()

    docs, labels = [], []
    for cat in mr.categories():  # 'pos', 'neg'
        for fid in mr.fileids(cat):
            docs.append(mr.raw(fid))
            labels.append(1 if cat == 'pos' else 0)

    df = pd.DataFrame({'text': docs, 'label': labels}).sample(frac=1, random_state=seed).reset_index(drop=True)
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'].tolist(), df['label'].values,
        test_size=test_size, random_state=seed, stratify=df['label']
    )
    return X_train, X_test, y_train, y_test

import numpy as np
import pandas as pd
import re
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('/Users/erwinqodariyah/Hobby/name-that-llm/data/train.csv')
models = sorted(train['model'].unique())
model_to_idx = {m: i for i, m in enumerate(models)}
y = train['model'].map(model_to_idx).values
groups = train['prompt_id'].values

def eval_metrics(y_true, oof_probs, groups, name=""):
    preds = oof_probs.argmax(axis=1)
    ba_full = balanced_accuracy_score(y_true, preds)
    
    # Also simulate skewed test priors (Dirichlet(alpha=0.8) model inclusion probabilities per prompt)
    rng = np.random.RandomState(42)
    skewed_bas = []
    unique_prompts = np.unique(groups)
    for _ in range(50):
        # Sample model keep-probabilities (between 0.25 and 1.0)
        keep_prob = rng.uniform(0.2, 1.0, size=len(models))
        mask = np.array([rng.rand() < keep_prob[y_true[i]] for i in range(len(y_true))])
        skewed_bas.append(balanced_accuracy_score(y_true[mask], preds[mask]))
    print(f"[{name:45s}] OOF Balanced Acc: {ba_full:.5f} | Simulated Skewed-Test BA: {np.mean(skewed_bas):.5f}")
    return ba_full

gkf = GroupKFold(n_splits=5)

# 1. Simple Word TF-IDF Baseline (Lowercase, standard token pattern)
oof_w = np.zeros((len(train), len(models)))
for fold, (tr_idx, va_idx) in enumerate(gkf.split(train, y, groups)):
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=25000, sublinear_tf=True)
    X_tr = vec.fit_transform(train['text'].iloc[tr_idx])
    X_va = vec.transform(train['text'].iloc[va_idx])
    clf = LogisticRegression(C=2.0, max_iter=1000)
    clf.fit(X_tr, y[tr_idx])
    oof_w[va_idx] = clf.predict_proba(X_va)

eval_metrics(y, oof_w, groups, "Exp01a: Standard Word TF-IDF (1,2) LogReg")

# 2. Case-Sensitive + Exact Punctuation Word & Char_wb + Char TF-IDF
oof_multi = np.zeros((len(train), len(models)))
for fold, (tr_idx, va_idx) in enumerate(gkf.split(train, y, groups)):
    # Word + Punctuation (Case-sensitive, preserving curly vs straight quotes & em-dashes!)
    v_word_cs = TfidfVectorizer(
        lowercase=False,
        token_pattern=r"(?u)\b\w+\b|[.,!?;:—–\-\'\"\’\“\”\(\)\[\]]",
        ngram_range=(1, 3),
        min_df=2,
        max_features=60000,
        sublinear_tf=True
    )
    # Character n-grams (capturing punctuation spacing like ' — ' vs '—', '’s' vs "'s", suffixes)
    v_char = TfidfVectorizer(
        lowercase=False,
        analyzer='char',
        ngram_range=(2, 5),
        min_df=3,
        max_features=80000,
        sublinear_tf=True
    )
    X_tr = hstack([
        v_word_cs.fit_transform(train['text'].iloc[tr_idx]),
        v_char.fit_transform(train['text'].iloc[tr_idx])
    ])
    X_va = hstack([
        v_word_cs.transform(train['text'].iloc[va_idx]),
        v_char.transform(train['text'].iloc[va_idx])
    ])
    clf = LogisticRegression(C=3.0, max_iter=1000)
    clf.fit(X_tr, y[tr_idx])
    oof_multi[va_idx] = clf.predict_proba(X_va)

eval_metrics(y, oof_multi, groups, "Exp01b: Case-Sens Word+Punct(1,3) + Char(2,5)")

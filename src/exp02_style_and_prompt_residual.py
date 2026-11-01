import numpy as np
import pandas as pd
import re
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax, log_softmax
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('/Users/erwinqodariyah/Hobby/name-that-llm/data/train.csv')
models = sorted(train['model'].unique())
model_to_idx = {m: i for i, m in enumerate(models)}
y = train['model'].map(model_to_idx).values
groups = train['prompt_id'].values

def reconstruct_prompt_clusters(texts, target_cluster_size=18.0, distance_threshold=0.72):
    """
    Unsupervised prompt cluster reconstruction using content TF-IDF cosine similarity.
    Does NOT use prompt_id at all (100% test-compatible).
    """
    vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=25000, sublinear_tf=True)
    X_topic = vec.fit_transform(texts)
    sim = cosine_similarity(X_topic)
    dist = np.clip(1.0 - sim, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    
    # Agglomerative average linkage clustering with distance threshold
    clust = AgglomerativeClustering(
        n_clusters=None,
        metric='precomputed',
        linkage='average',
        distance_threshold=distance_threshold
    )
    labels = clust.fit_predict(dist)
    return labels, sim

def center_by_cluster(X, cluster_ids):
    """
    Compute X_residual = X - mean(X within cluster) for dense or sparse matrix.
    """
    if isinstance(X, csr_matrix) or hasattr(X, 'toarray'):
        X_dense = X.toarray() if hasattr(X, 'toarray') else X
    else:
        X_dense = X.copy()
    X_res = np.zeros_like(X_dense)
    for c in np.unique(cluster_ids):
        mask = (cluster_ids == c)
        if mask.sum() > 1:
            X_res[mask] = X_dense[mask] - X_dense[mask].mean(axis=0, keepdims=True)
        else:
            X_res[mask] = X_dense[mask]
    return X_res

def extract_stylometric_features(texts):
    feats = []
    for t in texts:
        n_chars = max(len(t), 1)
        words = t.split()
        n_words = max(len(words), 1)
        sents = [s.strip() for s in re.split(r'[.!?]+', t) if s.strip()]
        n_sents = max(len(sents), 1)
        
        word_lens = [len(w.strip(".,!?;:—–-'\"’“”()[]")) for w in words]
        sent_lens = [len(s.split()) for s in sents]
        
        row = [
            np.log1p(n_chars),
            np.log1p(n_words),
            np.log1p(n_sents),
            n_chars / n_words,
            n_words / n_sents,
            np.std(word_lens) if len(word_lens) > 1 else 0.0,
            np.max(word_lens) if len(word_lens) > 0 else 0.0,
            np.std(sent_lens) if len(sent_lens) > 1 else 0.0,
            np.max(sent_lens) if len(sent_lens) > 0 else 0.0,
            np.min(sent_lens) if len(sent_lens) > 0 else 0.0,
            sum(1 for w in word_lens if w >= 8) / n_words,
            sum(1 for w in word_lens if w <= 3) / n_words,
            len(set(w.lower() for w in words)) / n_words, # TTR
            # Exact Unicode & punctuation binary + rate indicators
            1.0 if '’' in t else 0.0,
            t.count('’') / n_words,
            1.0 if "'" in t else 0.0,
            t.count("'") / n_words,
            1.0 if ('“' in t or '”' in t) else 0.0,
            (t.count('“') + t.count('”')) / n_words,
            1.0 if '"' in t else 0.0,
            t.count('"') / n_words,
            1.0 if '—' in t else 0.0,
            t.count('—') / n_words,
            1.0 if ' — ' in t else 0.0,
            t.count(' — ') / n_words,
            1.0 if re.search(r'\w—\w', t) else 0.0,
            len(re.findall(r'\w—\w', t)) / n_words,
            1.0 if '–' in t else 0.0,
            1.0 if ';' in t else 0.0,
            t.count(';') / n_sents,
            1.0 if ':' in t else 0.0,
            t.count(':') / n_sents,
            1.0 if '(' in t else 0.0,
            t.count('(') / n_sents,
            1.0 if '?' in t else 0.0,
            1.0 if '!' in t else 0.0,
            1.0 if '*' in t else 0.0,
            t.count(',') / n_words,
            t.count(',') / n_sents,
            t.count('-') / n_words,
            1.0 if ', and ' in t else 0.0,
            1.0 if re.search(r'\w, \w+ and ', t) else 0.0,
            1.0 if re.search(r'\d', t) else 0.0,
            len(re.findall(r'\b[A-Z]{2,}\b', t)) / n_words,
            # Sentence starters & discourse markers
            1.0 if re.search(r'(?:^|\.\s+)Yet\b', t) else 0.0,
            1.0 if re.search(r'(?:^|\.\s+)This\b', t) else 0.0,
            1.0 if re.search(r'(?:^|\.\s+)That\b', t) else 0.0,
            1.0 if re.search(r'(?:^|\.\s+)Each\b', t) else 0.0,
            1.0 if re.search(r'(?:^|\.\s+)In\b', t) else 0.0,
            1.0 if re.search(r'(?:^|\.\s+)By\b', t) else 0.0,
            1.0 if re.search(r'(?:^|\.\s+)As\b', t) else 0.0,
            1.0 if re.search(r'(?:^|\.\s+)Instead\b', t) else 0.0,
            1.0 if re.search(r'(?:^|\.\s+)Ultimately\b', t) else 0.0,
            1.0 if re.search(r'(?:^|\.\s+)However\b', t) else 0.0,
            1.0 if re.search(r',\s+while\b', t) else 0.0,
            1.0 if re.search(r',\s+but\b', t) else 0.0,
            1.0 if re.search(r',\s+yet\b', t) else 0.0,
            1.0 if re.search(r',\s+making\b', t) else 0.0,
            1.0 if re.search(r',\s+turning\b', t) else 0.0,
            1.0 if re.search(r'—and\b', t) else 0.0,
            1.0 if re.search(r'—a\b', t) else 0.0,
            1.0 if re.search(r'—that\b', t) else 0.0,
            1.0 if re.search(r"n['’]t\b", t) else 0.0,
            1.0 if re.search(r"\bit['’]s\b", t, re.I) else 0.0,
        ]
        feats.append(row)
    return np.array(feats, dtype=np.float32)

# Check unsupervised prompt clustering quality on validation folds
gkf = GroupKFold(n_splits=5)
X_style_all = extract_stylometric_features(train['text'].tolist())

oof_raw = np.zeros((len(train), len(models)))
oof_assigned = np.zeros((len(train), len(models)))
val_clusters_all = np.zeros(len(train), dtype=int)

for fold, (tr_idx, va_idx) in enumerate(gkf.split(train, y, groups)):
    tr_texts = train['text'].iloc[tr_idx].tolist()
    va_texts = train['text'].iloc[va_idx].tolist()
    
    # Reconstruct clusters purely from text (no prompt_id used!)
    tr_clust, _ = reconstruct_prompt_clusters(tr_texts, distance_threshold=0.75)
    va_clust, va_sim = reconstruct_prompt_clusters(va_texts, distance_threshold=0.75)
    val_clusters_all[va_idx] = va_clust + fold * 1000
    
    # Prompt-invariant vocabulary selection:
    # Keep n-grams that appear in at least 4 distinct reconstructed prompt clusters!
    v_word = TfidfVectorizer(
        lowercase=False,
        token_pattern=r"(?u)\b\w+\b|[.,!?;:—–\-\'\"\’\“\”\(\)\[\]]",
        ngram_range=(1, 3),
        min_df=5,
        max_features=50000,
        sublinear_tf=True
    )
    X_w_tr = v_word.fit_transform(tr_texts)
    X_w_va = v_word.transform(va_texts)
    
    # Compute Model F-stat vs Prompt F-stat per token in tr_idx to weight features by style-purity!
    # A token that occurs across many models within the same prompt is topic noise;
    # a token that occurs for specific models across prompts is style signal!
    X_bin_tr = (X_w_tr > 0).astype(np.float32)
    # Count how many distinct clusters each feature appears in
    n_clust_tr = len(np.unique(tr_clust))
    clust_mat = csr_matrix((np.ones(len(tr_idx)), (tr_clust, np.arange(len(tr_idx)))), shape=(n_clust_tr, len(tr_idx)))
    feat_cluster_df = np.array(((clust_mat @ X_bin_tr) > 0).sum(axis=0)).flatten()
    
    # Compute model chi2 / variance across models
    n_models = len(models)
    model_mat = csr_matrix((np.ones(len(tr_idx)), (y[tr_idx], np.arange(len(tr_idx)))), shape=(n_models, len(tr_idx)))
    model_rates = np.array((model_mat @ X_bin_tr).todense()) / np.maximum(np.bincount(y[tr_idx])[:, None], 1)
    mean_rate = model_rates.mean(axis=0) + 1e-6
    model_var = ((model_rates - mean_rate) ** 2).sum(axis=0) / mean_rate
    
    # Style weight: zero out single-cluster topic words, boost high model_var words
    style_weight = np.where(feat_cluster_df >= 3, np.sqrt(model_var), 0.05 * np.sqrt(model_var))
    
    X_w_tr = X_w_tr.multiply(style_weight)
    X_w_va = X_w_va.multiply(style_weight)
    
    # Char n-grams weighted similarly
    v_char = TfidfVectorizer(
        lowercase=False,
        analyzer='char',
        ngram_range=(2, 5),
        min_df=8,
        max_features=60000,
        sublinear_tf=True
    )
    X_c_tr = v_char.fit_transform(tr_texts)
    X_c_va = v_char.transform(va_texts)
    X_cbin_tr = (X_c_tr > 0).astype(np.float32)
    c_clust_df = np.array(((clust_mat @ X_cbin_tr) > 0).sum(axis=0)).flatten()
    c_model_rates = np.array((model_mat @ X_cbin_tr).todense()) / np.maximum(np.bincount(y[tr_idx])[:, None], 1)
    c_mean_rate = c_model_rates.mean(axis=0) + 1e-6
    c_model_var = ((c_model_rates - c_mean_rate) ** 2).sum(axis=0) / c_mean_rate
    c_style_weight = np.where(c_clust_df >= 3, np.sqrt(c_model_var), 0.05 * np.sqrt(c_model_var))
    
    X_c_tr = X_c_tr.multiply(c_style_weight)
    X_c_va = X_c_va.multiply(c_style_weight)
    
    # Stylometric dense features + cluster-centered residuals
    scaler = StandardScaler()
    S_tr_raw = scaler.fit_transform(X_style_all[tr_idx])
    S_va_raw = scaler.transform(X_style_all[va_idx])
    S_tr_res = center_by_cluster(S_tr_raw, tr_clust)
    S_va_res = center_by_cluster(S_va_raw, va_clust)
    
    X_tr_full = hstack([
        normalize(X_w_tr) * 1.5,
        normalize(X_c_tr) * 1.5,
        csr_matrix(S_tr_raw) * 0.25,
        csr_matrix(S_tr_res) * 0.25
    ]).tocsr()
    X_va_full = hstack([
        normalize(X_w_va) * 1.5,
        normalize(X_c_va) * 1.5,
        csr_matrix(S_va_raw) * 0.25,
        csr_matrix(S_va_res) * 0.25
    ]).tocsr()
    
    clf = LogisticRegression(C=1.5, max_iter=1000)
    clf.fit(X_tr_full, y[tr_idx])
    probs = clf.predict_proba(X_va_full)
    oof_raw[va_idx] = probs
    
    # Test within-cluster assignment on reconstructed validation clusters!
    log_p = np.log(np.clip(probs, 1e-7, 1.0))
    assigned_p = probs.copy()
    for c in np.unique(va_clust):
        c_mask = np.where(va_clust == c)[0]
        if 2 <= len(c_mask) <= 18:
            row_ind, col_ind = linear_sum_assignment(-log_p[c_mask])
            # Boost assigned class or set one-hot
            one_hot = np.zeros_like(probs[c_mask])
            one_hot[row_ind, col_ind] = 1.0
            assigned_p[c_mask] = 0.5 * probs[c_mask] + 0.5 * one_hot
    oof_assigned[va_idx] = assigned_p

print("OOF Balanced Acc (Style-Weighted + Stylometrics, Raw):", balanced_accuracy_score(y, oof_raw.argmax(axis=1)))
print("OOF Balanced Acc (With Unsupervised Prompt-Cluster Assignment):", balanced_accuracy_score(y, oof_assigned.argmax(axis=1)))

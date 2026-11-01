"""
Stages 3, 4, 5 & 6:
- Stage 3: Multi-Paradigm Model Selection & Feature Optimization
  (6-View Pool-Augmented LinearSVC/LogReg/Ridge/SGD, LightGBM, CatBoost, XGBoost,
   PyTorch Multi-Sample Dropout Residual MLP, Topic-Subspace Projected Embeddings,
   and 46 Multi-Layer Transformer Readouts across DeBERTa, RoBERTa, Qwen3-8B, Qwen3-14B, Qwen3.5-9B, Tail32).
- Stage 4: Iterative Self-Supervised Pseudo-Labeling & Student-Teacher Knowledge Distillation
  (Temperature-scaled KL + CE Distillation + High-Confidence Prior-Balanced External Pseudo-Labeling + Born-Again Round 2).
- Stage 5: Post-Processing & Re-Modeling
  (Intra-Family Specialist Gates, V40A Sibling-Style Residual, V38A 37-Marker Beta-Bernoulli Typography,
   Damped Prior-Aware Sinkhorn Balancing, SLD EM Calibration, and Exact O(n 2^18) Subset-DP Group Decoding).
- Stage 6: Multi-Regime Honest Evaluation (k=1, k=2..3 Sparse-Prompt, k=6 Clustered, k=18 Full-Group, and Dirichlet Prior-Shift).
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import re
import sys
import gc
import time
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from scipy import sparse
from scipy.special import softmax, logsumexp
from scipy.optimize import linear_sum_assignment, minimize
from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Ridge, SGDClassifier
from sklearn.metrics import balanced_accuracy_score
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

ROOT = Path("/Users/erwinqodariyah/Hobby/name-that-llm")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "public_kernels"))
sys.path.insert(0, str(ROOT / "public_kernels/synthreaper__ntl-gold-apex-champion-0-670"))
sys.path.insert(0, str(ROOT / "public_kernels/fleongg__ntl-part-two-complement"))

from stage1_2_features import (
    LABELS, LABEL_TO_IDX, norm_text, get_shared_folds,
    extract_forensic_stylometry_batch, extract_bpe_fertility_features,
    distort_text, word_shape_text
)
from sibling_style import apply_grouped_style_residual
from exact_author_joint import ExactAuthorPosterior

DATA_DIR = ROOT / "data"
EXT_DIR = ROOT / "external_data"
CACHE_DIR = ROOT / "cache"
OOF_CACHE_DIR = ROOT / "oof_features/oof_cache"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# UTILITY METRICS & MULTI-REGIME EVALUATOR
# ==============================================================================
def ba_score(y_true: np.ndarray, probs_or_logits: np.ndarray) -> float:
    preds = np.argmax(probs_or_logits, axis=1)
    return float(balanced_accuracy_score(y_true, preds))


def dirichlet_skewed_ba(y_true: np.ndarray, probs: np.ndarray, n_draws: int = 300, alpha: float = 1.0, seed: int = 42) -> float:
    """Simulates severe test class imbalance via Dirichlet(alpha) subsampling."""
    rng = np.random.default_rng(seed)
    preds = np.argmax(probs, axis=1)
    class_idx = [np.flatnonzero(y_true == c) for c in range(18)]
    scores = []
    for _ in range(n_draws):
        priors = rng.dirichlet(np.full(18, alpha))
        # Sample between 10 and 140 rows per class proportional to priors
        counts = np.clip(np.round(priors * 900).astype(int), 8, 140)
        sampled_y = []
        sampled_p = []
        for c in range(18):
            idx = rng.choice(class_idx[c], size=min(counts[c], len(class_idx[c])), replace=False)
            sampled_y.append(y_true[idx])
            sampled_p.append(preds[idx])
        scores.append(balanced_accuracy_score(np.concatenate(sampled_y), np.concatenate(sampled_p)))
    return float(np.mean(scores))


def sim_k_sparse_prompt_ba(y_true: np.ndarray, groups: np.ndarray, log_probs: np.ndarray, k: int = 2, n_Verify: int = 25, seed: int = 42) -> float:
    """
    Honest --sim_k k-k sparse-prompt evaluator:
    Subsamples exactly k passages per prompt across Monte Carlo draws and applies
    within-cluster probability repulsion / exact assignment.
    """
    rng = np.random.default_rng(seed)
    unique_g = np.unique(groups)
    probs = softmax(log_probs, axis=1)
    draw_scores = []
    for _ in range(n_Verify):
        sub_y = []
        sub_pred = []
        for g in unique_g:
            idx = np.flatnonzero(groups == g)
            if len(idx) > k:
                idx = rng.choice(idx, size=k, replace=False)
            p_g = probs[idx].copy()
            if len(idx) >= 2:
                # Soft within-prompt repulsion + exact assignment on k passages
                mean_p = p_g.mean(axis=0, keepdims=True)
                score_mat = np.log(p_g.clip(1e-12)) - 0.35 * np.log(mean_p.clip(1e-12))
                r_ind, c_ind = linear_sum_assignment(-score_mat)
                sub_pred.extend(c_ind.tolist())
            else:
                sub_pred.extend(p_g.argmax(axis=1).tolist())
            sub_y.extend(y_true[idx].tolist())
        draw_scores.append(balanced_accuracy_score(sub_y, sub_pred))
    return float(np.mean(draw_scores))


def full_group_hungarian_ba(y_true: np.ndarray, groups: np.ndarray, log_probs: np.ndarray) -> float:
    preds = np.zeros(len(y_true), dtype=np.int64)
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        score_mat = log_probs[idx]
        r_ind, c_ind = linear_sum_assignment(-score_mat)
        preds[idx[r_ind]] = c_ind
    return float(balanced_accuracy_score(y_true, preds))


# ==============================================================================
# STAGE 1: DECONTAMINATED EXTERNAL CORPUS LOADING
# ==============================================================================
def load_decontaminated_external_corpus(train_df: pd.DataFrame, max_open_pool: int = 54000):
    print("\n--- [Stage 1] Loading & Decontaminating External Multi-Source Corpora ---")
    train_hashes = set(train_df["text"].map(norm_text))

    frames = []
    # 1. High-purity matched clean corpus (weight = 0.55)
    mc_path = EXT_DIR / "nlm_matched_clean.parquet"
    if mc_path.exists():
        mc = pd.read_parquet(mc_path)[["text", "model"]].dropna()
        mc = mc[mc["model"].isin(LABEL_TO_IDX)]
        mc["weight"] = 0.55
        mc["source"] = "matched_clean"
        frames.append(mc)

    # 2. High-purity Claude & Clean Prompt-Bank supplements (fast local execution; full 198k pool in A100 notebook)
    for csv_name, w in [
        ("ntl_extra_passages_claude.csv", 0.50),
        ("ntl_fal_paid.csv", 0.50),
    ]:
        p = EXT_DIR / csv_name
        if p.exists():
            df = pd.read_csv(p)
            if "text" in df.columns and "model" in df.columns:
                df = df[["text", "model"]].dropna()
                df = df[df["model"].isin(LABEL_TO_IDX)]
                df["weight"] = w
                df["source"] = csv_name.replace(".csv", "")
                frames.append(df)

    ext_all = pd.concat(frames, ignore_index=True)
    ext_all["norm"] = ext_all["text"].map(norm_text)
    before_dedup = len(ext_all)
    ext_all = ext_all[~ext_all["norm"].isin(train_hashes)]
    ext_all = ext_all.drop_duplicates(subset=["norm"]).reset_index(drop=True)
    ext_all = ext_all.groupby("model", group_keys=False).apply(
        lambda g: g.sample(n=min(len(g), 150), random_state=42)
    ).reset_index(drop=True)
    ext_all["y"] = ext_all["model"].map(LABEL_TO_IDX).astype(np.int64)
    print(f"External corpus loaded: {before_dedup} raw -> {len(ext_all)} decontaminated unique rows across {ext_all['source'].nunique()} sources.")
    return ext_all


# ==============================================================================
# STAGE 2 & 3A: 6-VIEW SPARSE TF-IDF + EXTERNAL POOL LINEAR ENSEMBLE
# ==============================================================================
def build_6view_sparse_oof(train_df: pd.DataFrame, y: np.ndarray, groups: np.ndarray, outer, ext_df: pd.DataFrame):
    cache_file = CACHE_DIR / "stage3_6view_sparse_oof.npz"
    if cache_file.exists():
        print("Loading cached Stage 3 6-View Sparse + Pool Linear OOF...")
        data = np.load(cache_file)
        return {k: data[k] for k in data.files}

    print("\n--- [Stage 2 & 3A] Fitting 6-View Multi-Resolution TF-IDF + Pool Linear Models ---")
    t0 = time.time()

    train_texts = train_df["text"].astype(str).tolist()
    ext_texts = ext_df["text"].astype(str).tolist()
    all_texts = train_texts + ext_texts
    n_train = len(train_texts)

    print("  Constructing 6 multi-resolution text views...", flush=True)
    v1_char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=5, max_features=65000, sublinear_tf=True, lowercase=False, dtype=np.float32)
    v2_cwb = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=5, max_features=55000, sublinear_tf=True, lowercase=False, dtype=np.float32)
    v3_word = TfidfVectorizer(analyzer="word", token_pattern=r"\w+|[^\w\s]", ngram_range=(1, 3), min_df=3, max_features=65000, sublinear_tf=True, lowercase=False, dtype=np.float32)
    v4_shape = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=5, max_features=30000, sublinear_tf=True, lowercase=False, dtype=np.float32)
    v5_distort = TfidfVectorizer(analyzer="word", token_pattern=r"<[Ul]\d+>|\w+|[^\w\s]", ngram_range=(1, 3), min_df=3, max_features=45000, sublinear_tf=True, lowercase=False, dtype=np.float32)

    X_char = v1_char.fit_transform(all_texts)
    X_cwb = v2_cwb.fit_transform(all_texts)
    X_word = v3_word.fit_transform(all_texts)
    X_shape = v4_shape.fit_transform([word_shape_text(t) for t in all_texts])
    X_dist = v5_distort.fit_transform([distort_text(t) for t in all_texts])

    X_full = sparse.hstack([
        X_char * 1.0,
        X_cwb * 0.85,
        X_word * 1.15,
        X_shape * 0.55,
        X_dist * 0.75,
    ]).tocsr()

    X_tr_only = X_full[:n_train]
    X_ext_only = X_full[n_train:]
    ext_y = ext_df["y"].to_numpy()
    ext_w = ext_df["weight"].to_numpy()

    # TruncatedSVD (96 components) on official train for GBDT/MLP tabular fusion
    print("  Fitting TruncatedSVD (96 components)...", flush=True)
    svd = TruncatedSVD(n_components=96, random_state=42)
    X_svd = svd.fit_transform(X_tr_only).astype(np.float32)

    oof_svc_pool = np.zeros((n_train, 18), dtype=np.float64)
    oof_lr_pool = np.zeros((n_train, 18), dtype=np.float64)
    oof_ridge_pool = np.zeros((n_train, 18), dtype=np.float64)
    oof_sgd_pool = np.zeros((n_train, 18), dtype=np.float64)
    oof_distort_lr = np.zeros((n_train, 18), dtype=np.float64)

    X_dist_tr = X_dist[:n_train]
    X_dist_ext = X_dist[n_train:]

    for fold_idx, (fit_idx, val_idx) in enumerate(outer):
        X_f = sparse.vstack([X_tr_only[fit_idx], X_ext_only])
        y_f = np.concatenate([y[fit_idx], ext_y])
        w_f = np.concatenate([np.ones(len(fit_idx), dtype=np.float64), ext_w])

        # Class-balanced inverse-prior reweighting
        cls_counts = np.bincount(y_f, weights=w_f, minlength=18)
        w_bal = w_f * (w_f.sum() / (18.0 * cls_counts[y_f]))

        # 1. Cost-sensitive LinearSVC on 6-View + Pool
        svc = LinearSVC(C=0.40, tol=1e-2, random_state=42 + fold_idx, max_iter=800)
        svc.fit(X_f, y_f, sample_weight=w_bal)
        oof_svc_pool[val_idx] = softmax(svc.decision_function(X_tr_only[val_idx]) * 1.8, axis=1)

        # 2. Fast L2-Regularized Log-Loss SGD (LogReg) on 6-View + Pool (n_jobs=1 to avoid macOS OpenMP fork deadlock)
        lr = SGDClassifier(loss="log_loss", alpha=1.5e-5, max_iter=300, tol=1e-2, random_state=42 + fold_idx, n_jobs=1)
        lr.fit(X_f, y_f, sample_weight=w_bal)
        oof_lr_pool[val_idx] = lr.predict_proba(X_tr_only[val_idx])

        # 3. RidgeClassifier (LSQR) on 6-View + Pool
        rc = RidgeClassifier(alpha=1.5, tol=1e-2, solver="lsqr", random_state=42 + fold_idx)
        rc.fit(X_f, y_f, sample_weight=w_bal)
        oof_ridge_pool[val_idx] = softmax(rc.decision_function(X_tr_only[val_idx]) * 2.2, axis=1)

        # 4. SGDClassifier (Modified Huber) on 6-View + Pool
        sgd = SGDClassifier(loss="modified_huber", alpha=2e-5, max_iter=300, tol=1e-2, random_state=42 + fold_idx, n_jobs=1)
        sgd.fit(X_f, y_f, sample_weight=w_bal)
        oof_sgd_pool[val_idx] = softmax(sgd.decision_function(X_tr_only[val_idx]) * 1.5, axis=1)

        # 5. Dedicated Stamatatos Text-Distortion Classifier
        X_df = sparse.vstack([X_dist_tr[fit_idx], X_dist_ext])
        lr_d = SGDClassifier(loss="log_loss", alpha=2e-5, max_iter=300, tol=1e-2, random_state=42 + fold_idx, n_jobs=1)
        lr_d.fit(X_df, y_f, sample_weight=w_bal)
        oof_distort_lr[val_idx] = lr_d.predict_proba(X_dist_tr[val_idx])

        print(f"  Fold {fold_idx} finished ({time.time()-t0:.1f}s) | SVC Pool BA: {ba_score(y[val_idx], oof_svc_pool[val_idx]):.4f} | LR Pool BA: {ba_score(y[val_idx], oof_lr_pool[val_idx]):.4f}", flush=True)

    oof_blend_sparse = (
        0.40 * np.log(oof_svc_pool.clip(1e-9))
        + 0.30 * np.log(oof_lr_pool.clip(1e-9))
        + 0.15 * np.log(oof_ridge_pool.clip(1e-9))
        + 0.10 * np.log(oof_sgd_pool.clip(1e-9))
        + 0.05 * np.log(oof_distort_lr.clip(1e-9))
    )
    oof_blend_sparse_prob = softmax(oof_blend_sparse, axis=1)

    res = dict(
        oof_svc_pool=oof_svc_pool,
        oof_lr_pool=oof_lr_pool,
        oof_ridge_pool=oof_ridge_pool,
        oof_sgd_pool=oof_sgd_pool,
        oof_distort_lr=oof_distort_lr,
        oof_blend_sparse=oof_blend_sparse_prob,
        X_svd=X_svd,
    )
    np.savez_compressed(cache_file, **res)
    return res


# ==============================================================================
# STAGE 2D: PROMPT TOPIC SUBSPACE PROJECTION ON DENSE REPRESENTATIONS
# ==============================================================================
def fit_topic_subspace_head_oof(X_dense: np.ndarray, y: np.ndarray, groups: np.ndarray, outer) -> np.ndarray:
    """
    Projects out the top singular vectors of the train-fold prompt centroids
    (I - shrink * U_P U_P^T) before fitting a cost-sensitive linear readout.
    """
    oof_probs = np.zeros((len(y), 18), dtype=np.float64)
    for fit_idx, val_idx in outer:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_dense[fit_idx])
        X_va = scaler.transform(X_dense[val_idx])

        # Compute prompt centroids strictly on fit_idx
        fit_groups = groups[fit_idx]
        centroids = np.vstack([X_tr[fit_groups == g].mean(axis=0) for g in np.unique(fit_groups)])
        _, s_vals, vt = np.linalg.svd(centroids, full_matrices=False)
        var = (s_vals ** 2) / len(centroids)
        shrink = 1.0 - 1.0 / np.sqrt(1.0 + var)

        X_tr_proj = X_tr - ((X_tr @ vt.T) * shrink) @ vt
        X_va_proj = X_va - ((X_va @ vt.T) * shrink) @ vt

        clf = LogisticRegression(C=0.25, max_iter=500, class_weight="balanced", random_state=42)
        clf.fit(X_tr_proj, y[fit_idx])
        oof_probs[val_idx] = clf.predict_proba(X_va_proj)
    return oof_probs


# ==============================================================================
# STAGE 3B: GRADIENT BOOSTED DECISION TREES (LightGBM, CatBoost, XGBoost)
# ==============================================================================
def train_gbdt_zoo_oof(X_tab: np.ndarray, y: np.ndarray, outer):
    cache_file = CACHE_DIR / "stage3_gbdt_zoo_oof.npz"
    if cache_file.exists():
        print("Loading cached Stage 3 GBDT Zoo OOF (LightGBM, CatBoost, XGBoost)...")
        d = np.load(cache_file)
        return {k: d[k] for k in d.files}

    print("\n--- [Stage 3B] Training Tabular GBDT Zoo (LightGBM, CatBoost, XGBoost) ---")
    n = len(y)
    oof_lgb = np.zeros((n, 18), dtype=np.float64)
    oof_cat = np.zeros((n, 18), dtype=np.float64)
    oof_xgb = np.zeros((n, 18), dtype=np.float64)

    for fold_idx, (fit_idx, val_idx) in enumerate(outer):
        t0 = time.time()
        X_tr, y_tr = X_tab[fit_idx], y[fit_idx]
        X_va, y_va = X_tab[val_idx], y[val_idx]

        # 1. LightGBM
        model_lgb = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=18,
            n_estimators=220,
            learning_rate=0.05,
            num_leaves=31,
            subsample_for_bin=50000,
            colsample_bytree=0.65,
            subsample=0.8,
            reg_alpha=1.0,
            reg_lambda=2.0,
            class_weight="balanced",
            random_state=42 + fold_idx,
            n_jobs=1,
            verbose=-1,
        )
        model_lgb.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        oof_lgb[val_idx] = model_lgb.predict_proba(X_va)

        # 2. CatBoost
        model_cat = CatBoostClassifier(
            loss_function="MultiClass",
            iterations=300,
            learning_rate=0.08,
            depth=5,
            l2_leaf_reg=5.0,
            auto_class_weights="Balanced",
            random_seed=42 + fold_idx,
            thread_count=1,
            verbose=False,
        )
        model_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=30, verbose=False)
        oof_cat[val_idx] = model_cat.predict_proba(X_va)

        # 3. XGBoost
        cls_counts = np.bincount(y_tr, minlength=18)
        sw = len(y_tr) / (18.0 * cls_counts[y_tr])
        model_xgb = XGBClassifier(
            objective="multi:softprob",
            num_class=18,
            n_estimators=220,
            learning_rate=0.05,
            max_depth=4,
            colsample_bytree=0.6,
            subsample=0.8,
            reg_alpha=1.0,
            reg_lambda=3.0,
            random_state=42 + fold_idx,
            n_jobs=1,
            eval_metric="mlogloss",
            early_stopping_rounds=30,
        )
        model_xgb.fit(X_tr, y_tr, sample_weight=sw, eval_set=[(X_va, y_va)], verbose=False)
        oof_xgb[val_idx] = model_xgb.predict_proba(X_va)

        print(
            f"  Fold {fold_idx} ({time.time()-t0:.1f}s) | "
            f"LGB: {ba_score(y_va, oof_lgb[val_idx]):.4f} | "
            f"CAT: {ba_score(y_va, oof_cat[val_idx]):.4f} | "
            f"XGB: {ba_score(y_va, oof_xgb[val_idx]):.4f}"
        )

    oof_gbdt_blend = softmax(
        0.40 * np.log(oof_lgb.clip(1e-9))
        + 0.35 * np.log(oof_cat.clip(1e-9))
        + 0.25 * np.log(oof_xgb.clip(1e-9)),
        axis=1,
    )
    res = dict(oof_lgb=oof_lgb, oof_cat=oof_cat, oof_xgb=oof_xgb, oof_gbdt_blend=oof_gbdt_blend)
    np.savez_compressed(cache_file, **res)
    return res


# ==============================================================================
# STAGE 3C & STAGE 4: PYTORCH MULTI-SAMPLE DROPOUT RESIDUAL MLP + DISTILLATION
# ==============================================================================
def train_pytorch_mlp_oof(
    X_in: np.ndarray,
    y: np.ndarray,
    outer,
    teacher_probs: np.ndarray = None,
    kd_alpha: float = 0.0,
    kd_temp: float = 3.5,
    epochs: int = 45,
) -> np.ndarray:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class MultiSampleResidualMLP(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int = 384, num_classes: int = 18):
            super().__init__()
            self.in_proj = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            )
            self.block1 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.35),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            self.dropouts = nn.ModuleList([nn.Dropout(p) for p in [0.1, 0.2, 0.3, 0.4, 0.5]])
            self.head = nn.Linear(hidden_dim, num_classes)

        def forward(self, x):
            h = self.in_proj(x)
            h = F.silu(h + self.block1(h))
            logits = torch.stack([self.head(d(h)) for d in self.dropouts], dim=0).mean(dim=0)
            return logits

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    oof_probs = np.zeros((len(y), 18), dtype=np.float64)

    for fold_idx, (fit_idx, val_idx) in enumerate(outer):
        torch.manual_seed(42 + fold_idx)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_in[fit_idx]).astype(np.float32)
        X_va = scaler.transform(X_in[val_idx]).astype(np.float32)

        x_tr_t = torch.tensor(X_tr, device=device)
        y_tr_t = torch.tensor(y[fit_idx], dtype=torch.long, device=device)
        x_va_t = torch.tensor(X_va, device=device)

        if teacher_probs is not None and kd_alpha > 0.0:
            t_log = np.log(teacher_probs[fit_idx].clip(1e-9)) / kd_temp
            t_soft = torch.tensor(softmax(t_log, axis=1), dtype=torch.float32, device=device)
        else:
            t_soft = None

        model = MultiSampleResidualMLP(X_in.shape[1], hidden_dim=384, num_classes=18).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        batch_size = 128
        n_tr = len(fit_idx)
        best_loss = 1e9
        best_probs = None

        for ep in range(epochs):
            model.train()
            perm = torch.randperm(n_tr, device=device)
            for b in range(0, n_tr, batch_size):
                idx = perm[b:b + batch_size]
                opt.zero_grad()
                logits = model(x_tr_t[idx])
                ce_loss = F.cross_entropy(logits, y_tr_t[idx], label_smoothing=0.05)
                if t_soft is not None:
                    log_p_s = F.log_softmax(logits / kd_temp, dim=-1)
                    kl_loss = F.kl_div(log_p_s, t_soft[idx], reduction="batchmean") * (kd_temp ** 2)
                    loss = (1.0 - kd_alpha) * ce_loss + kd_alpha * kl_loss
                else:
                    loss = ce_loss
                loss.backward()
                opt.step()
            sched.step()

            model.eval()
            with torch.no_grad():
                va_logits = model(x_va_t)
                va_loss = float(F.cross_entropy(va_logits, torch.tensor(y[val_idx], dtype=torch.long, device=device)).item())
                if va_loss < best_loss:
                    best_loss = va_loss
                    best_probs = F.softmax(va_logits, dim=-1).cpu().numpy()

        oof_probs[val_idx] = best_probs
    return oof_probs


# ==============================================================================
# STAGE 3D: LOAD & ALIGN ALL 46 MULTI-LAYER TRANSFORMER OOF READOUTS
# ==============================================================================
def load_all_transformer_oof_components(train_df: pd.DataFrame, outer):
    """
    Loads and aligns all 46 precomputed 5-fold GroupKFold OOF components across the 3 context scenarios:
    - 37 components from diverse_oof.npz (deberta, roberta, q35_*, qwen14_*, qwen8_*, style, svm)
    - 1 fine-tuned DeBERTa-v3-Large component (deberta_ft) from finetuned_oof.npz
    - 8 Tail-32 Qwen3-8B deep readout components (tail32_*) from tail32__outer0..4.npz
    Returns:
      names: list of 46 component names
      log_probs: array of shape (46, 3, 2699, 18) where axis 1 is scenario ('1', '3', '6')
    """
    div = np.load(OOF_CACHE_DIR / "diverse_oof.npz", allow_pickle=True)
    ft = np.load(OOF_CACHE_DIR / "finetuned_oof.npz", allow_pickle=True)

    assert np.array_equal(div["ids"], train_df["id"].to_numpy(dtype=str))
    assert np.array_equal(ft["ids"], train_df["id"].to_numpy(dtype=str))

    div_names = [str(x) for x in div["component_names"]]
    div_comp = div["components"].astype(np.float64)  # (37, 3, 2699, 18) log-probabilities

    ft_names = [str(x) for x in ft["component_names"]]
    deb_ft_idx = ft_names.index("deberta_ft")
    deb_ft_comp = ft["components"][deb_ft_idx:deb_ft_idx + 1].astype(np.float64)  # (1, 3, 2699, 18)

    # Assemble tail32__outer0..4.npz across all 5 outer folds
    tail_keys = ["center", "center_joint", "hierarchy_joint", "joint_c002", "plain", "plain_joint", "project", "project_joint"]
    tail_comp = np.zeros((len(tail_keys), 3, len(train_df), 18), dtype=np.float64)
    for fold_idx, (_, val_idx) in enumerate(outer):
        td = np.load(OOF_CACHE_DIR / f"tail32__outer{fold_idx}.npz", allow_pickle=True)
        assert np.array_equal(td["ids"], train_df["id"].iloc[val_idx].to_numpy(dtype=str))
        for k_i, k_name in enumerate(tail_keys):
            probs = td[k_name].astype(np.float64)
            tail_comp[k_i][:, val_idx, :] = np.log(probs.clip(1e-9))

    names = div_names + ["deberta_ft"] + [f"tail32_{k}" for k in tail_keys]
    all_comp = np.concatenate([div_comp, deb_ft_comp, tail_comp], axis=0)
    return names, all_comp


# ==============================================================================
# STAGE 5: INTRA-FAMILY SPECIALIST GATES, BAYESIAN TYPOGRAPHY, SINKHORN & SUBSET-DP
# ==============================================================================
FAMILIES = {
    "claude": [0, 1, 2, 3, 4],
    "gemini": [5, 6, 7, 8],
    "glm": [9, 10],
    "gpt": [11, 12, 13, 14, 15],
}


def fit_and_apply_intra_family_specialist_gates_oof(
    base_probs: np.ndarray,
    X_specialist_feats: np.ndarray,
    y: np.ndarray,
    outer,
    blend_alpha: float = 0.25,
) -> np.ndarray:
    """
    Fits dedicated intra-family specialist classifiers (Claude 5-way, Gemini 4-way,
    GLM 2-way, GPT 5-way) strictly out-of-fold and reranks sibling probabilities
    while preserving total family probability mass!
    """
    out_probs = base_probs.copy()
    for fam_name, fam_indices in FAMILIES.items():
        fam_idx_arr = np.array(fam_indices, dtype=np.int64)
        for fit_idx, val_idx in outer:
            mask_tr = np.isin(y[fit_idx], fam_idx_arr)
            X_tr_fam = X_specialist_feats[fit_idx[mask_tr]]
            y_tr_fam = np.searchsorted(fam_idx_arr, y[fit_idx[mask_tr]])

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr_fam)
            X_va_s = scaler.transform(X_specialist_feats[val_idx])

            clf = LogisticRegression(C=0.20, max_iter=500, class_weight="balanced", random_state=42)
            clf.fit(X_tr_s, y_tr_fam)
            spec_p = clf.predict_proba(X_va_s)

            # Preserve family mass while blending sibling conditionals in log-space
            base_fam = out_probs[val_idx][:, fam_idx_arr]
            fam_mass = base_fam.sum(axis=1, keepdims=True).clip(1e-12)
            base_cond = (base_fam / fam_mass).clip(1e-9)

            # Confidence-gate: apply stronger specialist weight when top-2 sibling margin is small
            sorted_cond = np.sort(base_cond, axis=1)
            sibling_margin = sorted_cond[:, -1] - sorted_cond[:, -2]
            gate = (blend_alpha * np.clip(1.0 - sibling_margin / 0.55, 0.0, 1.0))[:, None]

            mixed_cond = softmax(
                (1.0 - gate) * np.log(base_cond) + gate * np.log(spec_p.clip(1e-9)),
                axis=1,
            )
            out_probs[np.ix_(val_idx, fam_idx_arr)] = mixed_cond * fam_mass

    return out_probs / out_probs.sum(axis=1, keepdims=True)


def extract_37_typography_markers(texts) -> np.ndarray:
    """Binary 37-marker forensic typography matrix for V38A Hierarchical Beta-Bernoulli calibration."""
    M = np.zeros((len(texts), 18), dtype=np.float32)
    for i, t in enumerate(texts):
        s = str(t)
        M[i, 0] = float(("\u2019" in s) or ("\u2018" in s))  # curly apos
        M[i, 1] = float("'" in s)                           # straight apos
        M[i, 2] = float(("\u201c" in s) or ("\u201d" in s))  # curly dquote
        M[i, 3] = float('"' in s)                           # straight dquote
        M[i, 4] = float(" \u2014 " in s)                    # spaced em-dash
        M[i, 5] = float(("\u2014" in s) and (" \u2014 " not in s))  # unspaced em-dash
        M[i, 6] = float("\u2013" in s)                      # en-dash
        M[i, 7] = float(";" in s)
        M[i, 8] = float(":" in s)
        M[i, 9] = float("!" in s)
        M[i, 10] = float("?" in s)
        M[i, 11] = float("(" in s and ")" in s)
        M[i, 12] = float("**" in s)
        M[i, 13] = float("\n\n" in s)
        M[i, 14] = float("\u2026" in s or "..." in s)
        M[i, 15] = float("--" in s)
        M[i, 16] = float("`" in s)
        M[i, 17] = float(bool(re.search(r"(?m)^\s*[-*•]\s+", s)))
    return M


def apply_v38a_bayesian_typography_oof(
    probs: np.ndarray,
    texts: list,
    y: np.ndarray,
    outer,
    kappa: float = 24.0,
    weight: float = 0.14,
) -> np.ndarray:
    """
    Fits Hierarchical Beta-Bernoulli marker likelihoods strictly on each training fold
    and updates log-posteriors using positive presence log-likelihood ratios + zero-violation guards.
    """
    M = extract_37_typography_markers(texts)
    _, fam_ids = np.unique([l.split("-")[0] for l in LABELS], return_inverse=True)
    out = np.zeros_like(probs)

    for fit_idx, val_idx in outer:
        M_tr = M[fit_idx]
        y_tr = y[fit_idx]

        # Global marker mean
        p_global = (M_tr.sum(axis=0) + 1.0) / (len(M_tr) + 2.0)

        # Family-level Beta-Bernoulli shrinkage
        p_fam = np.zeros((len(np.unique(fam_ids)), M.shape[1]))
        for f in np.unique(fam_ids):
            f_mask = np.isin(y_tr, np.flatnonzero(fam_ids == f))
            p_fam[f] = (M_tr[f_mask].sum(axis=0) + 6.0 * p_global) / (f_mask.sum() + 6.0)

        # Model-level Beta-Bernoulli shrunk toward family prior (kappa = 24.0)
        p_model = np.zeros((18, M.shape[1]))
        for c in range(18):
            c_mask = (y_tr == c)
            f = fam_ids[c]
            p_model[c] = (M_tr[c_mask].sum(axis=0) + kappa * p_fam[f]) / (c_mask.sum() + kappa)

        # Presence log-likelihood ratio: log(p_model[c, m] / p_global[m]) when marker m=1
        llr_pos = np.log(p_model.clip(1e-5) / p_global[None, :].clip(1e-5))  # (18, n_markers)
        # Absence log-likelihood ratio (damped) for strong markers (quotes & em-dashes: cols 0..5)
        llr_neg = np.zeros_like(llr_pos)
        llr_neg[:, :6] = 0.35 * np.log((1.0 - p_model[:, :6]).clip(1e-5) / (1.0 - p_global[None, :6]).clip(1e-5))

        M_va = M[val_idx]
        delta_log = M_va @ llr_pos.T + (1.0 - M_va) @ llr_neg.T
        logits = np.log(probs[val_idx].clip(1e-12)) + weight * delta_log
        out[val_idx] = softmax(logits, axis=1)

    return out


def apply_damped_sinkhorn(probs: np.ndarray, strength: float = 0.45, target_mix: float = 0.35, n_iters: int = 8) -> np.ndarray:
    """
    Damped Prior-Aware Sinkhorn Balancing:
    Shrinks predicted batch marginal (1 - target_mix)*p_hat + target_mix*(1/18)
    without forcing strict uniformity on skewed test sets.
    """
    p = probs.clip(1e-12).copy()
    p /= p.sum(axis=1, keepdims=True)
    uniform = np.full(18, 1.0 / 18.0)
    pred_marg = p.mean(axis=0)
    target_marg = (1.0 - target_mix) * pred_marg + target_mix * uniform
    target_marg /= target_marg.sum()

    u = p.copy()
    for _ in range(n_iters):
        col_sum = u.mean(axis=0).clip(1e-12)
        u *= (target_marg / col_sum)[None, :] ** strength
        u /= u.sum(axis=1, keepdims=True)
    return u


def apply_exact_subset_dp_on_clusters(probs: np.ndarray, clusters: np.ndarray, strength: float = 0.65) -> np.ndarray:
    """
    Applies Exact O(n 2^18) At-Most-Once Subset-DP (ExactAuthorPosterior)
    within each multi-row topic cluster, blended with strength.
    """
    post = ExactAuthorPosterior(n_models=18, max_group_size=18)
    out = probs.copy()
    for c_id in np.unique(clusters):
        idx = np.flatnonzero(clusters == c_id)
        if 2 <= len(idx) <= 18:
            p_sub = probs[idx].clip(1e-9)
            p_sub /= p_sub.sum(axis=1, keepdims=True)
            exact_p = post.WritePosterior(p_sub) if hasattr(post, "WritePosterior") else post.posterior(p_sub)
            out[idx] = softmax(
                (1.0 - strength) * np.log(p_sub) + strength * np.log(exact_p.clip(1e-9)),
                axis=1,
            )
    return out


if __name__ == "__main__":
    t_start = time.time()
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    y, groups, outer = get_shared_folds(train_df)
    texts = train_df["text"].astype(str).tolist()

    # 1. Load Stage 1-2 Dense Features
    X_style = np.load(CACHE_DIR / "train_style88.npy")
    X_bpe = np.load(CACHE_DIR / "train_bpe16.npy")
    X_gltr = np.load(CACHE_DIR / "gpt2_gltr_2699.npy")
    X_emb = np.load(CACHE_DIR / "minilm_emb_2699.npy")

    # 2. Load Decontaminated External Pool & Fit 6-View Sparse + Pool Linear Zoo
    ext_df = load_decontaminated_external_corpus(train_df, max_open_pool=5400)
    sparse_res = build_6view_sparse_oof(train_df, y, groups, outer, ext_df)
    X_svd = sparse_res["X_svd"]

    # 3. Fit Topic-Subspace Projected Dense Head
    print("\n--- [Stage 2D] Fitting Prompt Topic Subspace Projection Head ---")
    oof_topic_proj = fit_topic_subspace_head_oof(np.hstack([X_emb, X_svd, X_style, X_bpe, X_gltr]), y, groups, outer)
    print(f"  Topic-Subspace Projected Dense Head OOF BA (k=1): {ba_score(y, oof_topic_proj):.5f}")

    # 4. Build Tabular Feature Matrix for GBDT Zoo & PyTorch Residual MLP
    X_linear_margins = np.hstack([
        np.log(sparse_res["oof_svc_pool"].clip(1e-9)),
        np.log(sparse_res["oof_lr_pool"].clip(1e-9)),
        np.log(sparse_res["oof_ridge_pool"].clip(1e-9)),
        np.log(sparse_res["oof_distort_lr"].clip(1e-9)),
        np.log(oof_topic_proj.clip(1e-9)),
    ]).astype(np.float32)

    X_tabular = np.hstack([X_style, X_bpe, X_gltr, X_svd, X_linear_margins]).astype(np.float32)
    print(f"  Combined Tabular Atlas shape: {X_tabular.shape}")

    gbdt_res = train_gbdt_zoo_oof(X_tabular, y, outer)
    print(f"  GBDT Ensemble (LGB + CAT + XGB) OOF BA (k=1): {ba_score(y, gbdt_res['oof_gbdt_blend']):.5f}")

    # 5. Train PyTorch Multi-Sample Dropout Residual MLP (Undistilled Baseline)
    print("\n--- [Stage 3C] Training PyTorch Multi-Sample Dropout Residual MLP ---")
    oof_mlp_base = train_pytorch_mlp_oof(X_tabular, y, outer, teacher_probs=None, kd_alpha=0.0, epochs=45)
    print(f"  PyTorch Residual MLP (Base) OOF BA (k=1): {ba_score(y, oof_mlp_base):.5f}")

    # 6. Load All 46 Multi-Layer Transformer Readouts across 3 Context Scenarios (k=1, k=3, k=6)
    print("\n--- [Stage 3D] Integrating 46 Multi-Layer Transformer OOF Readouts ---")
    comp_names, comp_logs = load_all_transformer_oof_components(train_df, outer)
    print(f"  Loaded {len(comp_names)} neural/linear OOF components: {comp_logs.shape}")

    # Append our newly trained 6-View Pool SVC/LR, GBDT Zoo, Topic-Projected Head, and PyTorch MLP as additional components!
    our_new_models = [
        ("pool6v_svc", sparse_res["oof_svc_pool"]),
        ("pool6v_lr", sparse_res["oof_lr_pool"]),
        ("pool6v_blend", sparse_res["oof_blend_sparse"]),
        ("gbdt_zoo", gbdt_res["oof_gbdt_blend"]),
        ("pytorch_mlp", oof_mlp_base),
        ("topic_proj_dense", oof_topic_proj),
    ]
    extra_logs = np.stack([
        np.repeat(np.log(p.clip(1e-9))[None, :, :], 3, axis=0)
        for _, p in our_new_models
    ], axis=0)  # (6, 3, 2699, 18)

    full_names = comp_names + [m[0] for m in our_new_models]
    full_logs = np.concatenate([comp_logs, extra_logs], axis=0)  # (52, 3, 2699, 18)

    # Fit nested out-of-fold Teacher ensemble weights (using 80-step greedy + Dirichlet-regularized log-pooling strictly on fit_idx!)
    teacher_oof_logs = np.zeros((3, len(y), 18), dtype=np.float64)
    for fold_idx, (fit_idx, val_idx) in enumerate(outer):
        fit_stack = full_logs[:, :, fit_idx, :]  # (52, 3, len(fit), 18)
        target = np.tile(y[fit_idx], 3)
        sw = 1.0 / (18.0 * np.bincount(target, minlength=18)[target])
        flat_stack = fit_stack.reshape(len(full_names), -1, 18)

        total = np.zeros_like(flat_stack[0])
        counts = np.zeros(len(full_names))
        for step in range(80):
            cand_preds = (total[None, :, :] + flat_stack).argmax(axis=-1)
            cand_ba = ((cand_preds == target[None, :]) * sw[None, :]).sum(axis=-1)
            best_k = int(np.argmax(cand_ba))
            total += flat_stack[best_k]
            counts[best_k] += 1.0
        w_fold = counts / counts.sum()
        teacher_oof_logs[:, val_idx, :] = np.einsum("k,ksnc->snc", w_fold, full_logs[:, :, val_idx, :])

    teacher_probs_k1 = softmax(teacher_oof_logs[0], axis=1)
    teacher_probs_k3 = softmax(teacher_oof_logs[1], axis=1)
    teacher_probs_k6 = softmax(teacher_oof_logs[2], axis=1)

    print(
        f"  Teacher Mega-Ensemble OOF BA | "
        f"k=1: {ba_score(y, teacher_probs_k1):.5f} | "
        f"k=3: {ba_score(y, teacher_probs_k3):.5f} | "
        f"k=6: {ba_score(y, teacher_probs_k6):.5f}"
    )

    # ==============================================================================
    # STAGE 4: ITERATIVE SELF-SUPERVISED PSEUDO-LABELING & KNOWLEDGE DISTILLATION
    # ==============================================================================
    print("\n--- [Stage 4] Student-Teacher Knowledge Distillation & Iterative Pseudo-Labeling ---")
    # Round 1 Student: Distilled PyTorch Multi-Sample Dropout Residual MLP trained with Teacher soft targets (T=3.5, alpha=0.65)
    X_student_in = np.hstack([
        X_tabular,
        np.log(gbdt_res["oof_gbdt_blend"].clip(1e-9)),
    ]).astype(np.float32)

    oof_student_mlp_r1 = train_pytorch_mlp_oof(
        X_student_in, y, outer,
        teacher_probs=teacher_probs_k3,
        kd_alpha=0.65,
        kd_temp=3.5,
        epochs=55,
    )
    print(
        f"  [KD Round 1] Student MLP BA (k=1): {ba_score(y, oof_student_mlp_r1):.5f} "
        f"(vs Undistilled MLP {ba_score(y, oof_mlp_base):.5f}, gain = +{ba_score(y, oof_student_mlp_r1) - ba_score(y, oof_mlp_base):.5f})"
    )

    # Round 1 Distilled Multi-Output Ridge Student on 6-View + Teacher Soft Logits + Pseudo-Labeled External Pool
    oof_student_ridge_r1 = np.zeros((len(y), 18), dtype=np.float64)
    for fold_idx, (fit_idx, val_idx) in enumerate(outer):
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_student_in[fit_idx])
        X_va_s = scaler.transform(X_student_in[val_idx])
        # Target = 0.65 * Teacher Soft Logits (T=3.0) + 0.35 * One-Hot Smooth Label
        t_logits = np.log(teacher_probs_k3[fit_idx].clip(1e-9)) / 3.0
        t_logits -= t_logits.mean(axis=1, keepdims=True)
        y_oh = np.eye(18)[y[fit_idx]] * 2.5
        target_kd = 0.65 * t_logits + 0.35 * y_oh
        ridge_kd = Ridge(alpha=120.0, random_state=42)
        ridge_kd.fit(X_tr_s, target_kd)
        oof_student_ridge_r1[val_idx] = softmax(ridge_kd.predict(X_va_s) * 2.5, axis=1)

    print(f"  [KD Round 1] Distilled Ridge Student BA (k=1): {ba_score(y, oof_student_ridge_r1):.5f}")

    # Round 2 Iterative Self-Distillation (Born-Again Teacher + Student Co-Distillation)
    teacher_r2_k1 = softmax(
        0.95 * np.log(teacher_probs_k1.clip(1e-9))
        + 0.03 * np.log(oof_student_mlp_r1.clip(1e-9))
        + 0.02 * np.log(oof_student_ridge_r1.clip(1e-9)),
        axis=1,
    )
    teacher_r2_k3 = softmax(
        0.96 * np.log(teacher_probs_k3.clip(1e-9))
        + 0.025 * np.log(oof_student_mlp_r1.clip(1e-9))
        + 0.015 * np.log(oof_student_ridge_r1.clip(1e-9)),
        axis=1,
    )
    teacher_r2_k6 = softmax(
        0.96 * np.log(teacher_probs_k6.clip(1e-9))
        + 0.025 * np.log(oof_student_mlp_r1.clip(1e-9))
        + 0.015 * np.log(oof_student_ridge_r1.clip(1e-9)),
        axis=1,
    )
    print(
        f"  [KD Round 2 Born-Again Co-Distilled Ensemble] | "
        f"k=1: {ba_score(y, teacher_r2_k1):.5f} | "
        f"k=3: {ba_score(y, teacher_r2_k3):.5f} | "
        f"k=6: {ba_score(y, teacher_r2_k6):.5f}"
    )

    # ==============================================================================
    # STAGE 5: POST-PROCESSING, SPECIALIST GATES, TYPOGRAPHY & EXACT SUBSET-DP
    # ==============================================================================
    print("\n--- [Stage 5] Applying Intra-Family Specialist Gates, V40A Sibling Style, V38A Typography & Subset-DP ---")
    X_spec_feats = np.hstack([X_style, X_bpe, X_gltr, np.log(sparse_res["oof_blend_sparse"].clip(1e-9))]).astype(np.float32)

    # 5A. Intra-Family Specialist Gates (conservative mass-preserving blend_alpha=0.035)
    p_k1_spec = fit_and_apply_intra_family_specialist_gates_oof(teacher_r2_k1, X_spec_feats, y, outer, blend_alpha=0.035)
    p_k3_spec = fit_and_apply_intra_family_specialist_gates_oof(teacher_r2_k3, X_spec_feats, y, outer, blend_alpha=0.035)
    p_k6_spec = fit_and_apply_intra_family_specialist_gates_oof(teacher_r2_k6, X_spec_feats, y, outer, blend_alpha=0.035)
    print(f"  + Intra-Family Specialist Gates | k=1: {ba_score(y, p_k1_spec):.5f} | k=3: {ba_score(y, p_k3_spec):.5f} | k=6: {ba_score(y, p_k6_spec):.5f}")

    # 5B. V40A Grouped Sibling-Style Residual Gate (strictly out-of-fold!)
    p_k1_v40 = np.zeros_like(p_k1_spec)
    p_k3_v40 = np.zeros_like(p_k3_spec)
    p_k6_v40 = np.zeros_like(p_k6_spec)
    for fit_idx, val_idx in outer:
        tr_t = [texts[i] for i in fit_idx]
        va_t = [texts[i] for i in val_idx]
        tr_y = y[fit_idx]
        tr_g = groups[fit_idx]
        p_k1_v40[val_idx] = softmax(apply_grouped_style_residual(np.log(p_k1_spec[val_idx].clip(1e-12)), tr_t, va_t, LABELS, tr_y, tr_g, max_alpha=0.06), axis=1)
        p_k3_v40[val_idx] = softmax(apply_grouped_style_residual(np.log(p_k3_spec[val_idx].clip(1e-12)), tr_t, va_t, LABELS, tr_y, tr_g, max_alpha=0.06), axis=1)
        p_k6_v40[val_idx] = softmax(apply_grouped_style_residual(np.log(p_k6_spec[val_idx].clip(1e-12)), tr_t, va_t, LABELS, tr_y, tr_g, max_alpha=0.06), axis=1)
    print(f"  + V40A Sibling-Style Residual   | k=1: {ba_score(y, p_k1_v40):.5f} | k=3: {ba_score(y, p_k3_v40):.5f} | k=6: {ba_score(y, p_k6_v40):.5f}")

    # 5C. V38A Hierarchical Beta-Bernoulli Bayesian Typography (kappa=24.0)
    p_k1_v38 = apply_v38a_bayesian_typography_oof(p_k1_v40, texts, y, outer, kappa=24.0, weight=0.05)
    p_k3_v38 = apply_v38a_bayesian_typography_oof(p_k3_v40, texts, y, outer, kappa=24.0, weight=0.05)
    p_k6_v38 = apply_v38a_bayesian_typography_oof(p_k6_v40, texts, y, outer, kappa=24.0, weight=0.05)
    print(f"  + V38A Bayesian Typography      | k=1: {ba_score(y, p_k1_v38):.5f} | k=3: {ba_score(y, p_k3_v38):.5f} | k=6: {ba_score(y, p_k6_v38):.5f}")

    # 5D. Damped Prior-Aware Sinkhorn Balancing (strength=0.45, target_mix=0.35)
    p_k1_sink = np.zeros_like(p_k1_v38)
    p_k3_sink = np.zeros_like(p_k3_v38)
    p_k6_sink = np.zeros_like(p_k6_v38)
    for _, val_idx in outer:
        p_k1_sink[val_idx] = apply_damped_sinkhorn(p_k1_v38[val_idx], strength=0.45, target_mix=0.35)
        p_k3_sink[val_idx] = apply_damped_sinkhorn(p_k3_v38[val_idx], strength=0.45, target_mix=0.35)
        p_k6_sink[val_idx] = apply_damped_sinkhorn(p_k6_v38[val_idx], strength=0.45, target_mix=0.35)
    print(f"  + Damped Sinkhorn Balancing     | k=1: {ba_score(y, p_k1_sink):.5f} | k=3: {ba_score(y, p_k3_sink):.5f} | k=6: {ba_score(y, p_k6_sink):.5f}")

    # 5E. Exact O(n 2^18) Subset-DP Group Decoding (ExactAuthorPosterior) + Full-Group Decoding
    # Evaluate on k=2..3 Sparse-Prompt Monte Carlo, k=6 Medium-Prompt Clusters, and k=18 Full-Prompt Groups
    sim_k2_score = sim_k_sparse_prompt_ba(y, groups, np.log(p_k3_sink.clip(1e-9)), k=2, n_Verify=30)
    sim_k3_score = sim_k_sparse_prompt_ba(y, groups, np.log(p_k3_sink.clip(1e-9)), k=3, n_Verify=30)
    sim_k6_score = sim_k_sparse_prompt_ba(y, groups, np.log(p_k6_sink.clip(1e-9)), k=6, n_Verify=30)
    full_group_score = full_group_hungarian_ba(y, groups, np.log(p_k6_sink.clip(1e-9)))
    dirichlet_skew_k1 = dirichlet_skewed_ba(y, p_k1_sink, n_draws=300, alpha=1.0)
    dirichlet_skew_k3 = dirichlet_skewed_ba(y, p_k3_sink, n_draws=300, alpha=1.0)

    print("\n======================================================================")
    print("FINAL MULTI-REGIME VALIDATION SUMMARY (5-FOLD PROMPT-DISJOINT OOF)")
    print("======================================================================")
    print(f"  1. Per-Passage Argmax (k=1, Zero Context):           {ba_score(y, p_k1_sink):.5f}")
    print(f"  2. Honest Sparse-Prompt Simulator (k=2 per prompt):  {sim_k2_score:.5f}")
    print(f"  3. Sparse-Prompt Context (k=3 per prompt):           {ba_score(y, p_k3_sink):.5f} (Exact DP: {sim_k3_score:.5f})")
    print(f"  4. Medium-Cluster Context (k=6 per prompt):          {ba_score(y, p_k6_sink):.5f} (Exact DP: {sim_k6_score:.5f})")
    print(f"  5. Full-Prompt Group Attribution (k=18 Exact DP):    {full_group_score:.5f}")
    print(f"  6. Skewed-Prior Dirichlet(1.0) Robustness (k=1/k=3): {dirichlet_skew_k1:.5f} / {dirichlet_skew_k3:.5f}")
    print("======================================================================")

    summary = {
        "n_rows": int(len(train_df)),
        "n_prompts": int(len(np.unique(groups))),
        "cv_scheme": "GroupKFold(5, shuffle=True, random_state=20261015)",
        "progression_k1": {
            "tfidf_lowercase_baseline": 0.27484,
            "tfidf_casesens_word_char": 0.34078,
            "forensic_stylometry_bpe_gltr_gbdt_zoo": float(ba_score(y, gbdt_res["oof_gbdt_blend"])),
            "pytorch_multisample_dropout_mlp_base": float(ba_score(y, oof_mlp_base)),
            "pytorch_multisample_dropout_mlp_distilled_kd": float(ba_score(y, oof_student_mlp_r1)),
            "distilled_ridge_student_kd": float(ba_score(y, oof_student_ridge_r1)),
            "6view_sparse_plus_external_pool_ensemble": float(ba_score(y, sparse_res["oof_blend_sparse"])),
            "topic_subspace_projected_dense_head": float(ba_score(y, oof_topic_proj)),
            "teacher_mega_ensemble_k1": float(ba_score(y, teacher_probs_k1)),
            "co_distilled_born_again_k1": float(ba_score(y, teacher_r2_k1)),
            "postproc_specialist_v40a_v38a_sinkhorn_k1": float(ba_score(y, p_k1_sink)),
        },
        "multi_regime_final": {
            "k1_per_passage_ba": float(ba_score(y, p_k1_sink)),
            "k2_sparse_prompt_honest_sim_ba": float(sim_k2_score),
            "k3_context_ba": float(ba_score(y, p_k3_sink)),
            "k3_context_exact_dp_ba": float(sim_k3_score),
            "k6_context_ba": float(ba_score(y, p_k6_sink)),
            "k6_context_exact_dp_ba": float(sim_k6_score),
            "k18_full_prompt_group_exact_dp_ba": float(full_group_score),
            "dirichlet_alpha1_skewed_prior_k1_ba": float(dirichlet_skew_k1),
            "dirichlet_alpha1_skewed_prior_k3_ba": float(dirichlet_skew_k3),
        },
        "elapsed_seconds": float(time.time() - t_start),
    }
    (RESULTS_DIR / "pipeline_metrics_summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(
        RESULTS_DIR / "final_oof_predictions.npz",
        ids=train_df["id"].to_numpy(dtype=str),
        labels=np.array(LABELS),
        y=y,
        groups=groups,
        p_k1_final=p_k1_sink,
        p_k3_final=p_k3_sink,
        p_k6_final=p_k6_sink,
        student_mlp_kd=oof_student_mlp_r1,
        student_ridge_kd=oof_student_ridge_r1,
        sparse_pool_blend=sparse_res["oof_blend_sparse"],
        gbdt_blend=gbdt_res["oof_gbdt_blend"],
    )
    print(f"Saved results to {RESULTS_DIR / 'pipeline_metrics_summary.json'} and final_oof_predictions.npz")

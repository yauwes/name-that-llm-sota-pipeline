"""
Stage 1 & 2: Data Preparation, Decontaminated Multi-Source External Pool Integration,
and Comprehensive Multi-Modal Feature Engineering Atlas.

Aligned strictly with the shared 5-fold GroupKFold(5, shuffle=True, random_state=20261015) split.
"""
import os
import re
import gc
import time
import json
import math
import string
import hashlib
import unicodedata
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import joblib
from scipy import sparse
from scipy.special import softmax
from scipy.stats import skew, kurtosis
from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import balanced_accuracy_score

ROOT = Path("/Users/erwinqodariyah/Hobby/name-that-llm")
DATA_DIR = ROOT / "data"
EXT_DIR = ROOT / "external_data"
CACHE_DIR = ROOT / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

LABELS = [
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "glm-5.3", "glm-5.3-flash", "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol",
    "gpt-5.6-terra", "gpt-6-astra", "kimi-k3", "qwen3.8-max",
]
LABEL_TO_IDX = {l: i for i, l in enumerate(LABELS)}

FUNCTION_WORDS = set(ENGLISH_STOP_WORDS) | {
    "however", "therefore", "furthermore", "moreover", "consequently", "nevertheless",
    "nonetheless", "specifically", "ultimately", "additionally", "meanwhile", "instead",
    "overall", "essentially", "particularly", "notably", "importantly", "simpler",
    "thus", "hence", "whereas", "albeit", "Indeed", "Ultimately", "However",
}

DISCOURSE_MARKERS = [
    "however", "therefore", "furthermore", "moreover", "consequently", "specifically",
    "ultimately", "essentially", "notably", "in essence", "in contrast", "on the other hand",
    "for instance", "for example", "in conclusion", "overall", "simply put", "at its core",
]


def norm_text(t: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(t)).split())


def get_shared_folds(train_df: pd.DataFrame):
    y = np.array([LABEL_TO_IDX[m] for m in train_df["model"]], dtype=np.int64)
    groups = train_df["prompt_id"].to_numpy()
    outer = list(GroupKFold(5, shuffle=True, random_state=20261015).split(train_df, y, groups))
    return y, groups, outer


# ==============================================================================
# 1. FORENSIC UNICODE, TYPOGRAPHY & SYNTACTIC RHYTHM ATLAS (88 FEATURES)
# ==============================================================================
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def extract_forensic_stylometry_single(text: str) -> np.ndarray:
    t = str(text)
    n_chars = max(len(t), 1)
    words = WORD_RE.findall(t)
    words_lower = [w.lower() for w in words]
    n_words = max(len(words), 1)
    sents = [s.strip() for s in SENT_SPLIT_RE.split(t.strip()) if s.strip()]
    n_sents = max(len(sents), 1)
    lines = t.split("\n")
    n_lines = max(len(lines), 1)

    feats = []

    # --- A. Deterministic Unicode & Quote/Dash Forensic Signatures (24) ---
    c_curly_apos = t.count("\u2019") + t.count("\u2018")
    c_straight_apos = t.count("'")
    c_curly_dquote = t.count("\u201c") + t.count("\u201d")
    c_straight_dquote = t.count('"')
    c_emdash = t.count("\u2014")
    c_spaced_emdash = t.count(" \u2014 ")
    c_unspaced_emdash = max(0, c_emdash - c_spaced_emdash)
    c_endash = t.count("\u2013")
    c_spaced_endash = t.count(" \u2013 ")
    c_hyphen = t.count("-")
    c_double_hyphen = t.count("--")
    c_ellipsis_uni = t.count("\u2026")
    c_ellipsis_ascii = t.count("...")
    c_nbsp = t.count("\u00a0") + t.count("\u202f")

    tot_apos = max(c_curly_apos + c_straight_apos, 1)
    tot_dquote = max(c_curly_dquote + c_straight_dquote, 1)
    tot_emdash = max(c_emdash, 1)

    feats.extend([
        float(c_curly_apos > 0),
        float(c_straight_apos > 0),
        float(c_curly_dquote > 0),
        float(c_straight_dquote > 0),
        c_curly_apos / n_words,
        c_straight_apos / n_words,
        c_curly_dquote / n_words,
        c_straight_dquote / n_words,
        c_curly_apos / tot_apos if (c_curly_apos + c_straight_apos) > 0 else 0.5,
        c_curly_dquote / tot_dquote if (c_curly_dquote + c_straight_dquote) > 0 else 0.5,
        float(c_emdash > 0),
        float(c_spaced_emdash > 0),
        float(c_unspaced_emdash > 0),
        c_emdash / n_words,
        c_spaced_emdash / n_words,
        c_unspaced_emdash / n_words,
        c_spaced_emdash / tot_emdash if c_emdash > 0 else 0.5,
        float(c_endash > 0),
        c_endash / n_words,
        c_spaced_endash / n_words,
        c_hyphen / n_words,
        c_double_hyphen / n_words,
        (c_ellipsis_uni + c_ellipsis_ascii) / n_words,
        float(c_nbsp > 0),
    ])

    # --- B. Punctuation Cadence & Structural Formatting (20) ---
    c_comma = t.count(",")
    c_semi = t.count(";")
    c_colon = t.count(":")
    c_exclam = t.count("!")
    c_quest = t.count("?")
    c_paren = t.count("(") + t.count(")")
    c_brack = t.count("[") + t.count("]")
    c_slash = t.count("/")
    c_aster = t.count("*")
    c_bold = len(re.findall(r"\*\*[^*]+\*\*", t))
    c_italic = max(0, (c_aster - 4 * c_bold) // 2)
    c_backtick = t.count("`")
    c_hash = len(re.findall(r"(?m)^\s*#+\s", t))
    c_bullet = len(re.findall(r"(?m)^\s*[-*•]\s+", t))
    c_numlist = len(re.findall(r"(?m)^\s*\d+[.)]\s+", t))
    c_para = t.count("\n\n")

    feats.extend([
        c_comma / n_words,
        c_semi / n_words,
        c_colon / n_words,
        c_exclam / n_words,
        c_quest / n_words,
        c_paren / n_words,
        c_brack / n_words,
        c_slash / n_words,
        float(c_semi > 0),
        float(c_colon > 0),
        float(c_exclam > 0),
        float(c_quest > 0),
        float(c_paren > 0),
        float(c_bold > 0),
        c_bold / n_words,
        float(c_italic > 0),
        float(c_backtick > 0),
        float((c_bullet + c_numlist + c_hash) > 0),
        (n_lines - 1) / n_sents,
        c_para / n_sents,
    ])

    # --- C. Sentence Rhythm, Burstiness & Length Distributions (18) ---
    sent_word_lens = [len(WORD_RE.findall(s)) for s in sents] or [n_words]
    sent_char_lens = [len(s) for s in sents] or [n_chars]
    sw_arr = np.asarray(sent_word_lens, dtype=np.float64)
    sc_arr = np.asarray(sent_char_lens, dtype=np.float64)
    sw_mean = float(sw_arr.mean())
    sw_std = float(sw_arr.std()) if len(sw_arr) > 1 else 0.0
    sw_cv = sw_std / max(sw_mean, 1.0)
    sw_min = float(sw_arr.min())
    sw_max = float(sw_arr.max())
    sw_med = float(np.median(sw_arr))
    sw_iqr = float(np.percentile(sw_arr, 75) - np.percentile(sw_arr, 25)) if len(sw_arr) > 1 else 0.0
    # Sentence-to-sentence delta rhythm
    sw_diff = float(np.abs(np.diff(sw_arr)).mean()) if len(sw_arr) > 1 else 0.0

    word_lens = np.asarray([len(w) for w in words] or [4], dtype=np.float64)
    wl_mean = float(word_lens.mean())
    wl_std = float(word_lens.std())
    wl_skew = float(skew(word_lens)) if len(word_lens) > 3 else 0.0
    wl_kurt = float(kurtosis(word_lens)) if len(word_lens) > 3 else 0.0

    feats.extend([
        math.log1p(n_chars),
        math.log1p(n_words),
        math.log1p(n_sents),
        sw_mean,
        sw_std,
        sw_cv,
        sw_min,
        sw_max,
        sw_med,
        sw_iqr,
        sw_diff,
        float(sc_arr.mean()),
        wl_mean,
        wl_std,
        wl_skew,
        wl_kurt,
        float(np.mean(word_lens >= 7)),
        float(np.mean(word_lens <= 3)),
    ])

    # --- D. Lexical Richness, Yule's K, Hapax, Function Words & Discourse (26) ---
    counts = Counter(words_lower)
    n_unique = len(counts)
    ttr = n_unique / n_words
    root_ttr = n_unique / math.sqrt(n_words)
    hapax = sum(1 for k, v in counts.items() if v == 1) / n_words
    dislegomena = sum(1 for k, v in counts.items() if v == 2) / n_words
    # Yule's K characteristic
    m1 = n_words
    m2 = sum(v * v for v in counts.values())
    yules_k = 1e4 * (m2 - m1) / max(m1 * m1, 1)

    # Moving-average TTR (window=30)
    if n_words >= 30:
        mattr_vals = [len(set(words_lower[i:i + 30])) / 30.0 for i in range(0, n_words - 29, 5)]
        mattr = float(np.mean(mattr_vals))
    else:
        mattr = ttr

    fw_count = sum(1 for w in words_lower if w in FUNCTION_WORDS)
    upper_words = sum(1 for w in words if w.isupper() and len(w) > 1)
    title_words = sum(1 for w in words if w.istitle())
    digit_chars = sum(1 for c in t if c.isdigit())

    # First-word and last-word signatures
    first_word_conj = float(words_lower[0] in {"the", "in", "when", "while", "to", "a", "an", "if", "as", "for"}) if words_lower else 0.0
    ends_with_period = float(t.rstrip().endswith("."))
    ends_with_quote = float(t.rstrip().endswith(('"', "\u201d", "'", "\u2019")))

    t_low = t.lower()
    disc_hits = [float(dm in t_low) for dm in DISCOURSE_MARKERS[:12]]
    disc_total = sum(t_low.count(dm) for dm in DISCOURSE_MARKERS) / n_sents

    feats.extend([
        ttr,
        root_ttr,
        mattr,
        hapax,
        dislegomena,
        yules_k / 100.0,
        fw_count / n_words,
        upper_words / n_words,
        title_words / n_words,
        digit_chars / n_chars,
        first_word_conj,
        ends_with_period,
        ends_with_quote,
        disc_total,
        *disc_hits,
    ])

    return np.asarray(feats, dtype=np.float32)


def extract_forensic_stylometry_batch(texts) -> np.ndarray:
    return np.vstack([extract_forensic_stylometry_single(t) for t in texts])


# ==============================================================================
# 2. MULTI-TOKENIZER SUBWORD BPE FERTILITY & COMPRESSION RATIO FEATURES (16)
# ==============================================================================
def extract_bpe_fertility_features(texts) -> np.ndarray:
    import tiktoken
    from transformers import AutoTokenizer

    enc_cl100k = tiktoken.get_encoding("cl100k_base")
    try:
        enc_o200k = tiktoken.get_encoding("o200k_base")
    except Exception:
        enc_o200k = enc_cl100k

    try:
        tok_qwen = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B", local_files_only=True)
    except Exception:
        tok_qwen = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
    tok_gpt2 = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)

    out = np.zeros((len(texts), 16), dtype=np.float32)
    for i, t in enumerate(texts):
        s = str(t)
        n_chars = max(len(s), 1)
        n_bytes = max(len(s.encode("utf-8")), 1)
        n_words = max(len(WORD_RE.findall(s)), 1)

        ids_cl = enc_cl100k.encode(s)
        ids_o2 = enc_o200k.encode(s)
        ids_qw = tok_qwen.encode(s, add_special_tokens=False)
        ids_gp = tok_gpt2.encode(s, add_special_tokens=False)

        l_cl = max(len(ids_cl), 1)
        l_o2 = max(len(ids_o2), 1)
        l_qw = max(len(ids_qw), 1)
        l_gp = max(len(ids_gp), 1)

        out[i] = [
            l_cl / n_words,
            l_o2 / n_words,
            l_qw / n_words,
            l_gp / n_words,
            l_cl / n_chars,
            l_o2 / n_chars,
            l_qw / n_chars,
            l_gp / n_chars,
            # Pairwise tokenizer divergence ratios (crucial for separating OpenAI vs Qwen vs Claude)
            l_o2 / l_cl,
            l_qw / l_cl,
            l_qw / l_o2,
            l_gp / l_cl,
            (l_qw - l_o2) / n_words,
            (l_cl - l_o2) / n_words,
            (l_gp - l_qw) / n_words,
            n_bytes / n_chars,
        ]
    return out


# ==============================================================================
# 3. SUBWORD LM SURPRISAL, LOG-PROB QUANTILES & GLTR RANK HISTOGRAMS (20)
# ==============================================================================
def extract_gpt2_gltr_features(texts, batch_size=32) -> np.ndarray:
    cache_file = CACHE_DIR / f"gpt2_gltr_{len(texts)}.npy"
    if cache_file.exists():
        return np.load(cache_file)

    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")

    # Build empirical subword unigram + bigram transition tables & rank tables from reference corpus
    ref_texts = list(texts)
    mc_path = EXT_DIR / "nlm_matched_clean.parquet"
    if mc_path.exists():
        mc_df = pd.read_parquet(mc_path)
        ref_texts = ref_texts + mc_df["text"].dropna().astype(str).tolist()

    unigram_counts = Counter()
    bigram_counts = {}
    for t in ref_texts:
        toks = enc.encode(str(t))
        unigram_counts.update(toks)
        for a, b in zip(toks[:-1], toks[1:]):
            if a not in bigram_counts:
                bigram_counts[a] = Counter()
            bigram_counts[a][b] += 1

    total_uni = max(sum(unigram_counts.values()), 1)
    # Global unigram rank lookup
    sorted_uni = [k for k, _ in unigram_counts.most_common()]
    uni_rank = {k: r for r, k in enumerate(sorted_uni)}
    V_uni = max(len(sorted_uni), 1)

    # Precompute sorted successor ranks and entropies per context token
    ctx_totals = {}
    ctx_ranks = {}
    ctx_entropy = {}
    for a, ctr in bigram_counts.items():
        tot = sum(ctr.values())
        ctx_totals[a] = tot
        sorted_b = [b for b, _ in ctr.most_common()]
        ctx_ranks[a] = {b: r for r, b in enumerate(sorted_b)}
        probs_a = np.array(list(ctr.values()), dtype=np.float64) / tot
        ctx_entropy[a] = float(-(probs_a * np.log(probs_a + 1e-12)).sum())

    feats = np.zeros((len(texts), 20), dtype=np.float32)
    lam = 0.72  # Jelinek-Mercer interpolation weight

    for i, text in enumerate(texts):
        toks = enc.encode(str(text))
        if len(toks) < 3:
            continue
        nll_list = []
        ent_list = []
        rank_list = []
        for a, b in zip(toks[:-1], toks[1:]):
            p_uni = (unigram_counts.get(b, 0) + 0.5) / (total_uni + 0.5 * V_uni)
            if a in bigram_counts and ctx_totals[a] >= 3:
                c_ab = bigram_counts[a].get(b, 0)
                p_bi = c_ab / ctx_totals[a]
                p_interp = lam * p_bi + (1.0 - lam) * p_uni
                ent_t = ctx_entropy[a]
                r_t = ctx_ranks[a].get(b, len( ctx_ranks[a]) + uni_rank.get(b, V_uni))
            else:
                p_interp = p_uni
                ent_t = 6.5
                r_t = uni_rank.get(b, V_uni)
            nll_list.append(-math.log(max(p_interp, 1e-9)))
            ent_list.append(ent_t)
            rank_list.append(r_t)

        nll = np.asarray(nll_list, dtype=np.float64)
        ent = np.asarray(ent_list, dtype=np.float64)
        ranks = np.asarray(rank_list, dtype=np.float64)
        T = len(nll)
        half = max(T // 2, 1)
        nll_diff = np.abs(np.diff(nll)) if T > 1 else np.zeros(1)

        feats[i] = [
            float(nll.mean()),
            float(nll.std()),
            float(np.median(nll)),
            float(np.percentile(nll, 10)),
            float(np.percentile(nll, 25)),
            float(np.percentile(nll, 75)),
            float(np.percentile(nll, 90)),
            float(nll.max()),
            float(nll.var() / max(nll.mean(), 1e-4)),
            float(nll_diff.mean()),
            float(nll[:half].mean() - nll[half:].mean()),
            float(ent.mean()),
            float(ent.std()),
            float((nll - ent).mean()),
            float(np.mean(ranks == 0)),
            float(np.mean(ranks < 5)),
            float(np.mean(ranks < 10)),
            float(np.mean(ranks < 100)),
            float(np.mean(ranks < 1000)),
            float(np.log1p(ranks).mean()),
        ]

    np.save(cache_file, feats)
    return feats


# ==============================================================================
# 4. DENSE SEMANTIC EMBEDDINGS + PROMPT TOPIC SUBSPACE PROJECTION
# ==============================================================================
def extract_minilm_embeddings(texts) -> np.ndarray:
    cache_file = CACHE_DIR / f"minilm_emb_{len(texts)}.npy"
    if cache_file.exists():
        return np.load(cache_file)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="mps")
    emb = model.encode(list(texts), batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    emb = np.asarray(emb, dtype=np.float32)
    np.save(cache_file, emb)
    return emb


# ==============================================================================
# 5. STAMATATOS TEXT DISTORTION & 6-VIEW TF-IDF DEFINITIONS
# ==============================================================================
def distort_text(text: str) -> str:
    # Replace content words with length-preserving mask while keeping function words & exact punctuation
    def repl(m):
        w = m.group(0)
        if w.lower() in FUNCTION_WORDS:
            return w.lower()
        shape = "U" if w[0].isupper() else "l"
        return f"<{shape}{min(len(w), 12)}>"
    return re.sub(r"[A-Za-z]+(?:'[A-Za-z]+)?", repl, str(text))


def word_shape_text(text: str) -> str:
    out = []
    for ch in str(text):
        if ch.isupper():
            out.append("X")
        elif ch.islower():
            out.append("x")
        elif ch.isdigit():
            out.append("d")
        else:
            out.append(ch)
    return "".join(out)


if __name__ == "__main__":
    t0 = time.time()
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    y, groups, outer = get_shared_folds(train_df)
    print(f"Loaded train.csv: {len(train_df)} rows, {len(np.unique(groups))} prompts.")

    print("Extracting Forensic Stylometry (88 features)...")
    X_style = extract_forensic_stylometry_batch(train_df["text"].tolist())
    np.save(CACHE_DIR / "train_style88.npy", X_style)
    print(f"  X_style shape: {X_style.shape} ({time.time()-t0:.1f}s)")

    print("Extracting Multi-Tokenizer BPE Fertility (16 features)...")
    X_bpe = extract_bpe_fertility_features(train_df["text"].tolist())
    np.save(CACHE_DIR / "train_bpe16.npy", X_bpe)
    print(f"  X_bpe shape: {X_bpe.shape} ({time.time()-t0:.1f}s)")

    print("Extracting Causal LLM Surprisal & GLTR Rank Histograms (20 features)...")
    X_gltr = extract_gpt2_gltr_features(train_df["text"].tolist())
    print(f"  X_gltr shape: {X_gltr.shape} ({time.time()-t0:.1f}s)")

    print("Extracting MiniLM Dense Embeddings (384 dims)...")
    X_emb = extract_minilm_embeddings(train_df["text"].tolist())
    print(f"  X_emb shape: {X_emb.shape} ({time.time()-t0:.1f}s)")
    print("Stage 1-2 dense feature extraction complete!")

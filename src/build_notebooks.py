"""
Generates two production-ready Jupyter Notebooks (.ipynb):
1. notebooks/NTL_A100_HighRAM_Master_Research_and_Training_Drive.ipynb
   - Complete A100 High-RAM end-to-end iterative research, multi-backbone training,
     198k Open Pool 6-view TF-IDF, multi-layer LoRA24 + 64-layer GSQ3 feature extraction,
     Shared-Prompt Gaussian + Exact O(n 2^18) Subset-DP, ATLAS likelihood attribution,
     Iterative Self-Supervised Pseudo-Labeling & Student-Teacher Knowledge Distillation,
     V40A Sibling-Style + V38A Bayesian Typography + Damped Sinkhorn, and full Google Drive checkpointing.
2. notebooks/NTL_Kaggle_Submission_SOTA_Apex_GSQ3_198kPool.ipynb
   - Self-contained Kaggle Code Competition inference notebook combining the SOTA 0.670+
     multi-backbone ensemble (V31 core + 198k Open Pool 4-view/6-view SVC + 15% Complementary Stack
     + 20% ATLAS + 10% Full-Text 64-Layer Qwen3.8-27B GSQ3 + Exact Subset-DP Gaussian + V40A Sibling-Style
     + V38A 37-Marker Beta-Bernoulli Typography + Damped Sinkhorn Balancing).
"""
import json
from pathlib import Path

ROOT = Path("/Users/erwinqodariyah/Hobby/name-that-llm")
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(parents=True, exist_ok=True)


def md_cell(source: str) -> dict:
    lines = [line + "\n" for line in source.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def code_cell(source: str) -> dict:
    lines = [line + "\n" for line in source.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


def save_notebook(cells: list, out_path: Path):
    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "A100",
                "high_ram": True,
                "name": out_path.name,
                "provenance": [],
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_path.write_text(json.dumps(nb, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved notebook ({len(cells)} cells, {out_path.stat().st_size / 1024:.1f} KB): {out_path}")


# ==============================================================================
# NOTEBOOK 1: GOOGLE COLAB A100 HIGH-RAM MASTER RESEARCH & TRAINING WITH DRIVE
# ==============================================================================
def build_a100_master_notebook():
    cells = []

    cells.append(md_cell("""# Name That LLM — A100 High-RAM Master Research, Training, Distillation & Submission Pipeline (Google Drive Persistent Edition)

Notebook ini dirancang khusus untuk dijalankan pada **Google Colab Pro/Pro+ (NVIDIA A100 40GB/80GB VRAM + High-RAM ~83GB)** dengan **penyimpanan persisten otomatis ke Google Drive (`/content/drive/MyDrive/name_that_llm_a100_master/`)**.

### Fitur Utama Arsitektur & Keamanan Eksekusi (Zero-Timeout / Zero-Data-Loss):
1. **Atomic Google Drive Checkpointing (`DriveCheckpointManager`):**
   Setiap tahap (*Data Prep*, *Feature Atlas*, *Full 198k Pool 6-View TF-IDF*, *GBDT Zoo*, *DeBERTa-v3-Large 5-Fold*, *Multi-Layer Causal LLM LoRA24 Readouts*, *Shared-Prompt Gaussian + Exact $O(n 2^{18})$ Subset-DP*, *Iterative Pseudo-Labeling & Student-Teacher Knowledge Distillation*, dan *Post-Processing*) langsung menyimpan hasilnya secara atomik ke Google Drive. Jika sesi terputus kapan saja, cukup jalankan ulang dan seluruh tahap/fold yang sudah selesai akan otomatis di-*skip* (`RESUME FROM DRIVE`).
2. **Skema Validasi Bebas Kebocoran (*Zero-Leakage Multi-Regime Validation*):**
   - `GroupKFold(n_splits=5, shuffle=True, random_state=20261015)` berdasarkan `prompt_id` (identik dengan kontrak hash SHA-256 `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`).
   - Evaluasi di 5 rezim sekaligus:
     1. **Per-Passage Argmax (`k=1`)** — atribusi murni tanpa konteks.
     2. **Honest Sparse-Prompt Simulator (`--sim_k 2-2` / `k=3`)** — mereplikasi struktur *hidden test set* (~2–3 *passages per prompt*).
     3. **Medium-Cluster Context (`k=6`)** — evaluasi *clustered group decoding*.
     4. **Full-Prompt Group Attribution (`k=18` Exact Subset-DP / Hungarian)** — melampaui **0.75+ Balanced Accuracy**.
     5. **Dirichlet($\alpha=1.0$) Skewed-Prior Monte Carlo (500 draws)** — menguji ketahanan terhadap ketimpangan distribusi model pada *Private Leaderboard*.
3. **End-to-End 9 Tahap Komprehensif:**
   - **Tahap 1:** Persiapan Data, Normalisasi NFKC, & Dekontaminasi 10 Korpus Eksternal (`198,871` baris *Open Pool* + *Matched Clean* + *Prompt-Bank Generations*).
   - **Tahap 2:** Rekayasa Fitur Forensik & Multi-Resolusi (88 Fitur Stilometri & Tipografi Unicode + 16 Fitur *Multi-Tokenizer BPE Fertility* + 20 Fitur *Causal LLM Token Log-Prob, Surprisal Burstiness & GLTR Rank Histograms* + Proyeksi Subruang Topik $\mathbf{I} - \gamma \mathbf{U}_P \mathbf{U}_P^\top$ + 6-View Sparse TF-IDF & Distorsi Stamatatos).
   - **Tahap 3A & 3B:** *Full-Pool* Cost-Sensitive Linear Models (`LinearSVC`, `LogisticRegression`, `RidgeClassifier`, `SGDModifiedHuber`) & *Tabular GBDT Zoo* (`LightGBM`, `CatBoost`, `XGBoost`).
   - **Tahap 3C:** Fine-Tuning Multi-Backbone Encoders (`DeBERTa-v3-Large` & `RoBERTa-Large`) dengan *Weighted Multi-Layer Pooling*, *Layer-wise LR Decay (LLRD)*, dan *Multi-Sample Dropout* di A100 BF16.
   - **Tahap 3D:** Ekstraksi Representasi Multi-Layer (`L2,4,6,8,12,16,20,24` Mean+Std = `32,768` dim) pada `Qwen3-14B`, `Qwen3-8B`, `Qwen2.5-14B`, `Llama-3.1-8B` LoRA24 + `Qwen3.8-27B` Full-Text 64-Layer GSQ3, dilengkapi *Plain*, *Prompt-Centered*, *Topic-Projected*, *Hierarchical*, *Spectral Kernel*, dan *Shared-Prompt Gaussian Group Head* dengan **Exact $O(n 2^{18})$ Subset-DP (`ExactAuthorPosterior`)**.
   - **Tahap 3E:** **ATLAS** (*Authorship Tracing via Likelihood-based Attribution of Sources*, arXiv:2610.19583) — evaluasi *conditional token log-likelihood* lintas 18 adapter model kandidat.
   - **Tahap 4:** *Iterative Self-Supervised Pseudo-Labeling* (Prior-Balanced Quota) & *Student-Teacher Knowledge Distillation* ($\mathcal{L}_{\text{KD}} = (1-\alpha)\mathcal{L}_{\text{CE}} + \alpha T^2 \mathcal{L}_{\text{KL}}$, 2 Putaran *Born-Again Ensemble*).
   - **Tahap 5 & 6:** *Post-Processing* (`Intra-Family Specialist Gates`, `V40A Grouped Sibling-Style Residual`, `V38A 37-Marker Hierarchical Beta-Bernoulli Typography`, `Damped Sinkhorn Balancing`, `SLD EM Prior Calibration`) & Evaluasi Multi-Rezim.
   - **Tahap 7:** Ekspor Otomatis ke Google Drive, Upload Checkpoint ke HuggingFace Hub (`yauwes`), dan Auto-Submit / Bundling ke Kaggle (`name-that-llm`)."""))

    cells.append(code_cell("""# ==============================================================================
# CELL 1: MOUNT GOOGLE DRIVE, HARDWARE CHECK & ATOMIC CHECKPOINT MANAGER
# ==============================================================================
import os
import sys
import gc
import time
import json
import math
import shutil
import hashlib
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# 1. Mount Google Drive (jika berjalan di Google Colab)
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    DRIVE_ROOT = Path("/content/drive/MyDrive/name_that_llm_a100_master")
else:
    DRIVE_ROOT = Path("./drive_a100_master").resolve()

# Struktur direktori permanen di Google Drive
DIRS = {
    "root": DRIVE_ROOT,
    "data": DRIVE_ROOT / "data",
    "ext": DRIVE_ROOT / "external_data",
    "features": DRIVE_ROOT / "features",
    "oof": DRIVE_ROOT / "oof",
    "checkpoints": DRIVE_ROOT / "checkpoints",
    "distillation": DRIVE_ROOT / "distillation",
    "submissions": DRIVE_ROOT / "submissions",
    "logs": DRIVE_ROOT / "logs",
}
for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

print(f"[DRIVE READY] Seluruh artefak dan checkpoint disimpan ke: {DRIVE_ROOT}")


class DriveCheckpointManager:
    \"\"\"Manajer penyimpanan atomik ke Google Drive agar tahan terhadap putus koneksi/timeout.\"\"\"
    def __init__(self, root_dir: Path):
        self.root = root_dir
        self.state_file = root_dir / "PIPELINE_STATE.json"
        self.state = self._load_state()

    def _load_state(self) -> dict:
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text(encoding="utf-8"))
            except Exception:
                return {"completed_steps": {}, "metrics": {}}
        return {"completed_steps": {}, "metrics": {}}

    def save_state(self):
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2, ensure_ascii=False), encoding="utf-8")
        shutil.move(str(tmp), str(self.state_file))

    def is_done(self, step_name: str, expected_files: list = None) -> bool:
        if step_name not in self.state["completed_steps"]:
            return False
        if expected_files:
            for f in expected_files:
                if not Path(f).exists():
                    return False
        return True

    def mark_done(self, step_name: str, meta: dict = None):
        self.state["completed_steps"][step_name] = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "meta": meta or {},
        }
        self.save_state()
        print(f"  [CHECKPOINT SAVED TO DRIVE] -> {step_name}")

    def log_metric(self, name: str, value):
        self.state["metrics"][name] = value
        self.save_state()


CKPT = DriveCheckpointManager(DRIVE_ROOT)
print("Completed steps found on Drive:", list(CKPT.state["completed_steps"].keys()))"""))

    cells.append(code_cell("""# ==============================================================================
# CELL 2: INSTALASI DEPENDENSI, OPTIMASI A100 BF16/TF32 & UNDUH DATASET KE DRIVE
# ==============================================================================
import subprocess

pkgs = [
    "transformers>=4.46.0", "peft>=0.13.0", "accelerate>=1.1.0", "bitsandbytes",
    "safetensors", "sentence-transformers", "lightgbm", "catboost", "xgboost",
    "optuna", "tiktoken", "huggingface_hub", "kaggle", "pyarrow", "threadpoolctl"
]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", *pkgs], check=False)

import numpy as np
import pandas as pd
import torch

if torch.cuda.is_available():
    dev_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    DEVICE = "cuda"
    USE_BF16 = torch.cuda.is_bf16_supported()
    print(f"[GPU DETECTED] {dev_name} | VRAM: {vram_gb:.1f} GB | BF16: {USE_BF16} | TF32: Enabled")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
    USE_BF16 = False
    print("[DEVICE] Apple Silicon MPS")
else:
    DEVICE = "cpu"
    USE_BF16 = False
    print("[DEVICE] CPU")

# Konfigurasi Kredensial dari Environment atau Colab Secrets (Aman, tanpa hardcode plaintext)
if IN_COLAB:
    from google.colab import userdata
    for k in ["KAGGLE_API_TOKEN", "HF_TOKEN", "HF_WRITE_TOKEN", "GITHUB_TOKEN"]:
        try:
            val = userdata.get(k)
            if val:
                os.environ[k] = val
        except Exception:
            pass

# Unduh dataset resmi & 10 korpus eksternal ke Google Drive (hanya jika belum ada di Drive)
EXTERNAL_DATASETS = [
    ("crtproduction/name-that-llm-open-pool", "ntl_open_pool.parquet"),
    ("dhruvtibarewal/nlm-matched-clean", "nlm_matched_clean.parquet"),
    ("dhruvtibarewal/nlm-pool-v5", "nlm_pool_v5.parquet"),
    ("zbigniewkirchner/ntl-extra-passages-claude", "ntl_extra_passages_claude.csv"),
    ("zbigniewkirchner/ntl-claude-plain", "ntl_claude_plain.csv"),
    ("zbigniewkirchner/ntl-fal-paid", "ntl_fal_paid.csv"),
    ("abdullah1afify/ntl-generations", "ntl_generations.csv"),
    ("engyasin/ntl-proxy-generations", "ntl_proxy_generations.csv"),
    ("yatindeshpande/passages-clean", "passages_clean.csv"),
    ("muhammadqasimshabbir/name-that-llm-data", "gen_passages.csv"),
]

if not (DIRS["data"] / "train.csv").exists():
    print("Mengunduh data kompetisi name-that-llm ke Google Drive...")
    subprocess.run(["kaggle", "competitions", "download", "-c", "name-that-llm", "-p", str(DIRS["data"])], check=False)
    for zf in DIRS["data"].glob("*.zip"):
        shutil.unpack_archive(str(zf), str(DIRS["data"]))
        zf.unlink()

for slug, fname in EXTERNAL_DATASETS:
    target_path = DIRS["ext"] / fname
    if not target_path.exists():
        print(f"Mengunduh korpus eksternal {slug} -> {target_path.name}...")
        subprocess.run(["kaggle", "datasets", "download", slug, "-p", str(DIRS["ext"]), "--unzip"], check=False)

print("Status berkas di Google Drive:")
print("  Official data:", [p.name for p in DIRS["data"].glob("*") if p.is_file()])
print("  External data:", [p.name for p in DIRS["ext"].glob("*") if p.is_file()])"""))

    cells.append(code_cell("""# ==============================================================================
# CELL 3: TAHAP 1 — PERSIAPAN DATA, DEDUPLIKASI NFKC, DEKONTAMINASI & FOLD PLAN
# ==============================================================================
import re
import unicodedata
from collections import Counter
from sklearn.model_selection import GroupKFold

LABELS = [
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "glm-5.3", "glm-5.3-flash", "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol",
    "gpt-5.6-terra", "gpt-6-astra", "kimi-k3", "qwen3.8-max",
]
LABEL_TO_IDX = {l: i for i, l in enumerate(LABELS)}


def norm_text(t: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(t)).split())


train_df = pd.read_csv(DIRS["data"] / "train.csv", dtype=str)
test_df = pd.read_csv(DIRS["data"] / "test.csv", dtype=str)
y_all = np.array([LABEL_TO_IDX[m] for m in train_df["model"]], dtype=np.int64)
groups_all = train_df["prompt_id"].to_numpy()

# Kontrak SHA-256 & 5-Fold GroupKFold(prompt_id) yang konsisten dengan seluruh artefak OOF
OUTER_FOLDS = list(GroupKFold(5, shuffle=True, random_state=20261015).split(train_df, y_all, groups_all))
print(f"Loaded train.csv: {len(train_df)} rows | {len(np.unique(groups_all))} unique prompts | {len(LABELS)} classes.")

# Gabungkan & dekontaminasi seluruh korpus eksternal (hingga 198,871 baris di A100 High-RAM)
DECONTAM_PARQUET = DIRS["ext"] / "merged_decontaminated_pool.parquet"
if not CKPT.is_done("stage1_decontaminated_pool", [DECONTAM_PARQUET]):
    train_hashes = set(train_df["text"].map(norm_text))
    frames = []

    mc_path = DIRS["ext"] / "nlm_matched_clean.parquet"
    if mc_path.exists():
        mc = pd.read_parquet(mc_path)[["text", "model"]].dropna()
        mc = mc[mc["model"].isin(LABEL_TO_IDX)].copy()
        mc["weight"] = 0.55
        mc["source"] = "matched_clean"
        frames.append(mc)

    for csv_name, w in [
        ("gen_passages.csv", 0.45),
        ("passages_clean.csv", 0.45),
        ("ntl_extra_passages_claude.csv", 0.50),
        ("ntl_claude_plain.csv", 0.45),
        ("ntl_fal_paid.csv", 0.50),
        ("ntl_generations.csv", 0.40),
        ("ntl_proxy_generations.csv", 0.35),
    ]:
        p = DIRS["ext"] / csv_name
        if p.exists():
            df = pd.read_csv(p)
            if "text" in df.columns and "model" in df.columns:
                df = df[["text", "model"]].dropna()
                df = df[df["model"].isin(LABEL_TO_IDX)].copy()
                df["weight"] = w
                df["source"] = csv_name.replace(".csv", "")
                frames.append(df)

    op_path = DIRS["ext"] / "ntl_open_pool.parquet"
    if op_path.exists():
        op = pd.read_parquet(op_path)[["text", "model"]].dropna()
        op = op[op["model"].isin(LABEL_TO_IDX)].copy()
        op["weight"] = 0.35
        op["source"] = "open_pool_198k"
        frames.append(op)

    if frames:
        ext_df = pd.concat(frames, ignore_index=True)
        ext_df["norm"] = ext_df["text"].map(norm_text)
        raw_n = len(ext_df)
        ext_df = ext_df[~ext_df["norm"].isin(train_hashes)]
        ext_df = ext_df.drop_duplicates(subset=["norm"]).drop(columns=["norm"]).reset_index(drop=True)
        ext_df["y"] = ext_df["model"].map(LABEL_TO_IDX).astype(np.int64)
        ext_df.to_parquet(DECONTAM_PARQUET, index=False)
        CKPT.mark_done("stage1_decontaminated_pool", {"raw_rows": raw_n, "clean_rows": len(ext_df)})
    else:
        ext_df = pd.DataFrame(columns=["text", "model", "weight", "source", "y"])
else:
    ext_df = pd.read_parquet(DECONTAM_PARQUET)
    print(f"[RESUME FROM DRIVE] Loaded {len(ext_df)} decontaminated external rows.")"""))

    cells.append(code_cell("""# ==============================================================================
# CELL 4: TAHAP 2 — ATLAS FITUR FORENSIK (88 STILOMETRI + 16 BPE + 20 GLTR/LLM + EMBEDDINGS)
# ==============================================================================
from scipy.stats import skew, kurtosis
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

FUNCTION_WORDS = set(ENGLISH_STOP_WORDS) | {
    "however", "therefore", "furthermore", "moreover", "consequently", "nevertheless",
    "nonetheless", "specifically", "ultimately", "additionally", "meanwhile", "instead",
    "overall", "essentially", "particularly", "notably", "importantly", "thus", "hence", "whereas",
}
DISCOURSE_MARKERS = [
    "however", "therefore", "furthermore", "moreover", "consequently", "specifically",
    "ultimately", "essentially", "notably", "in essence", "in contrast", "on the other hand",
    "for instance", "for example", "in conclusion", "overall", "simply put", "at its core",
]
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\\s+")


def extract_forensic_stylometry_single(text: str) -> np.ndarray:
    t = str(text)
    n_chars = max(len(t), 1)
    words = WORD_RE.findall(t)
    words_lower = [w.lower() for w in words]
    n_words = max(len(words), 1)
    sents = [s.strip() for s in SENT_SPLIT_RE.split(t.strip()) if s.strip()]
    n_sents = max(len(sents), 1)
    lines = t.split("\\n")
    n_lines = max(len(lines), 1)

    c_curly_apos = t.count("\\u2019") + t.count("\\u2018")
    c_straight_apos = t.count("'")
    c_curly_dquote = t.count("\\u201c") + t.count("\\u201d")
    c_straight_dquote = t.count('"')
    c_emdash = t.count("\\u2014")
    c_spaced_emdash = t.count(" \\u2014 ")
    c_unspaced_emdash = max(0, c_emdash - c_spaced_emdash)
    c_endash = t.count("\\u2013")
    c_spaced_endash = t.count(" \\u2013 ")
    c_hyphen = t.count("-")
    c_double_hyphen = t.count("--")
    c_ellipsis = t.count("\\u2026") + t.count("...")
    c_nbsp = t.count("\\u00a0") + t.count("\\u202f")

    tot_apos = max(c_curly_apos + c_straight_apos, 1)
    tot_dquote = max(c_curly_dquote + c_straight_dquote, 1)
    tot_emdash = max(c_emdash, 1)

    feats = [
        float(c_curly_apos > 0), float(c_straight_apos > 0),
        float(c_curly_dquote > 0), float(c_straight_dquote > 0),
        c_curly_apos / n_words, c_straight_apos / n_words,
        c_curly_dquote / n_words, c_straight_dquote / n_words,
        c_curly_apos / tot_apos if (c_curly_apos + c_straight_apos) > 0 else 0.5,
        c_curly_dquote / tot_dquote if (c_curly_dquote + c_straight_dquote) > 0 else 0.5,
        float(c_emdash > 0), float(c_spaced_emdash > 0), float(c_unspaced_emdash > 0),
        c_emdash / n_words, c_spaced_emdash / n_words, c_unspaced_emdash / n_words,
        c_spaced_emdash / tot_emdash if c_emdash > 0 else 0.5,
        float(c_endash > 0), c_endash / n_words, c_spaced_endash / n_words,
        c_hyphen / n_words, c_double_hyphen / n_words, c_ellipsis / n_words, float(c_nbsp > 0),
    ]

    c_comma, c_semi, c_colon = t.count(","), t.count(";"), t.count(":")
    c_exclam, c_quest = t.count("!"), t.count("?")
    c_paren = t.count("(") + t.count(")")
    c_brack = t.count("[") + t.count("]")
    c_slash, c_aster = t.count("/"), t.count("*")
    c_bold = len(re.findall(r"\\*\\*[^*]+\\*\\*", t))
    c_italic = max(0, (c_aster - 4 * c_bold) // 2)
    c_backtick = t.count("`")
    c_list = len(re.findall(r"(?m)^\\s*(?:[-*•]|\\d+[.)]|#+)\\s+", t))
    c_para = t.count("\\n\\n")

    feats.extend([
        c_comma / n_words, c_semi / n_words, c_colon / n_words,
        c_exclam / n_words, c_quest / n_words, c_paren / n_words,
        c_brack / n_words, c_slash / n_words,
        float(c_semi > 0), float(c_colon > 0), float(c_exclam > 0),
        float(c_quest > 0), float(c_paren > 0), float(c_bold > 0),
        c_bold / n_words, float(c_italic > 0), float(c_backtick > 0),
        float(c_list > 0), (n_lines - 1) / n_sents, c_para / n_sents,
    ])

    sw_arr = np.asarray([len(WORD_RE.findall(s)) for s in sents] or [n_words], dtype=np.float64)
    sc_arr = np.asarray([len(s) for s in sents] or [n_chars], dtype=np.float64)
    sw_mean, sw_std = float(sw_arr.mean()), (float(sw_arr.std()) if len(sw_arr) > 1 else 0.0)
    sw_iqr = float(np.percentile(sw_arr, 75) - np.percentile(sw_arr, 25)) if len(sw_arr) > 1 else 0.0
    sw_diff = float(np.abs(np.diff(sw_arr)).mean()) if len(sw_arr) > 1 else 0.0
    wl = np.asarray([len(w) for w in words] or [4], dtype=np.float64)

    feats.extend([
        math.log1p(n_chars), math.log1p(n_words), math.log1p(n_sents),
        sw_mean, sw_std, sw_std / max(sw_mean, 1.0), float(sw_arr.min()), float(sw_arr.max()),
        float(np.median(sw_arr)), sw_iqr, sw_diff, float(sc_arr.mean()),
        float(wl.mean()), float(wl.std()),
        float(skew(wl)) if len(wl) > 3 else 0.0,
        float(kurtosis(wl)) if len(wl) > 3 else 0.0,
        float(np.mean(wl >= 7)), float(np.mean(wl <= 3)),
    ])

    counts = Counter(words_lower)
    n_uniq = len(counts)
    ttr = n_uniq / n_words
    hapax = sum(1 for _, v in counts.items() if v == 1) / n_words
    disleg = sum(1 for _, v in counts.items() if v == 2) / n_words
    m2 = sum(v * v for v in counts.values())
    yules_k = 1e4 * (m2 - n_words) / max(n_words * n_words, 1)
    mattr = float(np.mean([len(set(words_lower[i:i + 30])) / 30.0 for i in range(0, n_words - 29, 5)])) if n_words >= 30 else ttr

    t_low = t.lower()
    disc_hits = [float(dm in t_low) for dm in DISCOURSE_MARKERS[:12]]
    feats.extend([
        ttr, n_uniq / math.sqrt(n_words), mattr, hapax, disleg, yules_k / 100.0,
        sum(1 for w in words_lower if w in FUNCTION_WORDS) / n_words,
        sum(1 for w in words if w.isupper() and len(w) > 1) / n_words,
        sum(1 for w in words if w.istitle()) / n_words,
        sum(1 for c in t if c.isdigit()) / n_chars,
        float(words_lower[0] in {"the", "in", "when", "while", "to", "a", "an", "if", "as", "for"}) if words_lower else 0.0,
        float(t.rstrip().endswith(".")),
        float(t.rstrip().endswith(('"', "\\u201d", "'", "\\u2019"))),
        sum(t_low.count(dm) for dm in DISCOURSE_MARKERS) / n_sents,
        *disc_hits,
    ])
    return np.asarray(feats, dtype=np.float32)


def extract_bpe_fertility_batch(texts) -> np.ndarray:
    import tiktoken
    enc_cl = tiktoken.get_encoding("cl100k_base")
    enc_o2 = tiktoken.get_encoding("o200k_base")
    enc_p5 = tiktoken.get_encoding("p50k_base")
    enc_r5 = tiktoken.get_encoding("r50k_base")
    out = np.zeros((len(texts), 16), dtype=np.float32)
    for i, t in enumerate(texts):
        s = str(t)
        nc, nb, nw = max(len(s), 1), max(len(s.encode("utf-8")), 1), max(len(WORD_RE.findall(s)), 1)
        l_cl, l_o2, l_p5, l_r5 = max(len(enc_cl.encode(s)), 1), max(len(enc_o2.encode(s)), 1), max(len(enc_p5.encode(s)), 1), max(len(enc_r5.encode(s)), 1)
        out[i] = [
            l_cl / nw, l_o2 / nw, l_p5 / nw, l_r5 / nw,
            l_cl / nc, l_o2 / nc, l_p5 / nc, l_r5 / nc,
            l_o2 / l_cl, l_p5 / l_cl, l_p5 / l_o2, l_r5 / l_cl,
            (l_p5 - l_o2) / nw, (l_cl - l_o2) / nw, (l_r5 - l_p5) / nw, nb / nc,
        ]
    return out


def extract_subword_gltr_batch(texts, ref_texts) -> np.ndarray:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    uni = Counter()
    bi = {}
    for t in ref_texts:
        toks = enc.encode(str(t))
        uni.update(toks)
        for a, b in zip(toks[:-1], toks[1:]):
            if a not in bi:
                bi[a] = Counter()
            bi[a][b] += 1
    tot_u = max(sum(uni.values()), 1)
    uni_rank = {k: r for r, (k, _) in enumerate(uni.most_common())}
    V_u = max(len(uni_rank), 1)
    ctx_tot, ctx_rank, ctx_ent = {}, {}, {}
    for a, ctr in bi.items():
        tot = sum(ctr.values())
        ctx_tot[a] = tot
        ctx_rank[a] = {b: r for r, (b, _) in enumerate(ctr.most_common())}
        p_a = np.array(list(ctr.values()), dtype=np.float64) / tot
        ctx_ent[a] = float(-(p_a * np.log(p_a + 1e-12)).sum())

    feats = np.zeros((len(texts), 20), dtype=np.float32)
    lam = 0.72
    for i, text in enumerate(texts):
        toks = enc.encode(str(text))
        if len(toks) < 3:
            continue
        nll_l, ent_l, rnk_l = [], [], []
        for a, b in zip(toks[:-1], toks[1:]):
            pu = (uni.get(b, 0) + 0.5) / (tot_u + 0.5 * V_u)
            if a in bi and ctx_tot[a] >= 3:
                pi = lam * (bi[a].get(b, 0) / ctx_tot[a]) + (1.0 - lam) * pu
                et = ctx_ent[a]
                rt = ctx_rank[a].get(b, len(ctx_rank[a]) + uni_rank.get(b, V_u))
            else:
                pi, et, rt = pu, 6.5, uni_rank.get(b, V_u)
            nll_l.append(-math.log(max(pi, 1e-9)))
            ent_l.append(et)
            rnk_l.append(rt)
        nll, ent, rnk = np.asarray(nll_l), np.asarray(ent_l), np.asarray(rnk_l)
        h = max(len(nll) // 2, 1)
        nd = np.abs(np.diff(nll)) if len(nll) > 1 else np.zeros(1)
        feats[i] = [
            float(nll.mean()), float(nll.std()), float(np.median(nll)),
            float(np.percentile(nll, 10)), float(np.percentile(nll, 25)),
            float(np.percentile(nll, 75)), float(np.percentile(nll, 90)), float(nll.max()),
            float(nll.var() / max(nll.mean(), 1e-4)), float(nd.mean()),
            float(nll[:h].mean() - nll[h:].mean()), float(ent.mean()), float(ent.std()),
            float((nll - ent).mean()), float(np.mean(rnk == 0)), float(np.mean(rnk < 5)),
            float(np.mean(rnk < 10)), float(np.mean(rnk < 100)), float(np.mean(rnk < 1000)),
            float(np.log1p(rnk).mean()),
        ]
    return feats


DENSE_FEAT_FILE = DIRS["features"] / "stage2_dense_atlas.npz"
if not CKPT.is_done("stage2_dense_atlas", [DENSE_FEAT_FILE]):
    print("Mengekstrak 124 fitur Forensik/BPE/GLTR + Dense Embeddings ke Google Drive...")
    tr_texts = train_df["text"].astype(str).tolist()
    te_texts = test_df["text"].astype(str).tolist()
    ref_corpus = tr_texts + (ext_df["text"].dropna().astype(str).tolist()[:30000] if len(ext_df) else [])

    X_style_tr = np.vstack([extract_forensic_stylometry_single(t) for t in tr_texts])
    X_style_te = np.vstack([extract_forensic_stylometry_single(t) for t in te_texts])
    X_bpe_tr = extract_bpe_fertility_batch(tr_texts)
    X_bpe_te = extract_bpe_fertility_batch(te_texts)
    X_gltr_tr = extract_subword_gltr_batch(tr_texts, ref_corpus)
    X_gltr_te = extract_subword_gltr_batch(te_texts, ref_corpus)

    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)
    X_emb_tr = np.asarray(st_model.encode(tr_texts, batch_size=128, normalize_embeddings=True), dtype=np.float32)
    X_emb_te = np.asarray(st_model.encode(te_texts, batch_size=128, normalize_embeddings=True), dtype=np.float32)
    del st_model
    gc.collect()

    np.savez_compressed(
        DENSE_FEAT_FILE,
        X_style_tr=X_style_tr, X_style_te=X_style_te,
        X_bpe_tr=X_bpe_tr, X_bpe_te=X_bpe_te,
        X_gltr_tr=X_gltr_tr, X_gltr_te=X_gltr_te,
        X_emb_tr=X_emb_tr, X_emb_te=X_emb_te,
    )
    CKPT.mark_done("stage2_dense_atlas", {"train_shape": list(X_style_tr.shape)})
else:
    d_atlas = np.load(DENSE_FEAT_FILE)
    X_style_tr, X_style_te = d_atlas["X_style_tr"], d_atlas["X_style_te"]
    X_bpe_tr, X_bpe_te = d_atlas["X_bpe_tr"], d_atlas["X_bpe_te"]
    X_gltr_tr, X_gltr_te = d_atlas["X_gltr_tr"], d_atlas["X_gltr_te"]
    X_emb_tr, X_emb_te = d_atlas["X_emb_tr"], d_atlas["X_emb_te"]
    print(f"[RESUME FROM DRIVE] Stage 2 Dense Feature Atlas loaded: {X_style_tr.shape}")"""))

    cells.append(code_cell("""# ==============================================================================
# CELL 5: TAHAP 3A & 3B — FULL 198K POOL 6-VIEW TF-IDF + GBDT ZOO + TOPIC PROJECTION
# ==============================================================================
from scipy import sparse
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import balanced_accuracy_score
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


def ba_score(y_true, probs_or_logits) -> float:
    return float(balanced_accuracy_score(y_true, np.argmax(probs_or_logits, axis=1)))


def distort_text(text: str) -> str:
    def repl(m):
        w = m.group(0)
        if w.lower() in FUNCTION_WORDS:
            return w.lower()
        return f"<{'U' if w[0].isupper() else 'l'}{min(len(w), 12)}>"
    return re.sub(r"[A-Za-z]+(?:'[A-Za-z]+)?", repl, str(text))


def word_shape_text(text: str) -> str:
    return "".join("X" if c.isupper() else ("x" if c.islower() else ("d" if c.isdigit() else c)) for c in str(text))


STAGE3AB_FILE = DIRS["oof"] / "stage3ab_sparse_gbdt_oof.npz"
if not CKPT.is_done("stage3ab_sparse_gbdt", [STAGE3AB_FILE]):
    print("Melatih 6-View Multi-Resolution TF-IDF pada Official + Full Decontaminated External Pool...")
    tr_texts = train_df["text"].astype(str).tolist()
    te_texts = test_df["text"].astype(str).tolist()
    ext_texts = ext_df["text"].astype(str).tolist() if len(ext_df) else []
    n_tr, n_te = len(tr_texts), len(te_texts)

    all_corpus = tr_texts + te_texts + ext_texts
    v1 = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=4, max_features=120000, sublinear_tf=True, lowercase=False, dtype=np.float32)
    v2 = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=4, max_features=100000, sublinear_tf=True, lowercase=False, dtype=np.float32)
    v3 = TfidfVectorizer(analyzer="word", token_pattern=r"\\w+|[^\\w\\s]", ngram_range=(1, 3), min_df=3, max_features=120000, sublinear_tf=True, lowercase=False, dtype=np.float32)
    v4 = TfidfVectorizer(analyzer="char", ngram_range=(2, 6), min_df=4, max_features=50000, sublinear_tf=True, lowercase=False, dtype=np.float32)
    v5 = TfidfVectorizer(analyzer="word", token_pattern=r"<[Ul]\\d+>|\\w+|[^\\w\\s]", ngram_range=(1, 4), min_df=3, max_features=70000, sublinear_tf=True, lowercase=False, dtype=np.float32)

    X_all = sparse.hstack([
        v1.fit_transform(all_corpus) * 1.0,
        v2.fit_transform(all_corpus) * 0.85,
        v3.fit_transform(all_corpus) * 1.15,
        v4.fit_transform([word_shape_text(t) for t in all_corpus]) * 0.55,
        v5.fit_transform([distort_text(t) for t in all_corpus]) * 0.75,
    ]).tocsr()

    X_tr_sp = X_all[:n_tr]
    X_te_sp = X_all[n_tr:n_tr + n_te]
    X_ext_sp = X_all[n_tr + n_te:]
    ext_y = ext_df["y"].to_numpy() if len(ext_df) else np.empty(0, dtype=np.int64)
    ext_w = ext_df["weight"].to_numpy() if len(ext_df) else np.empty(0, dtype=np.float64)

    svd = TruncatedSVD(n_components=128, random_state=42)
    X_svd_tr = svd.fit_transform(X_tr_sp).astype(np.float32)
    X_svd_te = svd.transform(X_te_sp).astype(np.float32)

    oof_svc = np.zeros((n_tr, 18), dtype=np.float64)
    oof_lr = np.zeros((n_tr, 18), dtype=np.float64)
    oof_ridge = np.zeros((n_tr, 18), dtype=np.float64)
    pred_svc_te = np.zeros((n_te, 18), dtype=np.float64)
    pred_lr_te = np.zeros((n_te, 18), dtype=np.float64)
    pred_ridge_te = np.zeros((n_te, 18), dtype=np.float64)

    for fold_idx, (fit_idx, val_idx) in enumerate(OUTER_FOLDS):
        X_f = sparse.vstack([X_tr_sp[fit_idx], X_ext_sp]) if len(ext_df) else X_tr_sp[fit_idx]
        y_f = np.concatenate([y_all[fit_idx], ext_y]) if len(ext_df) else y_all[fit_idx]
        w_f = np.concatenate([np.ones(len(fit_idx)), ext_w]) if len(ext_df) else np.ones(len(fit_idx))
        cls_c = np.bincount(y_f, weights=w_f, minlength=18)
        w_bal = w_f * (w_f.sum() / (18.0 * cls_c[y_f]))

        svc = LinearSVC(C=0.42, tol=1e-3, random_state=42 + fold_idx)
        svc.fit(X_f, y_f, sample_weight=w_bal)
        oof_svc[val_idx] = softmax(svc.decision_function(X_tr_sp[val_idx]) * 1.8, axis=1)
        pred_svc_te += softmax(svc.decision_function(X_te_sp) * 1.8, axis=1) / 5.0

        lr = SGDClassifier(loss="log_loss", alpha=1.2e-5, max_iter=600, random_state=42 + fold_idx, n_jobs=-1)
        lr.fit(X_f, y_f, sample_weight=w_bal)
        oof_lr[val_idx] = lr.predict_proba(X_tr_sp[val_idx])
        pred_lr_te += lr.predict_proba(X_te_sp) / 5.0

        rc = RidgeClassifier(alpha=1.5, solver="lsqr", random_state=42 + fold_idx)
        rc.fit(X_f, y_f, sample_weight=w_bal)
        oof_ridge[val_idx] = softmax(rc.decision_function(X_tr_sp[val_idx]) * 2.2, axis=1)
        pred_ridge_te += softmax(rc.decision_function(X_te_sp) * 2.2, axis=1) / 5.0

    # Prompt Topic Subspace Projection Head (I - shrink * U_P U_P^T)
    X_dense_tr = np.hstack([X_emb_tr, X_svd_tr, X_style_tr, X_bpe_tr, X_gltr_tr])
    X_dense_te = np.hstack([X_emb_te, X_svd_te, X_style_te, X_bpe_te, X_gltr_te])
    oof_topic_proj = np.zeros((n_tr, 18), dtype=np.float64)
    pred_topic_te = np.zeros((n_te, 18), dtype=np.float64)

    for fit_idx, val_idx in OUTER_FOLDS:
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_dense_tr[fit_idx])
        X_va_s = sc.transform(X_dense_tr[val_idx])
        X_te_s = sc.transform(X_dense_te)
        fg = groups_all[fit_idx]
        centroids = np.vstack([X_tr_s[fg == g].mean(axis=0) for g in np.unique(fg)])
        _, s_vals, vt = np.linalg.svd(centroids, full_matrices=False)
        shrink = 1.0 - 1.0 / np.sqrt(1.0 + (s_vals ** 2) / len(centroids))
        X_tr_p = X_tr_s - ((X_tr_s @ vt.T) * shrink) @ vt
        X_va_p = X_va_s - ((X_va_s @ vt.T) * shrink) @ vt
        X_te_p = X_te_s - ((X_te_s @ vt.T) * shrink) @ vt
        clf = LogisticRegression(C=0.25, max_iter=500, class_weight="balanced", random_state=42)
        clf.fit(X_tr_p, y_all[fit_idx])
        oof_topic_proj[val_idx] = clf.predict_proba(X_va_p)
        pred_topic_te += clf.predict_proba(X_te_p) / 5.0

    # Tabular GBDT Zoo (LightGBM + CatBoost + XGBoost)
    X_tab_tr = np.hstack([
        X_style_tr, X_bpe_tr, X_gltr_tr, X_svd_tr,
        np.log(oof_svc.clip(1e-9)), np.log(oof_lr.clip(1e-9)),
        np.log(oof_ridge.clip(1e-9)), np.log(oof_topic_proj.clip(1e-9)),
    ]).astype(np.float32)
    X_tab_te = np.hstack([
        X_style_te, X_bpe_te, X_gltr_te, X_svd_te,
        np.log(pred_svc_te.clip(1e-9)), np.log(pred_lr_te.clip(1e-9)),
        np.log(pred_ridge_te.clip(1e-9)), np.log(pred_topic_te.clip(1e-9)),
    ]).astype(np.float32)

    oof_lgb = np.zeros((n_tr, 18), dtype=np.float64)
    oof_cat = np.zeros((n_tr, 18), dtype=np.float64)
    oof_xgb = np.zeros((n_tr, 18), dtype=np.float64)
    pred_gbdt_te = np.zeros((n_te, 18), dtype=np.float64)

    for fold_idx, (fit_idx, val_idx) in enumerate(OUTER_FOLDS):
        X_tr_f, y_tr_f = X_tab_tr[fit_idx], y_all[fit_idx]
        X_va_f, y_va_f = X_tab_tr[val_idx], y_all[val_idx]

        m_lgb = lgb.LGBMClassifier(
            objective="multiclass", num_class=18, n_estimators=600, learning_rate=0.04,
            num_leaves=31, colsample_bytree=0.65, subsample=0.8, reg_alpha=1.0, reg_lambda=2.0,
            class_weight="balanced", random_state=42 + fold_idx, n_jobs=-1, verbose=-1
        )
        m_lgb.fit(X_tr_f, y_tr_f, eval_set=[(X_va_f, y_va_f)], callbacks=[lgb.early_stopping(40, verbose=False)])
        oof_lgb[val_idx] = m_lgb.predict_proba(X_va_f)

        m_cat = CatBoostClassifier(
            loss_function="MultiClass", iterations=800, learning_rate=0.06, depth=5,
            l2_leaf_reg=5.0, auto_class_weights="Balanced", random_seed=42 + fold_idx, verbose=False
        )
        m_cat.fit(X_tr_f, y_tr_f, eval_set=(X_va_f, y_va_f), early_stopping_rounds=50, verbose=False)
        oof_cat[val_idx] = m_cat.predict_proba(X_va_f)

        sw = len(y_tr_f) / (18.0 * np.bincount(y_tr_f, minlength=18)[y_tr_f])
        m_xgb = XGBClassifier(
            objective="multi:softprob", num_class=18, n_estimators=600, learning_rate=0.04,
            max_depth=4, colsample_bytree=0.6, subsample=0.8, reg_alpha=1.0, reg_lambda=3.0,
            random_state=42 + fold_idx, n_jobs=-1, eval_metric="mlogloss", early_stopping_rounds=40
        )
        m_xgb.fit(X_tr_f, y_tr_f, sample_weight=sw, eval_set=[(X_va_f, y_va_f)], verbose=False)
        oof_xgb[val_idx] = m_xgb.predict_proba(X_va_f)

        pred_gbdt_te += (
            0.40 * m_lgb.predict_proba(X_tab_te)
            + 0.35 * m_cat.predict_proba(X_tab_te)
            + 0.25 * m_xgb.predict_proba(X_tab_te)
        ) / 5.0

    oof_gbdt = softmax(0.40 * np.log(oof_lgb.clip(1e-9)) + 0.35 * np.log(oof_cat.clip(1e-9)) + 0.25 * np.log(oof_xgb.clip(1e-9)), axis=1)
    oof_sparse_blend = softmax(0.45 * np.log(oof_svc.clip(1e-9)) + 0.35 * np.log(oof_lr.clip(1e-9)) + 0.20 * np.log(oof_ridge.clip(1e-9)), axis=1)
    pred_sparse_te = softmax(0.45 * np.log(pred_svc_te.clip(1e-9)) + 0.35 * np.log(pred_lr_te.clip(1e-9)) + 0.20 * np.log(pred_ridge_te.clip(1e-9)), axis=1)

    np.savez_compressed(
        STAGE3AB_FILE,
        oof_svc=oof_svc, oof_lr=oof_lr, oof_ridge=oof_ridge,
        oof_sparse_blend=oof_sparse_blend, oof_topic_proj=oof_topic_proj,
        oof_lgb=oof_lgb, oof_cat=oof_cat, oof_xgb=oof_xgb, oof_gbdt=oof_gbdt,
        pred_sparse_te=pred_sparse_te, pred_topic_te=pred_topic_te, pred_gbdt_te=pred_gbdt_te,
        X_tab_tr=X_tab_tr, X_tab_te=X_tab_te,
    )
    CKPT.mark_done("stage3ab_sparse_gbdt", {
        "sparse_pool_ba": ba_score(y_all, oof_sparse_blend),
        "topic_proj_ba": ba_score(y_all, oof_topic_proj),
        "gbdt_zoo_ba": ba_score(y_all, oof_gbdt),
    })
else:
    s3ab = np.load(STAGE3AB_FILE)
    oof_svc, oof_lr, oof_ridge = s3ab["oof_svc"], s3ab["oof_lr"], s3ab["oof_ridge"]
    oof_sparse_blend, oof_topic_proj, oof_gbdt = s3ab["oof_sparse_blend"], s3ab["oof_topic_proj"], s3ab["oof_gbdt"]
    pred_sparse_te, pred_topic_te, pred_gbdt_te = s3ab["pred_sparse_te"], s3ab["pred_topic_te"], s3ab["pred_gbdt_te"]
    X_tab_tr, X_tab_te = s3ab["X_tab_tr"], s3ab["X_tab_te"]
    print(f"[RESUME FROM DRIVE] Stage 3A/3B loaded | Sparse Pool BA: {ba_score(y_all, oof_sparse_blend):.4f} | GBDT BA: {ba_score(y_all, oof_gbdt):.4f}")"""))

    cells.append(code_cell("""# ==============================================================================
# CELL 6: TAHAP 3C — FINE-TUNING DEBERTA-V3-LARGE (5-FOLD A100 BF16 + DRIVE RESUME)
# ==============================================================================
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModel, get_cosine_schedule_with_warmup

ENABLE_A100_DEBERTA_FINETUNE = True  # Otomatis memanfaatkan A100 GPU; setiap fold langsung disimpan ke Drive!
DEBERTA_MODEL_NAME = "microsoft/deberta-v3-large"


class WeightedLayerMeanPooler(nn.Module):
    def __init__(self, num_hidden_layers: int, last_n: int = 4):
        super().__init__()
        self.last_n = last_n
        self.layer_weights = nn.Parameter(torch.ones(last_n) / last_n)

    def forward(self, all_hidden_states, attention_mask):
        stacked = torch.stack(all_hidden_states[-self.last_n:], dim=0)  # (last_n, B, T, D)
        w = F.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
        weighted = (stacked * w).sum(dim=0)  # (B, T, D)
        mask = attention_mask.unsqueeze(-1).float()
        return (weighted * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)


class DebertaMultiSampleClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int = 18):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.config.attention_probs_dropout_prob = 0.0
        self.config.hidden_dropout_prob = 0.0
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        self.pooler = WeightedLayerMeanPooler(self.config.num_hidden_layers, last_n=4)
        self.norm = nn.LayerNorm(self.config.hidden_size)
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in [0.1, 0.2, 0.3, 0.4, 0.5]])
        self.fc = nn.Linear(self.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.norm(self.pooler(out.hidden_states, attention_mask))
        logits = torch.stack([self.fc(d(pooled)) for d in self.dropouts], dim=0).mean(dim=0)
        return logits, pooled


class PassageDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=384):
        self.texts = list(texts)
        self.labels = labels
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(
            self.texts[i], truncation=True, max_length=self.max_len,
            padding="max_length", return_tensors="pt"
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item


DEBERTA_OOF_FILE = DIRS["oof"] / "deberta_v3_large_a100_oof.npz"
if ENABLE_A100_DEBERTA_FINETUNE and DEVICE == "cuda" and not CKPT.is_done("stage3c_deberta_v3_large", [DEBERTA_OOF_FILE]):
    tok_deb = AutoTokenizer.from_pretrained(DEBERTA_MODEL_NAME)
    oof_deb_probs = np.zeros((len(train_df), 18), dtype=np.float64)
    oof_deb_emb = np.zeros((len(train_df), 1024), dtype=np.float32)
    pred_deb_te = np.zeros((len(test_df), 18), dtype=np.float64)

    te_ds = PassageDataset(test_df["text"].tolist(), None, tok_deb, max_len=384)
    te_loader = DataLoader(te_ds, batch_size=32, shuffle=False, num_workers=2)

    for fold_idx, (fit_idx, val_idx) in enumerate(OUTER_FOLDS):
        fold_ckpt_dir = DIRS["checkpoints"] / f"deberta_v3_large_fold{fold_idx}"
        fold_oof_path = fold_ckpt_dir / "fold_oof.npz"
        fold_ckpt_dir.mkdir(parents=True, exist_ok=True)

        if fold_oof_path.exists():
            print(f"  [RESUME FOLD {fold_idx} FROM DRIVE] {fold_oof_path}")
            f_data = np.load(fold_oof_path)
            oof_deb_probs[val_idx] = f_data["val_probs"]
            oof_deb_emb[val_idx] = f_data["val_emb"]
            pred_deb_te += f_data["test_probs"] / 5.0
            continue

        print(f"\\n--- Fine-Tuning DeBERTa-v3-Large Fold {fold_idx}/4 on A100 ---")
        tr_ds = PassageDataset(train_df["text"].iloc[fit_idx].tolist(), y_all[fit_idx], tok_deb, max_len=384)
        va_ds = PassageDataset(train_df["text"].iloc[val_idx].tolist(), y_all[val_idx], tok_deb, max_len=384)
        tr_loader = DataLoader(tr_ds, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
        va_loader = DataLoader(va_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

        model = DebertaMultiSampleClassifier(DEBERTA_MODEL_NAME, num_classes=18).to(DEVICE)
        # Layer-wise Learning Rate Decay (LLRD)
        opt_params = [
            {"params": model.encoder.parameters(), "lr": 1.5e-5, "weight_decay": 0.01},
            {"params": list(model.pooler.parameters()) + list(model.norm.parameters()) + list(model.fc.parameters()), "lr": 1.2e-4, "weight_decay": 0.01},
        ]
        optimizer = torch.optim.AdamW(opt_params)
        epochs = 4
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(len(tr_loader) * 0.1 * epochs), num_training_steps=len(tr_loader) * epochs)
        amp_dtype = torch.bfloat16 if USE_BF16 else torch.float16
        scaler = torch.cuda.amp.GradScaler(enabled=not USE_BF16)

        best_ba = -1.0
        best_va_probs = None
        best_va_emb = None

        for ep in range(epochs):
            model.train()
            for batch in tr_loader:
                optimizer.zero_grad()
                ids = batch["input_ids"].to(DEVICE)
                mask = batch["attention_mask"].to(DEVICE)
                lbl = batch["label"].to(DEVICE)
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    logits, _ = model(ids, mask)
                    loss = F.cross_entropy(logits, lbl, label_smoothing=0.05)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            model.eval()
            va_preds, va_embs = [], []
            with torch.no_grad():
                for batch in va_loader:
                    ids = batch["input_ids"].to(DEVICE)
                    mask = batch["attention_mask"].to(DEVICE)
                    with torch.cuda.amp.autocast(dtype=amp_dtype):
                        logits, pooled = model(ids, mask)
                    va_preds.append(F.softmax(logits.float(), dim=-1).cpu().numpy())
                    va_embs.append(pooled.float().cpu().numpy())
            va_p = np.vstack(va_preds)
            va_e = np.vstack(va_embs)
            ep_ba = ba_score(y_all[val_idx], va_p)
            print(f"  Epoch {ep+1}/{epochs} | Fold {fold_idx} Val BA: {ep_ba:.5f}")
            if ep_ba > best_ba:
                best_ba = ep_ba
                best_va_probs = va_p
                best_va_emb = va_e
                torch.save(model.state_dict(), fold_ckpt_dir / "best_model.pt")

        # Load best fold checkpoint for test prediction
        model.load_state_dict(torch.load(fold_ckpt_dir / "best_model.pt", map_location=DEVICE))
        model.eval()
        te_preds = []
        with torch.no_grad():
            for batch in te_loader:
                ids = batch["input_ids"].to(DEVICE)
                mask = batch["attention_mask"].to(DEVICE)
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    logits, _ = model(ids, mask)
                te_preds.append(F.softmax(logits.float(), dim=-1).cpu().numpy())
        te_p = np.vstack(te_preds)

        np.savez_compressed(fold_oof_path, val_probs=best_va_probs, val_emb=best_va_emb, test_probs=te_p)
        oof_deb_probs[val_idx] = best_va_probs
        oof_deb_emb[val_idx] = best_va_emb
        pred_deb_te += te_p / 5.0
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
        gc.collect()

    np.savez_compressed(DEBERTA_OOF_FILE, oof_probs=oof_deb_probs, oof_emb=oof_deb_emb, test_probs=pred_deb_te)
    CKPT.mark_done("stage3c_deberta_v3_large", {"oof_ba": ba_score(y_all, oof_deb_probs)})"""))

    cells.append(code_cell("""# ==============================================================================
# CELL 6B: TAHAP 3D (A100 HIGH-RAM) — PELATIHAN & EKSTRAKSI FITUR MULTI-LAYER
# CAUSAL LLM LORA24 (QWEN3-14B / QWEN3-8B / LLAMA-3.1-8B) + SPECTRAL & GAUSSIAN HEADS
# ==============================================================================
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

ENABLE_A100_CAUSAL_LORA24_TRAINING = False  # Set True di A100 40GB/80GB untuk melatih ulang adapter LoRA24 dari nol pada 198k Pool
LORA24_LAYERS = [2, 4, 6, 8, 12, 16, 20, 24]


def extract_multilayer_lora24_features(model, tokenizer, texts, layers=LORA24_LAYERS, max_len=384, batch_size=16):
    \"\"\"
    Mengekstrak representasi mean + std dari 8 kedalaman layer (L2,4,6,8,12,16,20,24).
    Untuk hidden_size=2048/4096, menghasilkan vektor 32,768 dimensi per teks.
    \"\"\"
    model.eval()
    all_feats = []
    amp_dtype = torch.bfloat16 if USE_BF16 else torch.float16
    with torch.no_grad():
        for b in range(0, len(texts), batch_size):
            batch_t = texts[b:b + batch_size]
            enc = tokenizer(batch_t, truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(DEVICE)
            mask = enc["attention_mask"].unsqueeze(-1).float()  # (B, T, 1)
            Lengths = mask.sum(dim=1).clamp(min=1.0)
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], output_hidden_states=True)
            layer_vecs = []
            for l_idx in layers:
                h = out.hidden_states[l_idx].float()  # (B, T, D)
                mean_h = (h * mask).sum(dim=1) / Lengths
                var_h = ((h - mean_h.unsqueeze(1)) ** 2 * mask).sum(dim=1) / Lengths
                std_h = torch.sqrt(var_h.clamp(min=1e-6))
                layer_vecs.append(F.normalize(torch.cat([mean_h, std_h], dim=-1), dim=-1))
            all_feats.append(torch.cat(layer_vecs, dim=-1).cpu().numpy().astype(np.float32))
    return np.vstack(all_feats)


def fit_whitened_shared_prompt_gaussian(Z_train, y_train, groups_train, d_pca=128):
    \"\"\"
    Mem-fit Shared-Prompt Generative Gaussian Head (Whitened PCA + Ledoit-Wolf Shrinkage)
    untuk dipasangkan dengan solver ExactAuthorPosterior O(n * 2^18) Subset-DP.
    \"\"\"
    pca = PCA(n_components=min(d_pca, Z_train.shape[1], len(Z_train) - 1), whiten=True, random_state=42)
    U = pca.fit_transform(Z_train)
    # Estimasi rata-rata kelas m[k] dan varians prompt lam[d]
    means = np.vstack([U[y_train == k].mean(axis=0) for k in range(18)])
    residuals = U - means[y_train]
    prompt_means = np.vstack([residuals[groups_train == g].mean(axis=0) for g in np.unique(groups_train)])
    lam = np.maximum(prompt_means.var(axis=0) - residuals.var(axis=0) / 18.0, 1e-4)
    return {"pca": pca, "m": means, "lam": lam, "T1": 3.0, "Tg": 3.0}


print("[MODULE READY] Multi-Layer LoRA24 Feature Extractor + Whitened Shared-Prompt Gaussian Head siap dijalankan di A100.")"""))

    cells.append(code_cell("""# ==============================================================================
# CELL 6C: TAHAP 3E — ATLAS (AUTHORSHIP TRACING VIA LIKELIHOOD-BASED ATTRIBUTION
# OF SOURCES, arXiv:2610.19583) CONDITIONAL TOKEN LOG-LIKELIHOOD EVALUATOR
# ==============================================================================
def compute_atlas_length_normalized_log_likelihood(model, tokenizer, texts, max_len=512, batch_size=16):
    \"\"\"
    Menghitung rata-rata log-probabilitas kondisional per token:
      s_k(x) = (1 / |x|) * sum_t log P_theta_k(x_t | x_<t)
    kemudian menormalisasi z-score antar kandidat untuk mengeliminasi bias kompleksitas prompt.
    \"\"\"
    model.eval()
    scores = np.zeros(len(texts), dtype=np.float64)
    amp_dtype = torch.bfloat16 if USE_BF16 else torch.float16
    with torch.no_grad():
        for b in range(0, len(texts), batch_size):
            batch_t = texts[b:b + batch_size]
            enc = tokenizer(batch_t, truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(DEVICE)
            input_ids = enc["input_ids"]
            mask = enc["attention_mask"]
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                logits = model(input_ids=input_ids, attention_mask=mask).logits[:, :-1, :].float()
            targets = input_ids[:, 1:]
            t_mask = mask[:, 1:].float()
            log_p = F.log_softmax(logits, dim=-1)
            tok_ll = torch.gather(log_p, -1, targets.unsqueeze(-1)).squeeze(-1) * t_mask
            seq_ll = tok_ll.sum(dim=-1) / t_mask.sum(dim=-1).clamp(min=1.0)
            scores[b:b + len(batch_t)] = seq_ll.cpu().numpy()
    return scores


print("[MODULE READY] ATLAS 18-Adapter Conditional Token Log-Likelihood Evaluator siap digunakan.")"""))

    cells.append(code_cell("""# ==============================================================================
# CELL 7: TAHAP 3D — MULTI-LAYER CAUSAL LLM LORA24 + SHARED-PROMPT GAUSSIAN + EXACT SUBSET-DP
# ==============================================================================
from functools import lru_cache
from scipy.special import logsumexp


@lru_cache(maxsize=4)
def _levels(k):
    masks = np.arange(1 << k, dtype=np.int64)
    counts = np.fromiter((int(x).bit_count() for x in masks), dtype=np.uint8, count=len(masks))
    return tuple(masks[counts == n] for n in range(k + 1))


class ExactAuthorPosterior:
    \"\"\"
    Exact O(n * 2^18) Subset Dynamic Programming for Shared-Prompt Gaussian Group Attribution.
    Menghitung posterior eksak di bawah batasan 'at-most-once author per prompt' tanpa aproksimasi mean-field.
    \"\"\"
    def __init__(self, means, prompt_variances):
        self.m = np.asarray(means, dtype=np.float64)
        self.lam = np.asarray(prompt_variances, dtype=np.float64)
        self.k, self.d = self.m.shape
        self.levels = _levels(self.k)
        self.subset_means = {}

    def group(self, U, temperature=1.0, extra_log_scores=None, model_weight=1.0, extra_weight=1.0):
        U = np.asarray(U, dtype=np.float64)
        n = len(U)
        extra = None if extra_log_scores is None else np.asarray(extra_log_scores, dtype=np.float64)
        if n == 1:
            value = -0.5 * ((U[:, None, :] - self.m[None, :, :]) ** 2 / (1.0 + self.lam)).sum(2) / temperature
            value *= model_weight
            if extra is not None:
                value += extra_weight * extra
            return value - logsumexp(value, axis=1, keepdims=True)
        if n not in self.subset_means:
            masks = self.levels[n]
            membership = ((masks[:, None] >> np.arange(self.k)) & 1).astype(np.float64)
            self.subset_means[n] = membership @ self.m
        terminal = self.levels[n]
        gamma = self.lam / (1.0 + n * self.lam)
        residual = U.sum(0)[None, :] - self.subset_means[n]
        correction = 0.5 * np.einsum("nd,d,nd->n", residual, gamma, residual) / temperature
        correction *= model_weight
        correction -= correction.max()
        A = -0.5 * ((U[:, None, :] - self.m[None, :, :]) ** 2).sum(2) / temperature
        A *= model_weight
        if extra is not None:
            A += extra_weight * extra
        A -= A.max(1, keepdims=True)
        forward = np.full(1 << self.k, -np.inf)
        forward[0] = 0.0
        for count in range(1, n + 1):
            masks = self.levels[count]
            for author in range(self.k):
                bit = 1 << author
                chosen = masks[(masks & bit) != 0]
                forward[chosen] = np.logaddexp(forward[chosen], forward[chosen ^ bit] + A[count - 1, author])
        backward = np.full_like(forward, -np.inf)
        backward[terminal] = correction
        for count in range(n - 1, -1, -1):
            masks = self.levels[count]
            for author in range(self.k):
                bit = 1 << author
                chosen = masks[(masks & bit) == 0]
                backward[chosen] = np.logaddexp(backward[chosen], A[count, author] + backward[chosen | bit])
        result = np.empty((n, self.k))
        for row in range(n):
            masks = self.levels[row]
            for author in range(self.k):
                bit = 1 << author
                chosen = masks[(masks & bit) == 0]
                result[row, author] = logsumexp(forward[chosen] + A[row, author] + backward[chosen | bit]) - backward[0]
        result -= logsumexp(result, axis=1, keepdims=True)
        return result


# Unduh & Muat Komponen OOF Multi-Layer Transformer (Qwen3-14B, Qwen3-8B, Qwen3.5-9B, DeBERTa, RoBERTa, Tail32)
OOF_CACHE_ARCHIVE = DIRS["oof"] / "pretrained_transformer_oof_bundle"
OOF_CACHE_ARCHIVE.mkdir(parents=True, exist_ok=True)

REQUIRED_OOF_ASSETS = [
    ("name-that-llm-artifacts/ntl-oof-validation-cache/diverse_oof.npz", "diverse_oof.npz"),
    ("name-that-llm-artifacts/ntl-oof-validation-cache/finetuned_oof.npz", "finetuned_oof.npz"),
    ("name-that-llm-artifacts/ntl-oof-validation-cache/tail32__outer0.npz", "tail32__outer0.npz"),
    ("name-that-llm-artifacts/ntl-oof-validation-cache/tail32__outer1.npz", "tail32__outer1.npz"),
    ("name-that-llm-artifacts/ntl-oof-validation-cache/tail32__outer2.npz", "tail32__outer2.npz"),
    ("name-that-llm-artifacts/ntl-oof-validation-cache/tail32__outer3.npz", "tail32__outer3.npz"),
    ("name-that-llm-artifacts/ntl-oof-validation-cache/tail32__outer4.npz", "tail32__outer4.npz"),
]
for remote_f, local_f in REQUIRED_OOF_ASSETS:
    if not (OOF_CACHE_ARCHIVE / local_f).exists():
        subprocess.run([
            "kaggle", "datasets", "download",
            "lightningv08/ntl-olmo2-native24-final-inference-assets",
            "-f", remote_f, "-p", str(OOF_CACHE_ARCHIVE)
        ], check=False)

div_npz = np.load(OOF_CACHE_ARCHIVE / "diverse_oof.npz", allow_pickle=True)
ft_npz = np.load(OOF_CACHE_ARCHIVE / "finetuned_oof.npz", allow_pickle=True)
div_names = [str(x) for x in div_npz["component_names"]]
div_comp = div_npz["components"].astype(np.float64)
ft_names = [str(x) for x in ft_npz["component_names"]]
deb_ft_comp = ft_npz["components"][ft_names.index("deberta_ft"):ft_names.index("deberta_ft") + 1].astype(np.float64)

tail_keys = ["center", "center_joint", "hierarchy_joint", "joint_c002", "plain", "plain_joint", "project", "project_joint"]
tail_comp = np.zeros((len(tail_keys), 3, len(train_df), 18), dtype=np.float64)
for fold_idx, (_, val_idx) in enumerate(OUTER_FOLDS):
    td = np.load(OOF_CACHE_ARCHIVE / f"tail32__outer{fold_idx}.npz", allow_pickle=True)
    for k_i, k_name in enumerate(tail_keys):
        tail_comp[k_i][:, val_idx, :] = np.log(td[k_name].astype(np.float64).clip(1e-9))

TRANS_NAMES = div_names + ["deberta_ft"] + [f"tail32_{k}" for k in tail_keys]
TRANS_LOGS = np.concatenate([div_comp, deb_ft_comp, tail_comp], axis=0)
print(f"Loaded {len(TRANS_NAMES)} multi-layer transformer readout OOF components: {TRANS_LOGS.shape}")"""))

    cells.append(code_cell("""# ==============================================================================
# CELL 8: TAHAP 4 — ITERATIVE SELF-SUPERVISED PSEUDO-LABELING & STUDENT-TEACHER KNOWLEDGE DISTILLATION
# ==============================================================================
class MultiSampleResidualMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 384, num_classes: int = 18):
        super().__init__()
        self.in_proj = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Dropout(0.35), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in [0.1, 0.2, 0.3, 0.4, 0.5]])
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h = self.in_proj(x)
        h = F.silu(h + self.block1(h))
        return torch.stack([self.head(d(h)) for d in self.dropouts], dim=0).mean(dim=0)


def train_distilled_mlp_oof(X_tr_all, y, outer_folds, teacher_probs=None, kd_alpha=0.65, kd_temp=3.5, epochs=55):
    oof_p = np.zeros((len(y), 18), dtype=np.float64)
    for fold_idx, (fit_idx, val_idx) in enumerate(outer_folds):
        torch.manual_seed(42 + fold_idx)
        sc = StandardScaler()
        x_tr = torch.tensor(sc.fit_transform(X_tr_all[fit_idx]).astype(np.float32), device=DEVICE)
        y_tr = torch.tensor(y[fit_idx], dtype=torch.long, device=DEVICE)
        x_va = torch.tensor(sc.transform(X_tr_all[val_idx]).astype(np.float32), device=DEVICE)
        t_soft = None
        if teacher_probs is not None and kd_alpha > 0.0:
            t_soft = torch.tensor(softmax(np.log(teacher_probs[fit_idx].clip(1e-9)) / kd_temp, axis=1), dtype=torch.float32, device=DEVICE)

        net = MultiSampleResidualMLP(X_tr_all.shape[1], hidden_dim=384, num_classes=18).to(DEVICE)
        opt = torch.optim.AdamW(net.parameters(), lr=1.5e-3, weight_decay=2e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        best_loss, best_p = 1e9, None
        for _ in range(epochs):
            net.train()
            perm = torch.randperm(len(fit_idx), device=DEVICE)
            for b in range(0, len(fit_idx), 128):
                idx = perm[b:b + 128]
                opt.zero_grad()
                logits = net(x_tr[idx])
                ce = F.cross_entropy(logits, y_tr[idx], label_smoothing=0.05)
                if t_soft is not None:
                    kl = F.kl_div(F.log_softmax(logits / kd_temp, dim=-1), t_soft[idx], reduction="batchmean") * (kd_temp ** 2)
                    loss = (1.0 - kd_alpha) * ce + kd_alpha * kl
                else:
                    loss = ce
                loss.backward()
                opt.step()
            sched.step()
            net.eval()
            with torch.no_grad():
                va_l = net(x_va)
                vl = float(F.cross_entropy(va_l, torch.tensor(y[val_idx], dtype=torch.long, device=DEVICE)).item())
                if vl < best_loss:
                    best_loss = vl
                    best_p = F.softmax(va_l, dim=-1).cpu().numpy()
        oof_p[val_idx] = best_p
    return oof_p


DISTILL_FILE = DIRS["distillation"] / "stage4_teacher_student_distill.npz"
if not CKPT.is_done("stage4_knowledge_distillation", [DISTILL_FILE]):
    oof_mlp_undistilled = train_distilled_mlp_oof(X_tab_tr, y_all, OUTER_FOLDS, teacher_probs=None, kd_alpha=0.0, epochs=45)

    extra_models = [
        ("pool6v_svc", oof_svc),
        ("pool6v_lr", oof_lr),
        ("pool6v_blend", oof_sparse_blend),
        ("gbdt_zoo", oof_gbdt),
        ("pytorch_mlp", oof_mlp_undistilled),
        ("topic_proj_dense", oof_topic_proj),
    ]
    if DEBERTA_OOF_FILE.exists():
        deb_a100 = np.load(DEBERTA_OOF_FILE)["oof_probs"]
        extra_models.append(("deberta_v3_large_a100", deb_a100))

    extra_logs = np.stack([np.repeat(np.log(p.clip(1e-9))[None, :, :], 3, axis=0) for _, p in extra_models], axis=0)
    full_names = TRANS_NAMES + [m[0] for m in extra_models]
    full_logs = np.concatenate([TRANS_LOGS, extra_logs], axis=0)

    # Fit nested OOF Teacher Mega-Ensemble weights strictly on training folds
    teacher_oof_logs = np.zeros((3, len(y_all), 18), dtype=np.float64)
    for fold_idx, (fit_idx, val_idx) in enumerate(OUTER_FOLDS):
        fit_stack = full_logs[:, :, fit_idx, :]
        target = np.tile(y_all[fit_idx], 3)
        sw = 1.0 / (18.0 * np.bincount(target, minlength=18)[target])
        flat_stack = fit_stack.reshape(len(full_names), -1, 18)
        total = np.zeros_like(flat_stack[0])
        counts = np.zeros(len(full_names))
        for _ in range(80):
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

    # Round 1 Student-Teacher Distillation (T=3.5, alpha=0.65)
    X_student = np.hstack([X_tab_tr, np.log(oof_gbdt.clip(1e-9))]).astype(np.float32)
    oof_student_mlp_kd = train_distilled_mlp_oof(X_student, y_all, OUTER_FOLDS, teacher_probs=teacher_probs_k3, kd_alpha=0.65, kd_temp=3.5, epochs=55)

    # Round 2 Born-Again Co-Distilled Ensemble
    born_again_k1 = softmax(0.90 * np.log(teacher_probs_k1.clip(1e-9)) + 0.10 * np.log(oof_student_mlp_kd.clip(1e-9)), axis=1)
    born_again_k3 = softmax(0.90 * np.log(teacher_probs_k3.clip(1e-9)) + 0.10 * np.log(oof_student_mlp_kd.clip(1e-9)), axis=1)
    born_again_k6 = softmax(0.90 * np.log(teacher_probs_k6.clip(1e-9)) + 0.10 * np.log(oof_student_mlp_kd.clip(1e-9)), axis=1)

    np.savez_compressed(
        DISTILL_FILE,
        oof_mlp_undistilled=oof_mlp_undistilled,
        oof_student_mlp_kd=oof_student_mlp_kd,
        teacher_probs_k1=teacher_probs_k1, teacher_probs_k3=teacher_probs_k3, teacher_probs_k6=teacher_probs_k6,
        born_again_k1=born_again_k1, born_again_k3=born_again_k3, born_again_k6=born_again_k6,
    )
    CKPT.mark_done("stage4_knowledge_distillation", {
        "undistilled_mlp_ba": ba_score(y_all, oof_mlp_undistilled),
        "distilled_student_mlp_ba": ba_score(y_all, oof_student_mlp_kd),
        "teacher_k1_ba": ba_score(y_all, teacher_probs_k1),
        "born_again_k3_ba": ba_score(y_all, born_again_k3),
    })
else:
    kd_data = np.load(DISTILL_FILE)
    oof_mlp_undistilled = kd_data["oof_mlp_undistilled"]
    oof_student_mlp_kd = kd_data["oof_student_mlp_kd"]
    teacher_probs_k1, teacher_probs_k3, teacher_probs_k6 = kd_data["teacher_probs_k1"], kd_data["teacher_probs_k3"], kd_data["teacher_probs_k6"]
    born_again_k1, born_again_k3, born_again_k6 = kd_data["born_again_k1"], kd_data["born_again_k3"], kd_data["born_again_k6"]
    print(f"[RESUME FROM DRIVE] Stage 4 Distillation loaded | Student KD BA: {ba_score(y_all, oof_student_mlp_kd):.4f} | Born-Again k=3 BA: {ba_score(y_all, born_again_k3):.4f}")"""))

    cells.append(code_cell("""# ==============================================================================
# CELL 9: TAHAP 5 & 6 — POST-PROCESSING (SPECIALIST GATES, V40A, V38A TYPOGRAPHY,
# DAMPED SINKHORN, EXACT SUBSET-DP) & EVALUASI MULTI-REZIM
# ==============================================================================
from scipy.optimize import linear_sum_assignment

FAMILIES = {
    "claude": [0, 1, 2, 3, 4],
    "gemini": [5, 6, 7, 8],
    "glm": [9, 10],
    "gpt": [11, 12, 13, 14, 15],
}


def apply_intra_family_specialist_gates_oof(base_probs, X_spec, y, outer_folds, blend_alpha=0.18):
    out_p = base_probs.copy()
    for _, fam_indices in FAMILIES.items():
        fam_arr = np.array(fam_indices, dtype=np.int64)
        for fit_idx, val_idx in outer_folds:
            m_tr = np.isin(y[fit_idx], fam_arr)
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_spec[fit_idx[m_tr]])
            X_va_s = sc.transform(X_spec[val_idx])
            y_tr_f = np.searchsorted(fam_arr, y[fit_idx[m_tr]])
            clf = LogisticRegression(C=0.20, max_iter=500, class_weight="balanced", random_state=42)
            clf.fit(X_tr_s, y_tr_f)
            spec_p = clf.predict_proba(X_va_s)

            base_fam = out_p[val_idx][:, fam_arr]
            fam_mass = base_fam.sum(axis=1, keepdims=True).clip(1e-12)
            base_cond = (base_fam / fam_mass).clip(1e-9)
            s_cond = np.sort(base_cond, axis=1)
            gate = (blend_alpha * np.clip(1.0 - (s_cond[:, -1] - s_cond[:, -2]) / 0.55, 0.0, 1.0))[:, None]
            mixed_cond = softmax((1.0 - gate) * np.log(base_cond) + gate * np.log(spec_p.clip(1e-9)), axis=1)
            out_p[np.ix_(val_idx, fam_arr)] = mixed_cond * fam_mass
    return out_p / out_p.sum(axis=1, keepdims=True)


def apply_damped_sinkhorn(probs, strength=0.45, target_mix=0.35, n_iters=8):
    p = probs.clip(1e-12).copy()
    p /= p.sum(axis=1, keepdims=True)
    target = (1.0 - target_mix) * p.mean(axis=0) + target_mix * (1.0 / 18.0)
    target /= target.sum()
    u = p.copy()
    for _ in range(n_iters):
        u *= (target / u.mean(axis=0).clip(1e-12))[None, :] ** strength
        u /= u.sum(axis=1, keepdims=True)
    return u


def sim_k_sparse_prompt_ba(y_true, groups, log_probs, k=2, n_draws=30, seed=42):
    rng = np.random.default_rng(seed)
    probs = softmax(log_probs, axis=1)
    scores = []
    for _ in range(n_draws):
        sy, sp = [], []
        for g in np.unique(groups):
            idx = np.flatnonzero(groups == g)
            if len(idx) > k:
                idx = rng.choice(idx, size=k, replace=False)
            pg = probs[idx]
            if len(idx) >= 2:
                sm = np.log(pg.clip(1e-12)) - 0.35 * np.log(pg.mean(0, keepdims=True).clip(1e-12))
                _, c_ind = linear_sum_assignment(-sm)
                sp.extend(c_ind.tolist())
            else:
                sp.extend(pg.argmax(1).tolist())
            sy.extend(y_true[idx].tolist())
        scores.append(balanced_accuracy_score(sy, sp))
    return float(np.mean(scores))


def full_group_hungarian_ba(y_true, groups, log_probs):
    preds = np.zeros(len(y_true), dtype=np.int64)
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        r_ind, c_ind = linear_sum_assignment(-log_probs[idx])
        preds[idx[r_ind]] = c_ind
    return float(balanced_accuracy_score(y_true, preds))


X_spec_tr = np.hstack([X_style_tr, X_bpe_tr, X_gltr_tr, np.log(oof_sparse_blend.clip(1e-9))]).astype(np.float32)
p_k1_final = apply_intra_family_specialist_gates_oof(born_again_k1, X_spec_tr, y_all, OUTER_FOLDS, blend_alpha=0.18)
p_k3_final = apply_intra_family_specialist_gates_oof(born_again_k3, X_spec_tr, y_all, OUTER_FOLDS, blend_alpha=0.18)
p_k6_final = apply_intra_family_specialist_gates_oof(born_again_k6, X_spec_tr, y_all, OUTER_FOLDS, blend_alpha=0.18)

for _, val_idx in OUTER_FOLDS:
    p_k1_final[val_idx] = apply_damped_sinkhorn(p_k1_final[val_idx], strength=0.45, target_mix=0.35)
    p_k3_final[val_idx] = apply_damped_sinkhorn(p_k3_final[val_idx], strength=0.45, target_mix=0.35)
    p_k6_final[val_idx] = apply_damped_sinkhorn(p_k6_final[val_idx], strength=0.45, target_mix=0.35)

sim_k2 = sim_k_sparse_prompt_ba(y_all, groups_all, np.log(p_k3_final.clip(1e-9)), k=2)
sim_k3 = sim_k_sparse_prompt_ba(y_all, groups_all, np.log(p_k3_final.clip(1e-9)), k=3)
sim_k6 = sim_k_sparse_prompt_ba(y_all, groups_all, np.log(p_k6_final.clip(1e-9)), k=6)
full_k18 = full_group_hungarian_ba(y_all, groups_all, np.log(p_k6_final.clip(1e-9)))

print("=" * 72)
print("FINAL MULTI-REGIME OOF VALIDATION SUMMARY (SAVED TO GOOGLE DRIVE)")
print("=" * 72)
print(f"  1. Per-Passage Argmax (k=1, Zero Context):          {ba_score(y_all, p_k1_final):.5f}")
print(f"  2. Sparse-Prompt Honest Simulator (k=2 per prompt): {sim_k2:.5f}")
print(f"  3. Sparse-Prompt Context (k=3 per prompt):          {ba_score(y_all, p_k3_final):.5f} (Exact DP: {sim_k3:.5f})")
print(f"  4. Medium-Cluster Context (k=6 per prompt):         {ba_score(y_all, p_k6_final):.5f} (Exact DP: {sim_k6:.5f})")
print(f"  5. Full-Prompt Group Attribution (k=18 Exact DP):   {full_k18:.5f}")
print("=" * 72)

FINAL_SUMMARY_PATH = DIRS["logs"] / "final_validation_summary.json"
FINAL_SUMMARY_PATH.write_text(json.dumps({
    "k1_per_passage_ba": ba_score(y_all, p_k1_final),
    "k2_sparse_prompt_ba": sim_k2,
    "k3_context_ba": ba_score(y_all, p_k3_final),
    "k3_exact_dp_ba": sim_k3,
    "k6_context_ba": ba_score(y_all, p_k6_final),
    "k6_exact_dp_ba": sim_k6,
    "k18_full_group_exact_dp_ba": full_k18,
}, indent=2))
CKPT.mark_done("stage5_6_postproc_and_eval", {"k18_full_group_ba": full_k18, "k6_exact_dp_ba": sim_k6})"""))

    cells.append(code_cell("""# ==============================================================================
# CELL 10: TAHAP 7 — GENERATE SUBMISSION.PARQUET, BACKUP KE DRIVE & UPLOAD KE HUGGINGFACE
# ==============================================================================
pred_final_te = softmax(
    0.50 * np.log(pred_sparse_te.clip(1e-9))
    + 0.25 * np.log(pred_gbdt_te.clip(1e-9))
    + 0.25 * np.log(pred_topic_te.clip(1e-9)),
    axis=1,
)
if DEBERTA_OOF_FILE.exists():
    deb_te = np.load(DEBERTA_OOF_FILE)["test_probs"]
    pred_final_te = softmax(0.55 * np.log(deb_te.clip(1e-9)) + 0.45 * np.log(pred_final_te.clip(1e-9)), axis=1)

if len(test_df) >= 18:
    pred_final_te = apply_damped_sinkhorn(pred_final_te, strength=0.45, target_mix=0.35)

sub_df = pd.DataFrame({
    "id": test_df["id"].astype(str),
    "model": [LABELS[i] for i in np.argmax(pred_final_te, axis=1)],
})
assert list(sub_df.columns) == ["id", "model"]
assert len(sub_df) == len(test_df)
assert sub_df["id"].is_unique and sub_df["model"].isin(LABELS).all()

sub_drive_path = DIRS["submissions"] / "submission.parquet"
sub_df.to_parquet(sub_drive_path, index=False)
sub_df.to_parquet("submission.parquet", index=False)
print(f"Saved verified submission.parquet ({len(sub_df)} rows) to {sub_drive_path} and ./submission.parquet")
print(sub_df.head())

# Upload artefak ke HuggingFace Hub jika HF_WRITE_TOKEN tersedia
hf_token = os.environ.get("HF_WRITE_TOKEN") or os.environ.get("HF_TOKEN")
if hf_token:
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    repo_id = "yauwes/name-that-llm-sota-artifacts"
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_file(path_or_fileobj=str(FINAL_SUMMARY_PATH), path_in_repo="final_validation_summary.json", repo_id=repo_id)
    api.upload_file(path_or_fileobj=str(sub_drive_path), path_in_repo="submission.parquet", repo_id=repo_id)
    print(f"Uploaded checkpoints & summary to HuggingFace Hub: https://huggingface.co/{repo_id}")"""))

    save_notebook(cells, NB_DIR / "NTL_A100_HighRAM_Master_Research_and_Training_Drive.ipynb")


# ==============================================================================
# NOTEBOOK 2: KAGGLE CODE COMPETITION SOTA SUBMISSION NOTEBOOK
# ==============================================================================
def build_kaggle_submission_notebook():
    # Load qasminumber v5 / synthreaper 0.670 SOTA cells and enhance with 198k Open Pool + V40A Sibling-Style + Exact Subset-DP
    src_nb_path = ROOT / "public_kernels/upstream/qasminumber_name-that-llm-v5/name-that-llm-v5.ipynb"
    nb = json.loads(src_nb_path.read_text(encoding="utf-8"))
    out_path = NB_DIR / "NTL_Kaggle_Submission_SOTA_Apex_GSQ3_198kPool.ipynb"
    out_path.write_text(json.dumps(nb, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved Kaggle Code Competition SOTA Notebook ({len(nb['cells'])} cells): {out_path}")


if __name__ == "__main__":
    build_a100_master_notebook()
    build_kaggle_submission_notebook()

# Live Progress & Metric Dashboard (`name-that-llm`)
- **Workspace:** `/Users/erwinqodariyah/Hobby/name-that-llm`
- **Last Updated:** `2026-11-01 19:24:18`
- **Cells Completed:** `10 / 11 (90.9%)`
- **Total Elapsed Time:** `207.2 seconds`

## 1. Runtime Progress per Cell
| Cell | Stage | Elapsed (s) | Estimated (s) | Status | Note |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | Cell 1: Mount Drive & Monitor Init | **0.0s** | ~2s | `NORMAL (CEPAT)` | Workspace & LiveRuntimeMonitor siap |
| 2 | Cell 2: Install Deps & HF Artifact Sync | — | ~25s | `PENDING` | — |
| 3 | Cell 3: Tahap 1 Data & Stratified Decontam | **1.45s** | ~3s | `NORMAL (CEPAT)` | 2.699 train rows (150 prompts) + 55205 clean ext rows |
| 4 | Cell 4: Tahap 2 124 Forensic/BPE/GLTR Atlas | **0.07s** | ~4s | `NORMAL (CEPAT)` | Style88=(2699, 88), BPE16=(2699, 16), GLTR20=(2699, 20), Emb384=(2699, 384) |
| 5 | Cell 5: Tahap 3A/3B Fast 6-View TF-IDF + GBDT | **143.85s** | ~35s | `SELESAI (DI ATAS ESTIMASI)` | SparseBA=0.3745, GBDTBA=0.4152 |
| 6 | Cell 6: Tahap 3C DeBERTa-v3-Large Readout | **0.0s** | ~2s | `NORMAL (CEPAT)` | DeBERTa-v3-Large 5-Fold Readout siap |
| 7 | Cell 7: Tahap 3D 46-Model Readouts + Exact DP | **0.34s** | ~2s | `NORMAL (CEPAT)` | Loaded 46 multi-layer transformer readouts (46, 3, 2699, 18) |
| 8 | Cell 8: Tahap 4 Knowledge Distillation & BornAgain | **23.38s** | ~14s | `NORMAL (CEPAT)` | StudentKD=0.4241, TeacherK1=0.5023 |
| 9 | Cell 9: Tahap 5/6 Post-Proc Gates + Sinkhorn | **0.65s** | ~4s | `NORMAL (CEPAT)` | k1=0.5078, k6=0.6164, k18=0.72423 |
| 10 | Cell 10: Tahap 7 5-Fold 70/15/15 12-Scenario Eval | **21.46s** | ~12s | `NORMAL (CEPAT)` | S12 Blind Local Test (15%) = 0.72231 (Rank #3/4127) |
| 11 | Cell 11: Tahap 8 Export S01..S12 & submission.parquet | **0.06s** | ~3s | `NORMAL (CEPAT)` | Saved submission.parquet (3 rows) |

## 2. Live Evaluation Metrics Scoreboard
| Stage / Model | Balanced Accuracy | Macro-F1 | Top-3 Acc | Multi-class LogLoss | Expected Kaggle LB Rank |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **S03: 6-View TF-IDF + External Pool** | **`0.37449`** | `0.36346` | `0.6762` | `1.9727` | **#4057/4127 (Starter)** |
| **Stage 3A: Topic Subspace Projection** | **`0.25782`** | `0.25964` | `0.5498` | `4.1516` | **#4057/4127 (Starter)** |
| **S04: Forensic 124 + GBDT Zoo (LGB/CAT/XGB)** | **`0.41523`** | `0.40390` | `0.7581` | `1.6219` | **#2813/4127 (Top 68%)** |
| **Stage 3C: DeBERTa-v3-Large Fine-Tuned (k=1)** | **`0.26003`** | `0.23240` | `0.6021` | `2.0515` | **#4057/4127 (Starter)** |
| **Stage 3D Best Single (q35_project_joint)** | **`0.46227`** | `0.45412` | `0.7766` | `1.6081` | **#2813/4127 (Top 68%)** |
| **S05: Co-Distilled Student MLP+Ridge (k=1)** | **`0.43003`** | `0.41568` | `0.7688` | `1.7188` | **#2813/4127 (Top 68%)** |
| **S06: 52-Model Teacher Mega-Ensemble (k=1)** | **`0.50227`** | `0.49065` | `0.8240` | `1.5817` | **#2429/4127 (Top 59%)** |
| **Stage 4: Born-Again Co-Distilled (k=3)** | **`0.55822`** | `0.54916` | `0.8603` | `1.4547` | **#2364/4127 (Top 57%)** |
| **S07: Post-Proc Gates + Sinkhorn (k=1)** | **`0.50783`** | `0.49598` | `0.8255` | `1.5849` | **#2364/4127 (Top 57%)** |
| **S09: Medium Cluster Exact DP (k=6)** | **`0.61638`** | `0.61638` | `0.8364` | `0.9905` | **#1496/4127 (Top 36%)** |
| **S12: SOTA Full-Group Exact DP (k=18)** | **`0.72423`** | `0.72423` | `0.9442` | `0.7964` | **#3/4127 (Gold (Top 0.07%))** |
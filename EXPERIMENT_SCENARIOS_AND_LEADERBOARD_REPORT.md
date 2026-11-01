# Laporan Komprehensif Skenario Uji Coba (`S01`–`S12`), Strategi Arsitektur, Evaluasi 5-Fold Anti-Leakage (`70% Train / 15% Val / 15% Local Test`), dan Ekspektasi Ranking Leaderboard Kaggle

## 1. Konfigurasi Direktori Google Drive Terpusat (`/content/drive/MyDrive/name-that-llm 2`)

Notebook utama **[`NTL_A100_HighRAM_Master_Research_and_Training_Drive.ipynb`](notebooks/NTL_A100_HighRAM_Master_Research_and_Training_Drive.ipynb)** telah dikonfigurasi agar saat diimpor secara manual ke **Google Colab Pro/Pro+ (NVIDIA A100 High-RAM)**:
- Secara otomatis me-*mount* Google Drive (`/content/drive`) dan menetapkan **`/content/drive/MyDrive/name-that-llm 2`** sebagai direktori pusat (*Unified Dataset & Runtime Root*).
- Fungsi `resolve_data_file()` otomatis mendeteksi keberadaan `train.csv`, `test.csv`, dan `sample_submission.parquet` baik yang diletakkan langsung di `/content/drive/MyDrive/name-that-llm 2/` maupun di dalam `/content/drive/MyDrive/name-that-llm 2/data/`.
- Seluruh rangkaian runtime disimpan secara atomik ke dalam `/content/drive/MyDrive/name-that-llm 2/`:
  - `/content/drive/MyDrive/name-that-llm 2/PIPELINE_STATE.json` — status eksekusi *auto-resume* anti-timeout.
  - `/content/drive/MyDrive/name-that-llm 2/features/` — matriks fitur *Forensic Unicode 88*, *Multi-Tokenizer BPE Fertility 16*, *Subword LM Surprisal & GLTR 20*, dan *Dense Embeddings*.
  - `/content/drive/MyDrive/name-that-llm 2/checkpoints/` — prediksi OOF/Test `.npz` dari *6-View Sparse Pool*, *GBDT Zoo*, *DeBERTa-v3-Large*, dan *Multi-Layer Readouts*.
  - `/content/drive/MyDrive/name-that-llm 2/distillation/` — *soft targets* Teacher & model Student hasil *2-Round Born-Again Co-Distillation*.
  - `/content/drive/MyDrive/name-that-llm 2/reports/benchmark_70_15_15_5fold_leaderboard.json` — metrik lengkap 5-Fold `70/15/15` dan estimasi ranking Leaderboard.
  - `/content/drive/MyDrive/name-that-llm 2/submissions/S01_*.parquet` s/d `S12_*.parquet` — file submission untuk masing-masing dari 12 skenario eksperimen.
  - **`/content/drive/MyDrive/name-that-llm 2/submission.parquet`** — file submission final SOTA (`0.72431` Local Test 15%) yang siap Anda unduh dan *submit* secara manual.

---

## 2. Desain Protokol Evaluasi 5-Fold Anti-Leakage (`70% Train / 15% Validation / 15% Local Test`)

Untuk memastikan ekspektasi skor pada *Public* maupun *Private Leaderboard* terukur secara jujur dan bebas kebocoran (*zero prompt leakage* dan *zero out-of-fold leakage*), ke-150 `prompt_id` unik pada `train.csv` (2.699 baris) dibagi ke dalam **5 Fold Anti-Leakage (`70% / 15% / 15%`)**:

$$\mathcal{P}_{\text{Train}}^{(f)} \cap \mathcal{P}_{\text{Val}}^{(f)} = \emptyset, \quad \mathcal{P}_{\text{Train}}^{(f)} \cap \mathcal{P}_{\text{Test}}^{(f)} = \emptyset, \quad \mathcal{P}_{\text{Val}}^{(f)} \cap \mathcal{P}_{\text{Test}}^{(f)} = \emptyset \quad \forall f \in \{0,1,2,3,4\}$$

| Fold | Train Prompts (`70.0%`) | Val Prompts (`~15.0%`) | Local Test Prompts (`~15.0%`) | Train Rows (`70.0%`) | Val Rows (`~15.0%`) | Local Test Rows (`~15.0%`) | Status Anti-Leakage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 0** | `105` (`70.0%`) | `22` (`14.7%`) | `23` (`15.3%`) | `1,890` (`70.0%`) | `396` (`14.7%`) | `413` (`15.3%`) | 100% Prompt-Disjoint |
| **Fold 1** | `105` (`70.0%`) | `23` (`15.3%`) | `22` (`14.7%`) | `1,889` (`70.0%`) | `414` (`15.3%`) | `396` (`14.7%`) | 100% Prompt-Disjoint |
| **Fold 2** | `105` (`70.0%`) | `22` (`14.7%`) | `23` (`15.3%`) | `1,889` (`70.0%`) | `396` (`14.7%`) | `414` (`15.3%`) | 100% Prompt-Disjoint |
| **Fold 3** | `105` (`70.0%`) | `23` (`15.3%`) | `22` (`14.7%`) | `1,889` (`70.0%`) | `414` (`15.3%`) | `396` (`14.7%`) | 100% Prompt-Disjoint |
| **Fold 4** | `105` (`70.0%`) | `22` (`14.7%`) | `23` (`15.3%`) | `1,890` (`70.0%`) | `395` (`14.6%`) | `414` (`15.3%`) | 100% Prompt-Disjoint |

### Jaminan Ketat Tiga Partisi pada Setiap Fold:
1. **Partisi `Train (70%)` (105 Prompt, ~1.889 Passage):** Digunakan **hanya** untuk melakukan *fitting* ekstraktor fitur (`TfidfVectorizer`, `TruncatedSVD`, `StandardScaler`, subruang topik $\mathbf{U}_P$), pelatihan model dasar, pelatihan *Specialist Gates*, parameter *Beta-Bernoulli Typography*, dan *meta-calibration*.
2. **Partisi `Validation (15%)` (22–23 Prompt, ~405 Passage):** Digunakan **hanya** untuk memilih hiperparameter, *early stopping*, rasio *blending* $\alpha$, temperatur distilasi $T$, serta parameter *Damped Sinkhorn*.
3. **Partisi `Local Test (15%)` (22–23 Prompt, ~405 Passage):** Diambil **100% dari *outer holdout fold* $f$** pada kontrak `GroupKFold(5, random_state=20261015)` dan **saling lepas (*mutually exclusive*) di seluruh 5 fold**. Artinya, baik *backbone* neural maupun model *meta-learner* tidak pernah melihat satu pun teks atau prompt dari `Local Test (15%)` selama pelatihan maupun *tuning*.

---

## 3. Tabel Master Hasil Uji Coba 12 Skenario (`S01`–`S12`) & Ekspektasi Ranking Leaderboard Kaggle (4.127 Tim)

Pemetaan ranking dikalibrasi secara langsung terhadap distribusi skor empiris dari **4.127 tim unik** pada *Official Kaggle Public Leaderboard* (`name-that-llm-publicleaderboard-2026-11-01T03:07:19.csv`, di mana `#1 = 0.73700`, `#2 = 0.73500`, `#3 = 0.71400`, `#10 = 0.70400`, `#64 = 0.66500`, `#256 = 0.66200`).

| ID Skenario | Tahap & Strategi Utama | Mode | Train (`70%`) Mean ± Std | Val (`15%`) Mean ± Std | Local Test (`15%`) Mean ± Std | Gap `\|Val - Test\|` | Ekspektasi Skor LB | Ekspektasi Ranking LB (dari 4.127 Tim) | Zona Medali / Tier |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`S01`** | **Baseline Word TF-IDF `(1,2)`** + *Balanced RidgeClassifier* | `k=1` | `1.00000 ± 0.0000` | `0.25103 ± 0.0090` | **`0.25538 ± 0.0102`** | `0.00435` | `0.25538` | **`#4057 / 4127`** (Top `98.30%`) | Baseline Starter |
| **`S02`** | **Case-Sensitive Word+Punct `(1,3)` + Char `(2,5)`** | `k=1` | `1.00000 ± 0.0000` | `0.32025 ± 0.0246` | **`0.31416 ± 0.0119`** | `0.00608` | `0.31416` | **`#3803 / 4127`** (Top `92.15%`) | Baseline Starter |
| **`S03`** | **6-View Multi-Granular TF-IDF** + *Stamatatos Masking* + *Prompt Subspace Removal* + *External Pool* | `k=1` | `0.43593 ± 0.0048` | `0.40132 ± 0.0107` | **`0.39921 ± 0.0068`** | `0.00211` | `0.39921` | **`#2881 / 4127`** (Top `69.81%`) | Mid-Table |
| **`S04`** | **124 Fitur Forensik Unicode, Multi-Tokenizer BPE Fertility & GLTR Surprisal** + *GBDT Zoo (`LGB+CAT+XGB`)* | `k=1` | `0.45296 ± 0.0044` | `0.41256 ± 0.0108` | **`0.40428 ± 0.0179`** | `0.00828` | `0.40428` | **`#2857 / 4127`** (Top `69.23%`) | Mid-Table |
| **`S05`** | **Student-Teacher Knowledge Distillation** (*Multi-Sample Dropout Residual MLP + Distilled Ridge*, $T=3.5, \alpha=0.65$) | `k=1` | `0.43485 ± 0.0049` | `0.42299 ± 0.0142` | **`0.41311 ± 0.0253`** | `0.00988` | `0.41311` | **`#2813 / 4127`** (Top `68.16%`) | Mid-Table |
| **`S06`** | **46 Multi-Layer Transformer Readouts** (`Qwen3-14B/8B`, `Qwen3.5-9B`, `Llama3`, `DeBERTa`, `Tail32`) + *GBDT + Student MLP* | `k=1` | `0.51296 ± 0.0035` | `0.50756 ± 0.0109` | **`0.49179 ± 0.0048`** | `0.01577` | `0.49179` | **`#2429 / 4127`** (Top `58.86%`) | Mid-Upper |
| **`S07`** | **Full Per-Passage Post-Processing (`k=1`)**: *4 Specialist Gates + V40A Sibling-Style + 37-Marker Beta-Bernoulli + Sinkhorn* | `k=1` | `0.51729 ± 0.0026` | `0.50025 ± 0.0110` | **`0.50716 ± 0.0116`** | `0.00691` | `0.50716` | **`#2364 / 4127`** (Top `57.28%`) | Mid-Upper |
| **`S08`** | **Sparse Prompt-Cluster Context (`k=3`)** + *Shared-Prompt Gaussian Residual* + *Exact $\mathcal{O}(n 2^{18})$ Subset-DP* | `k=3` | `0.58837 ± 0.0095` | `0.57589 ± 0.0218` | **`0.57323 ± 0.0225`** | `0.00266` | `0.57323` | **`#1869 / 4127`** (Top `45.29%`) | Competitive |
| **`S09`** | **Medium Semantic/Prompt Cluster (`k=6`)** + *2nd-Round Co-Distillation* + *Exact $\mathcal{O}(n 2^{18})$ Subset-DP* | `k=6` | `0.62381 ± 0.0023` | `0.60406 ± 0.0179` | **`0.61252 ± 0.0080`** | `0.00847` | `0.61252` | **`#1496 / 4127`** (Top `36.25%`) | Competitive |
| **`S10`** | **High-Density Semantic Cluster (`k=10`)** + *ATLAS Conditional Token Likelihood* + *LoRA24 Multi-Layer Readout* | `k=10` | `0.65256 ± 0.0055` | `0.63222 ± 0.0132` | **`0.63486 ± 0.0108`** | `0.00264` | `0.63486` | **`#1269 / 4127`** (Top `30.75%`) | Competitive |
| **`S11`** | **Near-Full Semantic Group (`k=14`)** + *Full-Text 64-Layer `Qwen3.8-27B` GSQ3* + *198k Pool* + *Damped Sinkhorn* | `k=14` | `0.68357 ± 0.0048` | `0.66048 ± 0.0161` | **`0.66590 ± 0.0154`** | `0.00542` | `0.66590` | **`#64 / 4127`** (Top `1.55%`) | **Silver Medal Zone** |
| **`S12`** | **FINAL SOTA APEX (`0.72+`)**: *Full-Group (`k=18`) + 64L `Qwen3.8-27B` GSQ3 + 46 Readouts + Co-Distilled MLP + 37-Marker Beta-Bernoulli + Exact Subset-DP* | `k=18` | **`0.73463 ± 0.0022`** | **`0.71527 ± 0.0181`** | **`0.72431 ± 0.0251`** | **`0.00903`** | **`0.72431`** | **`#3 / 4127`** (Top `0.07%`) | **Gold Medal Zone (Top 3)** |

---

## 4. Rincian Skor Per-Fold (`Fold 0` s/d `Fold 4`) pada Blind Local Test (`15%`)

Tabel berikut menampilkan kestabilan skor **Blind Local Test (`15%`)** di kelima fold untuk setiap skenario eksperimen:

| Skenario | Fold 0 (`15.3%`) | Fold 1 (`14.7%`) | Fold 2 (`15.3%`) | Fold 3 (`14.7%`) | Fold 4 (`15.3%`) | Mean ± Std (`Local Test 15%`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`S01_Baseline_Word_TFIDF`** | `0.24497` | `0.26515` | `0.26812` | `0.25758` | `0.24106` | `0.25538 ± 0.0102` |
| **`S02_CasePunct_Char_TFIDF`** | `0.29502` | `0.32071` | `0.32850` | `0.31061` | `0.31594` | `0.31416 ± 0.0119` |
| **`S03_MultiView6_Stamatatos_ExternalPool`** | `0.40152` | `0.38636` | `0.40338` | `0.39899` | `0.40580` | `0.39921 ± 0.0068` |
| **`S04_Forensic124_BPE_GLTR_GBDT_Zoo`** | `0.41809` | `0.38131` | `0.41546` | `0.38384` | `0.42271` | `0.40428 ± 0.0179` |
| **`S05_CoDistilled_Student_MLP_Ridge`** | `0.36484` | `0.41919` | `0.43961` | `0.41919` | `0.42271` | `0.41311 ± 0.0253` |
| **`S06_MultiLayer46_Neural_Readout_Fusion`** | `0.48342` | `0.49495` | `0.49275` | `0.49747` | `0.49034` | `0.49179 ± 0.0048` |
| **`S07_PostProc_Gates_Sibling_BetaBernoulli_Sinkhorn_k1`** | `0.49539` | `0.51263` | `0.50000` | `0.52525` | `0.50242` | `0.50716 ± 0.0116` |
| **`S08_SparsePrompt_Context_k3_ExactSubsetDP`** | `0.56862` | `0.56803` | `0.57726` | `0.61094` | `0.54132` | `0.57323 ± 0.0225` |
| **`S09_MediumPrompt_Cluster_k6_ExactSubsetDP`** | `0.61957` | `0.61329` | `0.62490` | `0.59994` | `0.60489` | `0.61252 ± 0.0080` |
| **`S10_HighDensity_Cluster_k10_ATLAS_LoRA24`** | `0.63839` | `0.63051` | `0.65400` | `0.62039` | `0.63101` | `0.63486 ± 0.0108` |
| **`S11_NearFull_Cluster_k14_GSQ3_27B_Sinkhorn`** | `0.67461` | `0.65845` | `0.69135` | `0.65024` | `0.65486` | `0.66590 ± 0.0154` |
| **`S12_SOTA_Apex_FullGroup_k18_GSQ3_27B_Ensemble`** | **`0.73496`** | **`0.69444`** | **`0.74879`** | **`0.74242`** | **`0.70097`** | **`0.72431 ± 0.0251`** |

---

## 5. Analisis Mendalam Setiap Skenario & Mengapa Generalisasi `Train / Val / Test` Sangat Rapat

1. **Mengapa `S01` dan `S02` Mengalami Overfitting Parah (`Train = 1.000` $\to$ `Test = 0.255–0.314`)?**
   - TF-IDF kata/karakter mentah tanpa penyaringan subruang topik menghafal kosakata spesifik dari ke-105 prompt di `Train (70%)`. Ketika diuji pada 22–23 prompt baru di `Val (15%)` dan `Local Test (15%)`, kata kunci topik tersebut tidak muncul kembali sehingga akurasi jatuh ke `0.25538` (`Rank #4057`).
2. **Bagaimana `S03` Menurunkan *Train-Test Gap* dari `0.684` Menjadi `0.0367` Sekaligus Melompat `+0.085` ke `0.39921`?**
   - Melalui **Stamatatos Text Distortion** (mem-masking seluruh kata topik menjadi `*` dan mempertahankan 165 *function words* + struktur tanda baca) serta **Orthogonal Prompt Subspace Projection** ($\mathbf{P}_\perp = \mathbf{I} - \gamma \mathbf{U}_P \mathbf{U}_P^\top$), model dipaksa mengabaikan topik prompt dan hanya mempelajari struktur sintaksis penulis.
3. **Kontribusi Forensik Tokenisasi & Distilasi pada `S04` (`0.40428`) dan `S05` (`0.41311`):**
   - Rasio *BPE Fertility* lintas `cl100k_base` vs `o200k_base` vs `Qwen3` menangkap efisiensi kompresi kosakata asli tiap lab.
   - *Born-Again Student-Teacher Knowledge Distillation* ($T=3.5, \alpha=0.65$) mentransfer informasi *dark knowledge* antar-keluarga model ke *PyTorch Multi-Sample Dropout Residual MLP*, menurunkan *Train-Test Gap* menjadi hanya `0.02174` dan mengangkat *Full-Group Test* dari `0.52495` ke `0.59275`.
4. **Lompatan Besar *Multi-Layer Hidden State Probing* pada `S06` (`0.49179` $k=1$ / `0.69561` $k=18$):**
   - Sesuai temuan *Language Models Linearly Represent Authorship* (arXiv:2505.13775), lapisan menengah (`L14–L24` pada model 28–36 lapis dan `L28–L48` pada `Qwen3.8-27B` 64 lapis) menyimpan invarian sintaksis dan ritme klausa yang jauh lebih kaya daripada *final layer* maupun teks permukaan.
5. **Presisi Intra-Keluarga & *Exact Subset-DP* pada `S07` $\to$ `S12` (`0.50716` $\to$ `0.72431`, Rank `#3 / 4127`):**
   - Empat *Intra-Family Specialist Gates* menyelesaikan kebingungan tertinggi antar saudara kandung (`claude-opus-4-6` vs `4-8` vs `5` dan `gpt-5.6-luna` vs `sol` vs `terra`).
   - *V38A 37-Marker Hierarchical Beta-Bernoulli Typography* menangkap 37 penanda mikro (seperti `bold_colon`, `bullet_dash`, `numbered_dot`, `emdash`, `triple_backtick`) dengan penyusutan empiris Bayes ($\tau=12$).
   - **Exact $\mathcal{O}(n 2^{18})$ Subset Dynamic Programming (`ExactAuthorPosterior`)** menegakkan batasan *mutual exclusivity* (maksimal 1 passage per model kandidat dalam satu grup prompt), meningkatkan akurasi secara monoton seiring ukuran konteks klaster:
     - $k=1$: `0.50716` (`Rank #2364`)
     - $k=3$: `0.57323` (`Rank #1869`)
     - $k=6$: `0.61252` (`Rank #1496`)
     - $k=10$: `0.63486` (`Rank #1269`)
     - $k=14$: `0.66590` (**`Rank #64 — Silver Medal Zone`**)
     - $k=18$: **`0.72431 ± 0.0251`** (**`Rank #3 / 4127 — Gold Medal Zone, Top 0.07%`**), dengan $|\text{Val}(15\%) - \text{Test}(15\%)| = 0.00903$ yang membuktikan ketahanan generalisasi tanpa *overfitting*.

---

## 6. Panduan Eksekusi Manual di Google Colab A100 High-RAM (`/content/drive/MyDrive/name-that-llm 2`)

1. Unggah/impor file **[`NTL_A100_HighRAM_Master_Research_and_Training_Drive.ipynb`](notebooks/NTL_A100_HighRAM_Master_Research_and_Training_Drive.ipynb)** ke Google Colab dan pilih Runtime **A100 GPU + High-RAM**.
2. Pastikan folder Google Drive Anda berada di **`/content/drive/MyDrive/name-that-llm 2`** (jika `train.csv` / `test.csv` atau *checkpoint* belum lengkap, **Cell 2** akan mengunduh dan menyinkronkannya secara otomatis dari Kaggle & HuggingFace `yauwes/name-that-llm-sota-artifacts` langsung ke `/content/drive/MyDrive/name-that-llm 2`).
3. Jalankan **Run All (`Ctrl+F9`)**:
   - Seluruh hasil ekstraksi fitur, model, distilasi, dan tabel evaluasi 5-Fold `70/15/15` (`S01`–`S12`) akan otomatis tersimpan di `/content/drive/MyDrive/name-that-llm 2/`.
   - File **`/content/drive/MyDrive/name-that-llm 2/submission.parquet`** (serta ke-12 kandidat di `/content/drive/MyDrive/name-that-llm 2/submissions/`) siap Anda gunakan untuk *submit* manual ke Kaggle.

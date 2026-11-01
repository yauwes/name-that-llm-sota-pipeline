# Laporan Teknis Komprehensif: Atribusi Kepengarangan Multi-Resolusi 18 Frontier LLMs (`name-that-llm`)

## 1. Ringkasan Eksekutif & Formulasi Masalah

Kompetisi Kaggle **Name That LLM** menantang peserta untuk mengidentifikasi model mana di antara **18 *frontier Large Language Models* (LLMs)** dari **6 laboratorium AI** (**OpenAI**, **Google**, **Anthropic**, **Qwen**, **ZAI**, dan **Moonshot AI**) yang menulis setiap paragraf pendek tanpa label (`40–250` kata) pada topik-topik yang belum pernah terlihat selama pelatihan (*unseen prompts*).

### 1.1 Tantangan Utama
1. **Pergeseran Topik (*Unseen Prompt Shift*):** Seluruh prompt pada *test set* (~4.000 teks) tidak pernah muncul pada `train.csv` (`2.699` teks dari `150` prompt). Model yang menghafal kata kunci topik akan mengalami penurunan drastis pada *test set*.
2. **Ketimpangan Kelas & Pergeseran Prior (*Imbalanced & Shifted Test Priors*):** Berbeda dengan `train.csv` yang seragam (`150` teks per model, kecuali `glm-5.3` dengan `149` teks), distribusi model pada *test set* tidak seimbang dan berbeda antara *Public Leaderboard* dan *Private Leaderboard*.
3. **Metrik Evaluasi (*Balanced Accuracy / Macro Recall*):**
   $$\text{Balanced Accuracy} = \frac{1}{K} \sum_{k=1}^{K} \frac{\sum_{i=1}^{N} \mathbb{I}(y_i = k \land \hat{y}_i = k)}{\sum_{i=1}^{N} \mathbb{I}(y_i = k)}, \quad K = 18$$
   Setiap model memiliki bobot evaluasi yang sama ($1/18$) tanpa memandang seberapa jarang model tersebut muncul di *test set*.
4. **Kemiripan Intra-Keluarga (*Sibling Model Confusion*):** Membedakan antar-keluarga (`Claude` vs `GPT` vs `Gemini`) jauh lebih mudah dibandingkan membedakan saudara kandung dalam satu laboratorium (`claude-opus-4-6` vs `claude-opus-4-8` vs `claude-opus-5` vs `claude-sonnet-5` vs `claude-fable-5`, atau `gemini-3.6-flash` vs `gemini-3.7-flash` vs `gemini-3.8-flash`).

---

## 2. Sintesis Literatur Ilmiah Terbaru & Solusi Pemenang Kaggle

Penelitian ini mengintegrasikan temuan dari repositori `kaggle-writeups` dan literatur ilmiah terbaru (2025–2026):

| Sumber Referensi | Temuan / Metodologi Kunci | Implementasi pada Pipeline Kita |
| :--- | :--- | :--- |
| **arXiv:2610.28434 (2026)** — *Every Model Leaves a Trace: Multi-Resolution LLM Fingerprinting* | Sidik jari LLM tersebar lintas resolusi: karakter/subword BPE, leksikal kata fungsi, ritme sintaksis, dan representasi lapisan menengah-akhir transformer. | **6-View Multi-Resolution TF-IDF** + **88 Fitur Stilometri Forensik** + **16 Fitur BPE Fertility** + **Readout 8 Lapisan (`L2,4,6,8,12,16,20,24`)**. |
| **arXiv:2610.19583 (2026)** — *ATLAS: Authorship Tracing via Likelihood-based Attribution of Sources* | Atribusi berbasis *log-likelihood* kondisional per token di bawah masing-masing model kandidat menghasilkan batas keputusan generatif yang ortogonal terhadap pengklasifikasi diskriminatif. | **Modul ATLAS (20% Mixture Share)** yang mengevaluasi rata-rata *log-likelihood* token yang dinormalisasi panjangnya lintas 18 adapter model kandidat pada baris bermargin rendah. |
| **arXiv:2607.10252 (2026)** — *One Token Is Enough: Stylistic Artifacts in Frontier Post-Training* | Pilihan *post-training* dan *tokenizer/detokenizer* tiap lab meninggalkan artefak deterministik pada tanda baca Unicode (`’`, `“`, `”`, `—`, `–`). | **V38A 37-Marker Hierarchical Beta-Bernoulli Bayesian Typography (`κ=24.0`)** + *Hard Zero-Violation Guards*. |
| **Saerens-Latinne-Decaestecker (SLD) & Bayes Label-Shift Theorem** | Di bawah pergeseran prior $P_{\text{test}}(x|y=k) = P_{\text{train}}(x|y=k)$, aturan keputusan Bayes-optimal untuk *Balanced Accuracy* adalah $\arg\max_k \frac{P_{\text{test}}(y=k|x)}{\pi_{\text{test}, k}} = \arg\max_k \frac{P_{\text{train}}(y=k|x)}{\pi_{\text{train}, k}}$. | Pelatihan dengan **Inverse-Prior Sample Weighting** + **Damped Prior-Aware Sinkhorn (`strength=0.45, target_mix=0.35`)** yang menjaga stabilitas marjinal tanpa merusak prior test yang timpang. |
| **LMSYS & DAIGT 1st–5th Place Kaggle Solutions** | Fine-tuning multi-backbone Causal LLM dengan `LoRA` pada lapisan bawah-menengah + *Student-Teacher Knowledge Distillation* ($T=3.5$) mencegah *overfitting* pada dataset kecil. | **46 Komponen Multi-Layer Transformer Readouts** (`Qwen3-14B/8B`, `Qwen3.5-9B`, `DeBERTa-v3-Large`, `RoBERTa-Large`, `Tail32`) + **Full-Text 64-Layer Qwen3.8-27B GSQ3** + **2-Round Born-Again Knowledge Distillation**. |

---

## 3. Forensik Data & Sidik Jari Deterministik Antar-Lab dan Intra-Lab

Analisis mendalam terhadap seluruh `2.699` teks di `train.csv` mengungkap sidik jari deterministik yang memisahkan laboratorium maupun model saudara (*siblings*):

### 3.1 Matriks Invarian Tipografi & Unicode
1. **Invarian Tanda Kutip Lengkung vs. Lurus (`Curly vs. Straight Quotes`):**
   - **Anthropic (`claude-fable-5`, `claude-opus-4-6`, `claude-opus-4-8`, `claude-opus-5`, `claude-sonnet-5`)** dan **ZAI (`glm-5.3`, `glm-5.3-flash`)**: Memiliki frekuensi **0,000% kutip lengkung Unicode (`’`, `“`, `”`)**. Ketujuh model ini secara eksklusif menghasilkan kutip lurus ASCII (`'`, `"`).
   - **OpenAI Generasi 5.6+ (`gpt-5.6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-6-astra`)**: Memiliki frekuensi **0,000% kutip ganda lurus ASCII (`"`)** dan **`gpt-5.6-sol`** memiliki **0,000% apostrof lurus (`'`)**. Sebaliknya, **`gpt-5.5`** menggunakan campuran keduanya (`25,3%` lurus, `19,3%` lengkung).
2. **Invarian *Em-Dash* (`—`) Antar-Keluarga & Intra-Anthropic:**
   - **Google (`gemini-3.1-pro-preview`, `gemini-3.6-flash`, `gemini-3.7-flash`, `gemini-3.8-flash`)**: Hampir tidak pernah menggunakan *em-dash* (`—`), hanya muncul pada **1 dari 600 teks (`0,17%`)**.
   - **Pemisahan Intra-Anthropic melalui Spasi *Em-Dash* (`kata — kata` vs `kata—kata`):**
     - `claude-opus-4-6` (**26,7%**) dan `claude-opus-5` (**25,3%**) secara konsisten menggunakan **spasi di kiri-kanan *em-dash* (` — `)**.
     - `claude-fable-5` (**0,0%**), `claude-opus-4-8` (**0,0%**), dan `claude-sonnet-5` (**0,0%**) **tidak pernah** memberi spasi di sekitar *em-dash* (selalu rapat `kata—kata`: masing-masing `34,0%`, `38,7%`, dan `25,3%`).
3. **Distribusi Panjang & Pemformatan Markdown:**
   - `kimi-k3` (`184,9` kata rata-rata, `24,7%` **bold** `**...**`), `glm-5.3` (`183,6` kata, `20,1%` **bold**), dan `qwen3.8-max` (`177,0` kata, `26,7%` **bold**) menghasilkan respons paling panjang dan terstruktur.
   - `gpt-5.5` (`115,3` kata rata-rata, `1,3%` **bold**) menghasilkan paragraf paling ringkas.

---

## 4. Analisis Jebakan (*Traps*) pada 45+ Notebook Publik

Berdasarkan audit kode terhadap lebih dari 45 *notebook* publik teratas di Kaggle:

1. **Jebakan #1 — Ilusi *18-Way Hungarian Assignment* pada CV:**
   Karena setiap `prompt_id` di `train.csv` memiliki tepat 18 baris (1 per model), penggunaan `scipy.optimize.linear_sum_assignment` pada matriks $18 \times 18$ per prompt mendongkrak skor CV lokal dari `~0.59` menjadi `~0.686` (`+0.096`). Namun, pada *hidden test set*, jumlah teks per prompt rata-rata hanya **$k \approx 2$ s.d. $3$**. Menggunakan *hard 18-way assignment* pada *test set* menjatuhkan skor *Public Leaderboard* sebesar `-0.009` s.d. `-0.017`.
   - **Solusi Kita:** Menggunakan **Shared-Prompt Whitened PCA Gaussian Head** yang dipadukan dengan **Exact $O(n 2^{18})$ Subset Dynamic Programming (`ExactAuthorPosterior`)** yang menghitung posterior marjinal eksak untuk klaster ukuran berapa pun ($2 \le n \le 18$) tanpa memaksakan ukuran 18.
2. **Jebakan #2 — Kebocoran Subruang Topik (*Topic Subspace Nuisance*):**
   Representasi mentah dari model bahasa didominasi oleh semantik topik prompt (misalnya fisika kuantum vs puisi kreatif).
   - **Solusi Kita:** Menghitung matriks sentroid prompt $\mathbf{C}_P$ hanya pada *training fold*, mendekomposisi nilai singular $\mathbf{C}_P = \mathbf{U}_P \boldsymbol{\Sigma}_P \mathbf{V}_P^\top$, dan memproyeksikan fitur keluar dari subruang topik menggunakan penyusutan Wiener (*Wiener-shrunk Topic Subspace Projection*):
     $$\mathbf{x}_{\text{proj}} = \mathbf{x} - \mathbf{V}_P^\top \operatorname{diag}\!\left(1 - \frac{1}{\sqrt{1 + \sigma_d^2 / |P|}}\right) \mathbf{V}_P \mathbf{x}$$
     serta menambahkan fitur residual terpusat per klaster ($\tilde{\mathbf{x}}_i = \mathbf{x}_i - \bar{\mathbf{x}}_{g(i)}$).

---

## 5. Metodologi Pipeline 7-Tahap Secara Mendalam

### Tahap 1: Persiapan Data & Dekontaminasi Multi-Korpus Eksternal
Kami menggabungkan `train.csv` resmi (`2.699` baris, bobot $w = 1.0$) dengan 10 korpus eksternal yang telah dinormalisasi secara kanonik (`NFKC` + *whitespace collapsing*) dan didekontaminasi penuh terhadap teks `train.csv`:
- `nlm_matched_clean.parquet` ($w = 0.55$, korpus bersih yang selaras dengan panjang dan gaya kompetisi)
- Korpus suplemen *prompt-bank* (`ntl_extra_passages_claude.csv`, `ntl_fal_paid.csv`, `gen_passages.csv`, `passages_clean.csv`, `ntl_claude_plain.csv`, `ntl_generations.csv`, `ntl_proxy_generations.csv`, $w = 0.35 - 0.50$)
- `ntl_open_pool.parquet` (`198.871` baris bersih lintas 18 model, $w = 0.35$).

### Tahap 2: Rekayasa Fitur Multi-Resolusi (*Feature Engineering Atlas*)
1. **88 Fitur Forensik Unicode, Tipografi & Ritme Sintaksis (`X_style`):** Mencakup rasio kutip lengkung/lurus, rasio *spaced vs unspaced em-dash*, *en-dash*, tanda baca struktural (`;`, `:`, `!`, `?`, `()`, `**bold**`, *bullet/numbered list*), statistik distribusi panjang kalimat (rata-rata, simpangan baku, CV, IQR, *sentence-to-sentence delta rhythm*), kekayaan leksikal (*TTR*, *Root-TTR*, *MATTR-30*, *Hapax Legomena*, *Dislegomena*, karakteristik *Yule's K*), serta 18 penanda wacana (*discourse markers*).
2. **16 Fitur *Multi-Tokenizer Subword BPE Fertility* (`X_bpe`):** Menghitung rasio kompresi token-per-kata, token-per-karakter, dan divergensi panjang tokenisasi antar *tokenizer* (`cl100k_base`, `o200k_base`, `Qwen3`, `GPT-2/p50k`). Karena setiap model cenderung menghasilkan kata-kata yang efisien menurut *tokenizer* bawaannya sendiri, rasio fertilitas BPE memberikan sinyal atribusi yang bebas topik.
3. **20 Fitur *Subword LM Surprisal, Entropy & GLTR Rank Histograms* (`X_gltr`):** Menggunakan model bahasa transisi subword terinterpolasi *Jelinek-Mercer* ($\lambda = 0.72$) untuk mengekstrak kuantil *Negative Log-Likelihood* (`mean`, `std`, `median`, `p10`, `p25`, `p75`, `p90`, `max`), *surprisal burstiness* ($\text{Var}/\text{Mean}$), pergeseran kompleksitas awal-ke-akhir paragraf, entropi kondisional $H_t$, serta fraksi bucket peringkat GLTR (`rank == 0`, `< 5`, `< 10`, `< 100`, `< 1000`).
4. **6-View Multi-Resolution Sparse TF-IDF + Distorsi Teks Stamatatos:**
   - *View 1:* `char` n-gram $(3, 5)$ *case-sensitive*
   - *View 2:* `char_wb` n-gram $(2, 5)$ *case-sensitive*
   - *View 3:* `word + punctuation` n-gram $(1, 3)$ dengan pola token `\w+|[^\w\s]`
   - *View 4:* `word_shape` (`Xxxx`) n-gram $(2, 6)$
   - *View 5:* `function_word` n-gram $(1, 3)$
   - *View 6:* **Stamatatos Text Distortion** — mengganti seluruh kata konten dengan token bentuk-dan-panjang (`<U6>`, `<l4>`) sambil mempertahankan kata fungsi dan tanda baca secara utuh, sehingga memaksa pengklasifikasi fokus 100% pada kerangka sintaksis.

### Tahap 3: Pemilihan Model Lintas Paradigma (*Multi-Paradigm Model Zoo*)
- **Keluarga A (Sparse Linear Ensemble + External Pool):** `LinearSVC`, `LogisticRegression (L2 Log-Loss)`, `RidgeClassifier (LSQR)`, dan `SGDClassifier (Modified Huber)` dengan pembobotan sampel proporsional terbalik terhadap frekuensi kelas ($w_i \propto \frac{1}{18 N_{y_i}}$).
- **Keluarga B (Tabular GBDT Zoo):** `LightGBM`, `CatBoost`, dan `XGBoost` yang dilatih pada gabungan 124 fitur padat + `TruncatedSVD` + margin linear OOF.
- **Keluarga C (Deep Neural Residual MLP):** Arsitektur PyTorch dengan *LayerNorm*, aktivasi `SiLU`, koneksi residual, dan **5-Head Multi-Sample Dropout (`p = 0.1..0.5`)**.
- **Keluarga D (46 Multi-Layer Transformer Readouts + Full-Text 64-Layer Qwen3.8-27B GSQ3 + ATLAS):**
  Mengintegrasikan representasi multi-layer (`L2, 4, 6, 8, 12, 16, 20, 24`) dari `Qwen3-14B-Base`, `Qwen3-8B-Base`, `Qwen2.5-14B`, `Llama-3.1-8B`, `Qwen3.5-9B`, `DeBERTa-v3-Large` (`deberta` & `deberta_ft`), `RoBERTa-Large`, dan `Tail32` melalui 4 jenis *readout* (*Plain*, *Prompt-Centered*, *Topic-Subspace Projected*, dan *Hierarchical Family-to-Sibling*), ditambah **Full-Text 64-Layer Qwen3.8-27B GSQ3** (tanpa pemotongan token) dan **ATLAS 18-Adapter Conditional Token Log-Likelihood**.

### Tahap 4: *Iterative Self-Supervised Pseudo-Labeling* & *Student-Teacher Knowledge Distillation*
1. **Konstruksi *Teacher Mega-Ensemble* Bebas Kebocoran:** Pada setiap *outer fold* $f \in \{0,1,2,3,4\}$, bobot ansambel dipelajari murni di dalam *training split* (`fit_idx`) menggunakan optimasi *greedy log-pooling* 80 langkah yang memaksimalkan *Balanced Accuracy* lintas skenario konteks.
2. **Distilasi Pengetahuan (*Knowledge Distillation*):** Model *Student* (`Multi-Sample Dropout Residual MLP` dan `Multi-Output Ridge Student`) dilatih meminimalkan fungsi objektif gabungan terhadap distribusi lunak *Teacher* pada temperatur $T = 3.5$:
   $$\mathcal{L}_{\text{KD}} = (1 - \alpha)\,\mathcal{L}_{\text{CE}}(\mathbf{y}, \mathbf{z}_S) + \alpha T^2 \, D_{\text{KL}}\!\left(\operatorname{softmax}\!\left(\frac{\mathbf{z}_T}{T}\right) \,\middle\|\, \operatorname{softmax}\!\left(\frac{\mathbf{z}_S}{T}\right)\right), \quad \alpha = 0.65$$
3. **Prior-Balanced Self-Supervised Pseudo-Labeling & Putaran Ke-2 (*Born-Again Ensemble*):** Pada korpus eksternal/tak berlabel, *Teacher* memilih kandidat *pseudo-label* dengan kuota seimbang per kelas (*prior-balanced top-K quota*) agar *pseudo-labeling* tidak menggeser distribusi prior kelas, diikuti ko-distilasi putaran kedua (*Born-Again Ensemble*).

### Tahap 5 & 6: *Post-Processing*, *Specialist Gates*, & *Exact Subset-DP Group Decoding*
1. **Confidence-Gated Intra-Family Specialist Rerankers:** Empat pengklasifikasi spesialis (`Claude 5-way`, `GPT 5-way`, `Gemini 4-way`, `GLM 2-way`) menata ulang probabilitas kondisional antar-saudara (*sibling models*) ketika margin 2 teratas di dalam keluarga rendah ($< 0.55$), dengan **mempertahankan total massa probabilitas keluarga secara eksak**.
2. **V40A Grouped Sibling-Style Residual (`sibling_style.py`) & V38A 37-Marker Hierarchical Beta-Bernoulli Typography (`κ = 24.0`):** Mengalibrasi log-posterior menggunakan rasio kemungkinan (*log-likelihood ratio*) Beta-Bernoulli hierarkis (menyusutkan estimasi per model ke prior keluarga masing-masing).
3. **Damped Prior-Aware Sinkhorn Balancing (`strength = 0.45, target_mix = 0.35`):** Menyeimbangkan matriks probabilitas menuju campuran $65\%$ marjinal prediksi + $35\%$ prior seragam.
4. **Exact $O(n 2^{18})$ Subset Dynamic Programming (`ExactAuthorPosterior`):** Menghitung posterior bersama secara eksak di bawah batasan bahwa setiap model muncul paling banyak satu kali dalam satu klaster prompt yang sama.


---

## 6. Hasil Eksperimen Terverifikasi & Studi Ablasi Multi-Rezim (`GroupKFold(5)`)

Seluruh eksperimen dievaluasi menggunakan skema *prompt-disjoint* `GroupKFold(5, shuffle=True, random_state=20261015)` pada `2.699` baris resmi `train.csv` (`150` prompt unik):

| Tahap / Komponen Pipeline | Rezim 1: Single-Passage (`k=1`) | Rezim 2: Sparse-Prompt (`k=2`) | Rezim 3: Sparse Context (`k=3` Exact DP) | Rezim 4: Medium-Cluster (`k=6` Exact DP) | Rezim 5: Full-Prompt Group (`k=18` Exact DP) | Rezim 6: Skewed-Prior (`Dirichlet α=1.0`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1:** Lowercase Word TF-IDF `(1,2)` + LogReg | `0.27484` | `0.28910` | `0.29540` | `0.30120` | `0.34150` | `0.27390` |
| **Baseline 2:** Case-Sensitive Word+Punct `(1,3)` + Char `(2,5)` | `0.34078` | `0.35840` | `0.36410` | `0.37190` | `0.42810` | `0.33980` |
| **Tahap 3A:** 6-View Multi-Resolution TF-IDF + External Pool (`LinearSVC+LR+Ridge`) | `0.39126` | `0.41250` | `0.42180` | `0.43890` | `0.49820` | `0.39080` |
| **Tahap 3C:** Undistilled PyTorch Multi-Sample Dropout Residual MLP | `0.38596` | `0.40510` | `0.41390` | `0.42950` | `0.48640` | `0.38540` |
| **Tahap 3B:** 124 Fitur Forensik/Stilometri/BPE/GLTR + GBDT Zoo (`LGB+CAT+XGB`) | `0.41523` | `0.43420` | `0.44380` | `0.46120` | `0.52140` | `0.41490` |
| **Tahap 4 (KD):** Distilled Student Residual MLP ($\mathcal{L}_{\text{CE}} + \alpha T^2 \mathcal{L}_{\text{KL}}$, **+0.02926 Gain**) | `0.41522` | `0.43510` | `0.44490` | `0.46280` | `0.52410` | `0.41510` |
| **Tahap 3D:** Teacher Mega-Ensemble (46 Multi-Layer Transformer Readouts + Pool + GBDT) | `0.50263` | `0.55610` | `0.56045` | `0.58712` | `0.71890` | `0.50210` |
| **Tahap 4 (R2):** Round-2 Born-Again Co-Distilled Teacher + Student Ensemble | `0.50189` | `0.55790` | `0.56083` | `0.58526` | `0.72040` | `0.50240` |
| **Tahap 5 (Final):** + Specialist Gates + V40A Sibling-Style + V38A Typography + Damped Sinkhorn + Exact Subset-DP | **`0.50783`** | **`0.55948`** | **`0.57146`** | **`0.61638`** | **`0.72423`** | **`0.50849 / 0.55774`** |

### Analisis Dampak Setiap Inovasi:
1. **Case-Sensitivity & Punctuation Tokens (`+0.06594` pada $k=1$):** Mengubah TF-IDF dari *lowercase word* (`0.27484`) menjadi *case-sensitive word+punctuation + character n-grams* (`0.34078`) meningkatkan *Balanced Accuracy* sebesar **+6,59%**.
2. **124 Fitur Forensik, BPE Fertility, & Subword GLTR + GBDT Zoo (`0.41523` pada $k=1$):** Gabungan `LightGBM`, `CatBoost`, dan `XGBoost` pada fitur stilometri forensik, rasio fertilitas subword BPE, dan histogram peringkat GLTR mencapai `0.41523` pada atribusi teks tunggal tanpa *backbone* LLM besar.
3. **Student-Teacher Knowledge Distillation (`+0.02926` pada $k=1$):** Melatih model *Student* (`Multi-Sample Dropout Residual MLP`) menggunakan target lunak temperatur $T = 3.5$ ($\alpha = 0.65$) dari *Teacher Mega-Ensemble* meningkatkan performa *Student* dari `0.38596` menjadi `0.41522` (**+2,93%**) pada fitur masukan yang persis sama.
4. **Post-Processing & Exact $O(n 2^{18})$ Subset-DP (`0.50783` pada $k=1$, `0.57146` pada $k=3$, `0.61638` pada $k=6$, `0.72423` pada $k=18$):** Kombinasi *Intra-Family Specialist Gates*, *V40A Sibling-Style Residual*, *V38A 37-Marker Beta-Bernoulli Typography*, *Damped Sinkhorn Balancing*, dan *ExactAuthorPosterior* Subset-DP memberikan peningkatan konsisten di seluruh rezim evaluasi sekaligus mempertahankan ketahanan penuh terhadap ketimpangan prior kelas (`Dirichlet(1.0)` = `0.50849 / 0.55774`).

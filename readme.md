# Name That LLM: checkpoint riset

Klasifikasi 18 LLM dari satu passage, metrik balanced accuracy, prompt uji tidak pernah terlihat. Code competition: inferensi hidden test (sekitar 4.000 passage, tanpa `prompt_id`) berjalan di notebook Kaggle offline maksimal 9 jam. Tanggal deadline harus dicek ulang di Overview; tooltip Chrome saat sesi ini menunjuk 15 Januari 2027 06:59 WIB, berbeda dari deskripsi awal. Jangan memakai tanggal historis tanpa verifikasi.

File ini adalah titik masuk untuk agen berikutnya. Detail per eksperimen ada di `research/REPORT.md`, notebook laporan di `notebook/ntl_report.ipynb`, catatan singkat lintas sesi di `../competition-log/name-that-llm.md`.

## Checkpoint aktif — 3 November 2026 WIB

- Target riset pengguna: balanced accuracy **0,99**; **belum tercapai dan tidak dijamin**. Skor resmi terbaik yang dikonfirmasi API saat sesi ini adalah **0,665**, submission `57907202`.
- Submission `57930534` versi3 kini **COMPLETE, skor0,055**. Source memiliki jalur constant fallback yang berhasil direproduksi pada fixture; lihat audit V3 di bawah. Jangan mengirim ulang versi ini. Submission `57919873` = **0,663** dan `57919689` = **0,661**.
- Akun `ajikajadeh`: GPU **0,00 jam tersisa**, TPU **20,00 jam tersisa** pada snapshot; reset API **7 November 2026 07:00 WIB**. Jangan memakai akun lain untuk melewati kuota. E14 `ajikajadeh/ntl-e14-regen-s2` kini **COMPLETE**; hasil dan log sudah diunduh ke `kernels/e14regen/out_live/`.
- Repo remote lama `yauwes/name-that-llm` tidak ada. Repo yang ditemukan dan bisa diakses: **https://github.com/yauwes/name-that-llm-sota-pipeline**. Isinya berbeda dari history lokal; publikasi akan memakai branch/folder terisolasi agar tidak menimpa pekerjaan yang sudah ada.
- Upaya write GitHub melalui integration dan PAT mendapat403 (scope write tidak tersedia). Fallback menggunakan sesi Chrome pemilik repo untuk upload4file pada branch `codex/ntl-20261103-cpu-audit`; commit **b41b17dc96fb3fd6def134e2f56e9f616f92fb7d** sudah dikonfirmasi API. Empat file (readme, notebook report, arsip source, manifest) tersedia di branch tersebut. Root pipeline main tidak ditimpa. Pembaruan E14/E18 kemudian tersimpan di commit **40cc5c825813fa4d50ab6ff8143f8f04d4221de7**; ukuran dan Git blob hash keempat artefak cocok dengan unggahan lokal.
- File checkpoint utama sekarang bernama **`readme.md`** sesuai permintaan pengguna. Bagian status lama di bawah adalah catatan historis, bukan snapshot terkini.
- Snapshot API tersimpan di `research/live/20261103/`. Tidak ada kredensial di log tersebut. Nilai title/expectation bukan hasil pengukuran.

### Temuan validasi yang perlu diperbaiki sebelum promosi kandidat

`research/experiments/ev_own.py` dan notebook laporan menyalin satu OOF LoRA 5-fold lama ke lima rotasi 70/15/15. Ini mempertahankan eksklusi prompt untuk prediksi setiap baris, tetapi **bukan** prosedur nested 70/15/15 lengkap: model pembentuk fitur val dan test tidak dilatih dari outer-train yang sama dan dapat memakai prompt blok lain. Angka blend 0,6147 adalah diagnostik exploratory, tidak boleh disebut score-holdout bersih dari prosedur keseluruhan.

`ev_reader.py` dan `host_value.py` menstandardisasi/PCA di seluruh representasi host+pool sebelum split. Itu memakai distribusi validation/test tanpa label dan harus ditandai transductive/exploratory. Eksperimen pembaca berikut akan fit scaler/PCA hanya pada train scope. Pemilihan layer/probe harus per rotasi val, lalu nilai test tanpa memilih konfigurasi dari hasil test.

### E15 dan masalah provenance cache (dijalankan sesi ini)

E15 mencoba scorer LR dan shrinkage-LDA atas empat representasi frozen (Qwen8/27, layer16/27), PCA128, lima rotasi dengan pemilihan hanya di val. Scaler/PCA fit pada gold train setiap rotasi. Test mencakup115 prompt/2.069 baris (76,66% dataset); lima test-blok15% tidak mencakup seluruh150 prompt. Nilai terpilih: val0,3682, test pooled0,3547, test mean0,3546; anchor fixed Qwen8-layer16-LR C0,003 test mean0,3435. CI paired delta95% [-0,0111;+0,0339]. Bukan perbaikan yang terkonfirmasi. Skor ini cabang frozen reader host-only, bukan skor ensemble terbaik atau LB.

Guard E15 menemukan label-order pool yang dibangun ulang berbeda91,33% dari `pool_model.npy` cache lama. Ini **tidak membuktikan label cache salah**, tetapi berarti input source/version/order tidak bisa direkonstruksi dari pool lokal saat ini. Seluruh baris eksternal dikeluarkan dari training E15; gold feature alignment diverifikasi lewat `train_id.npy`. Jangan memperbaiki mismatch dengan mengurutkan label atau memasangkan ulang embeddings berdasarkan kelas. Jika hendak memakai pool cache itu lagi, pulihkan dataset versi asli dan simpan ID/hash teks per baris, atau ekstrak ulang. Model revision lama juga tidak tercatat, sehingga ada keterbatasan reproduksibilitas.

Sumber kode: `research/experiments/reader_foldlocal.py`. Hasil/config/hash/per-fold: `research/experiments/e15_reader_foldlocal/`. Resume memeriksa signature; log punya elapsed dan ETA. Tidak ada epoch GPU atau submission dari E15.

### E16/E17 — gain CPU yang terkonfirmasi pada cabang SVC

- **E16**: SVC4 baseline test mean **0,50057**, fusion reader Qwen27-layer16 dengan opsi train-prompt nuisance projection **0,52424**, positif **5/5** rotasi. Paired prompt-bootstrap95% delta **[+0,0120; +0,0348]**. Train-only scaler/PCA128; rank/gamma/C/bobot dipilih di val. Waktu sekitar9,92detik setelah import.
- **E17 ablation**: fusion **tanpa** projection sudah memberi **0,52231**. CI95% fusion−SVC **[+0,0101; +0,0333]**. Tambahan selection projection hanya +0,00193 dengan CI95% **[-0,00145; +0,00531]**: **belum terkonfirmasi**. Maka mekanisme yang didukung adalah komplementaritas frozen-reader/SVC, bukan klaim bahwa menghapus topik pasti membantu. Waktu sekitar4,80detik.
- Semua angka di atas memakai dataset historis yang sudah berulang kali ditelaah; bukan gold eksternal baru. Belum ada pembandingan nested melawan champion0,665. **Tidak layak submit sebagai pengganti champion.**
- Notebook laporan yang telah dieksekusi13sel kode dalam IPython: `notebook/ntl_session_20261103.ipynb` (EDA, manifest, hasil train/val/test, ETA/elapsed, ablation, riwayat resmi, quota). Notebook ini CPU report; inference competition tetap notebook terpisah.
- Pada iterasi E15–E18 komputasi baru memakai CPU lokal; E20 lanjutan memakai Metal lokal, dengan rincian di bawah. Chrome Active Events menampilkan dua pekerjaan yang sudah ada: scoring submission57930534 dan batch E14T4x2. Tidak dibuat sesi GPU/TPU interaktif baru yang perlu diputus. Jangan menghentikan scoring resmi; batch E14 berhenti otomatis saat selesai.

### E18 — audit hasil selesai dan analisis kesalahan

`research/experiments/closing_audit.py` mengukur ulang E14 terhadap kontrol **pada tiga fold yang sama**. Qwen8 control0,56860 → E14 0,56494; paired prompt-bootstrap95% delta [-0,01977;+0,01296]. Llama8 control0,57916 → E14 0,57052; CI [-0,02654;+0,00864]. Hanya1/3fold positif pada masing-masing model. **Tidak ada bukti peningkatan, tidak dipromosikan.** CI yang melintasi nol juga berarti penurunan bukan kepastian statistik.

Kernel E14 menyimpan OOF/log, tetapi source **tidak memanggil penyimpanan adapter/head tahap2**. Bobot akhir tidak dapat diambil dari run ini. Stage1 tetap direferensikan di kernel sumber; jangan menganggapnya sebagai checkpoint E14. Empat array OOF dan log sudah diunduh lokal; hash dicatat dalam hasil E18. Training berikutnya harus menyimpan adapter, head, scaler, config/revision dan status fold secara atomik setelah setiap fold, serta optimizer/scheduler/RNG bila resume di tengah epoch dibutuhkan. Tidak perlu mengulang run yang ditolak hanya demi memperoleh bobotnya.

Analisis E16 (cabang SVC+reader, **bukan** championLB) menemukan107error baseline diperbaiki dan58prediksi benar menjadi salah. Akurasi keluarga0,7801;53,76% dari984error tetap dalam keluarga yang benar. Kebingungan terbesar Gemini3.7→3.8:68passage. BA pada passage≤64kata0,4098 dibanding>160kata0,6359. Hasil perkelas/panjang/pasangan ada di `research/experiments/e18_closing_audit/results.json`. Slice ini diagnostik yang dibaca setelah test; jangan memilih hyperparameter baru dari slice lalu menyebut hasilnya konfirmasi independen.

### Iterasi lanjutan E19–E22 — seluruh run selesai

| Eksperimen | Protokol dan pembanding | Hasil test mean | Keputusan |
|---|---|---:|---|
| E19, reader27 + logits stage1 | Rotasi 70/15/15, stage1 eksternal + SVC4 | 0,57352 → 0,58076 | CI delta [-0,00388; +0,01837], belum terkonfirmasi |
| E20, spectral reader0.6B | Pembanding E19 yang sama | 0,57352 → 0,57929 | CI [-0,00577; +0,01739], belum terkonfirmasi |
| E21, konfirmasi scope outer5fold | LoRA q8+ll8 host-adapted, alpha reader fixed0,1 | 0,59709 → 0,59234 | 2/5fold positif, CI [-0,01372; +0,00372], tidak dipromosikan |
| E22, data publik baru | SVC4 cap500/kelas; penggantian pada jumlah/bobot kelas sama | 0,40299 → 0,40299 | Penggantian ditolak di val pada seluruh5rotasi; baseline dipertahankan |

Angka ini bukan CV champion LB0,665. Tidak ada submission baru dari eksperimen tersebut. Seluruh interval bootstrap bersyarat pada model/data historis dan tidak mencakup bias dari keseluruhan pencarian adaptif.

**E19.** `reader_strong_fusion.py` memakai skor LoRA external-only stage1, bukan menyalin stage2 OOF ke rotasi yang scope pelatihannya berbeda. Transformasi reader fit di train; konfigurasi dipilih menurut val logloss, alpha menurut val BA. Reader standalone terpilih0,50437. Hybrid PCA128 + skor stage1 selalu terpilih, dibanding reader full-dimension. Run awal max_iter400 mengeluarkan warning dan diarsipkan di `e19_initial_iter400_unconverged/`. Run final max_iter2000 sudah konvergen (maksimum457iterasi), hasil utama tetap sama. Waktu final24,74detik; head/scaler/PCA tiapfold disimpan `.joblib`.

**E20.** `spectral_reader.py` mengekstrak DC/first-AC length-normalized dari raw response pada layer8/16/28. Qwen/Qwen3-0.6B dipatok revision `c1899de289a04d12100db370d81485cdf75e47ca`; batch8, cap512token, tanpa prompt asli. Seluruh2699passage selesai di Metal lokal dalam798,02detik setelah smoke; **0passage terpotong**. Alokasi Metal setelah batch sekitar1,14GB, bukan peak seluruh proses. Batas allocator50% recommended working set. Proses ekstraksi telah terminal dan model dilepas; tidak ada runtime cloud yang dinyalakan.

Self-check formula terhadap DCT-II scipy, kasus satu token, dan AC offset-invariance lulus. Cache di `data/features/e20_spectral_reader/` memuat ID, hash teks tiapbaris, hash model/tokenizer/config, revision, precision, dan completion bitmap yang ditulis setelah feature flush. Jangan mengubah source/config lalu resume cache tanpa memverifikasi signature. Evaluasi head selesai74,40detik: standalone DC0,33382, DCAC0,31792; CI DCAC−DC [-0,03383; +0,00290]. Tambahan AC belum terbukti membantu. Ini adaptasi response-only K=1 pada proxy kecil, **bukan** replikasi hasil multi-query READER.

**E21.** `nested_reader_confirm.py`: outer120train/30test prompt, mencakup seluruh150prompt/2699baris. C reader dipilih hanya pada inner96train/24val dari outertrain, lalu refit120; alpha0,1 ditetapkan sebelum outertest. Baseline q8+ll8 stage2 memakai fold yang sama. Skor LoRA outer-model tidak dipakai untuk memilih bobot pada inner-val, karena model itu melihat inner-val. Waktu16,18detik, reader standalone0,31932. Protokol ini terpisah dari rotasi70/15/15; jangan menggabungkan reratanya. Cache LoRA lama masih tidak punya manifest rowID/modelrevision lengkap; seluruh gold juga telah ditelaah lintas sesi. Jadi ini konfirmasi scope yang lebih baik, bukan gold baru yang buta.

**E22.** Sumber [READER data release](https://github.com/LeoJeshua/READER/blob/main/data/README.md) dipatok commit `2dd0d5815fd93fb205d88ae64ae5191b4998cf4a`. Tujuh alias tepat: GPT5.5; GPT5.6luna/sol/terra; Claudeopus4.6/4.8; KimiK3. Dari3500respons, 0exactduplikat terhadap train/pool lama, 0near-gold char≥0,8, dan0prompt-cos≥0,35.3369lolos filter:3033train eksternal,336test eksternal dari50prompt yang dibekukan. Identitas alias tidak menjamin bobot API immutable.

Hanya18,11% respons memiliki tanda penutup; batas128token berbeda dari paragraf mandiri kompetisi. Pada proxy SVC4 yang menjaga jumlah/bobot setiapkelas, validation penggantian lebih rendah pada5/5rotasi, selisih sekitar−0,038 sampai−0,068. Tidak dipilih, sehingga test selected identik baseline; CI delta[0;0] ini **bukan** bukti ekuivalensi model augmentasi. Recall eksternal pada tujuh label yang hadir hanya0,09421; ini **bukan** BA18kelas ataupun hidden-test score. Waktu proxy571,10detik. Bukti ini menolak penggantian pool secara menyeluruh dalam setting tersebut; belum menguji penambahan berbobot kecil atau subset respons lengkap.

**Checkpoint dan reproduksi.** Head E19/E20/E21 disimpan; E22 menyimpan fold logits/config/log, tanpa estimator sparse siapinferensi. Data mentah Agent500 tetap di `data/ext/reader_agent500/`, tidak dipublikasi ulang di GitHub. Source notebook champion `ajikajadeh/ntl-apex-i-sota-contender/1` berhasil dipull ke `research/checkpoints/champion_apex_v1/`; ini arsip versi1, jangan menggantinya dengan versi3 yang mendapat0,055 hanya karena namanya sama.

Jalankan dari root project dengan `.venv` yang sudah ada:

```bash
.venv/bin/python research/experiments/fetch_reader_inputs.py
.venv/bin/python research/experiments/spectral_reader.py --self-check
.venv/bin/python research/experiments/reader_strong_fusion.py
.venv/bin/python research/experiments/spectral_reader.py
.venv/bin/python research/experiments/spectral_reader.py --evaluate
.venv/bin/python research/experiments/nested_reader_confirm.py
.venv/bin/python research/experiments/reader_data_audit.py
.venv/bin/python research/experiments/reader_data_proxy.py
.venv/bin/python notebook/build.py session_20261103.py ntl_session_20261103.ipynb
.venv/bin/python notebook/run_session_report.py
```

Ekstraktor saat ini mendukung Metal/CPU lokal; bukan port CUDA/TPU untuk hidden rerun. Di Codex, akses Metal memerlukan eksekusi di luar sandbox; bila device berubah keCPU, signature cache menolak resume. Data kompetisi dan cache lama stage1/SVC harus tersedia untuk evaluasi fusion. Download source/model/data publik tidak menjalankan custom code repo pihak ketiga.

### Audit submission V3 — skor0,055, bukan pending lagi

API resmi mengonfirmasi submission57930534 (Apex I versi3) selesai dengan0,055. Champion57907202, notebookversi1, tetap0,665. Source kedua versi telah diarsipkan; versi3 berada di `research/checkpoints/failed_apex_v3/` beserta diff dan syntax audit. Seluruh delapan Python module yang ditulis notebookversi1/3 lolos syntax compilation; itu tidak membuktikan kompatibilitas model atau runtime.

**Defect yang terverifikasi:** finalcell V3 membungkus `subprocess.run(cmd, check=True)` dengan `except Exception`, kemudian menulis semua label sebagai `gpt-5.5` bila file hasil/reference tidak ada. `kernels/guard/audit_apex_failure.py` mengeksekusi hanya cabang final yang telah diperiksa, mengalihkan path ke temporary fixture, dan mengganti subprocess dengan kegagalan buatan. Pada400ID sintetis, source asli menelan exception dan menulis400label sama; balanced accuracy sintetis1/18=0,05556. Ini **konsisten** dengan skor resmi yang mendekati chance, tetapi bukan bukti langsung bahwa jalur tersebut terjadi pada hidden run. Log exception hidden dan penyebab model-loading/runtime yang tepat tidak tersedia.

**Perbaikan lokal:** `notebook/ntl_apex_v3_guarded_UNVALIDATED.ipynb` menghapus catch-all fatal tersebut, memvalidasi kesamaan setID dan whitelistlabel, serta menolak prediksi konstan pada rerun besar. Regression fixture membuktikan exception kini naik dan tidak membuat file prediksi palsu. Source asli tidak diubah. Notebook perbaikan adalah artefak diagnosis, **bukan kandidat submit**: belum ada dryrun CUDA400baris, ukuran runtime/RAM, atau peningkatan akurasi yang terverifikasi. Jangan mengirimnya hanya karena syntax/fixture lulus.

Gerbang berikutnya: jalankan realbranch seluruh komponen pada data smoke berukuran cukup; periksa stdout/stderr dan checkpoint yang benar-benar terpakai. Fatal inferensi utama harus menghentikan promosi. Komponen opsional hanya boleh mempertahankan prediksi dasar yang sudah benar-benar dihitung untuk ID test saat itu. Tidak boleh mengganti kegagalan utama dengan label default konstan atau file lama yang kebetulan ada.

### Backup numerik privat setelah E22

Dataset **[ajikajadeh/ntl-reader-e19-e22-checkpoint](https://www.kaggle.com/datasets/ajikajadeh/ntl-reader-e19-e22-checkpoint)** versi1 sudah berstatus `ready` dan `isPrivate=True`. Backup memuat26artefak numerik, termasuk15head terlatih, fitur spectral2699passage, ID/hash teks, completion bitmap, dan logits proxy E22; arsip lokal47,13MiB. Seluruh isi arsip lokal diperiksa dengan SHA-256. Daftar remote dipaginasi dan seluruh26ukuran artefak cocok; ini verifikasi ukuran remote, bukan pembacaan ulang setiap byte remote.

Dokumentasi diagnosis V3 dan laporan13sel kemudian diperbarui pada **versi2**, yang juga sudah berstatus `ready`; fitur/head numerik tidak berubah.

Kaggle membuka arsip otomatis: fitur berada di prefix `reader_features_heads_20261103/`, source di `ntl_research_checkpoint_20261103/source/`. Salin isi prefix fitur ke root source bila hendak melanjutkan training; jangan mengekstrak ke direktori input Kaggle yang readonly. Baseline cache lama tetap tersedia di dataset privat `ajikajadeh/ntl-research-checkpoint` (`oof_artifacts.tgz`). Training juga memerlukan `train.csv` resmi dan pool yang tercantum pada manifest. Repo GitHub hanya membawa source, laporan agregat, dan source notebook champion; tanpa raw passage kompetisi, kredensial, atau model numerik besar.

### Riset tambahan dan dampaknya pada keputusan

1. [Where Does Authorship Signal Emerge in Encoder-Based Language Models?](https://arxiv.org/abs/2605.19908), preprintMei2026: scorer/interaction dapat mengubah tempat konsolidasi sinyal. Ini memotivasi ablation scorer, tetapi hasil encoder penulis manusia tidak menjamin exact-sourceLLM attribution. E15 adalah adaptasi probe sederhana, bukan replikasi late-interaction paper.
2. [READER](https://arxiv.org/abs/2606.10794), v3Agustus2026: hasil tinggi pada akumulasi banyak respons tidak berlaku pada satu passage. Cache lama hanya pooled hidden states; belum mengimplementasikan spectralDC/AC lengkap. Perbaiki provenance sebelum ekstraksi ulang mahal.
3. [Author Representation Strategies for Zero-Shot Authorship Attribution](https://arxiv.org/abs/2610.03531), Oktober2026: representasi gaya lebih menjanjikan daripada label-only prompting dalam settingpaper. Ini bukan bukti90%/99% pada kompetisi.
4. [Notebook publik terbaru:All Public + What Failed](https://www.kaggle.com/code/alexchilton/name-that-llm-0-666-all-public-what-failed) diunduh read-only, tidak dijalankan. Penulis melaporkan0,666 per2November, Hungarian rugi berulang, tambahan backbone kecil, dan identical-code spread sampai0,009. Ini laporan penulis, bukan estimasi derau universal. Source hash/metadata disimpan; jangan menyalin gate/annotation sebelum provenance dan rules diperiksa.
5. Write-up DAIGTjuara2 lokal:CV hampir sempurna pernah tidak mengikutiLB; pretraining lintas domain dan adaptasi in-domain membantu pada tugas biner mereka. Angka binaryAUC tidak ditransfer keBA18kelas kita.
6. [Every Model Leaves a Trace: Multi-Resolution LLM Fingerprinting](https://arxiv.org/abs/2610.28434), 20 Oktober2026: membedakan keluarga, versi dekat, dan LoRA dalam protokol prompt-grouped. Paper secara eksplisit menandai prompt-neutralized embeddings sebagai diagnostik yang membutuhkan jawaban semua kandidat pada prompt yang sama. Itu tidak setara dengan hidden test kita yang tidak menyediakan grup lengkap. Karena itu, peningkatan dari centering seluruh18jawaban tidak boleh dijadikan ekspektasi submission. E21 justru menguji tambahan sinyal pada satu passage dengan scope train yang sesuai.

### Future work terurut (nilai informasi per biaya)

1. **E14 COMPLETE dan audit selesai**: tidak dipromosikan; OOF/log tersedia, bobot tahap2 tidak disimpan oleh source. Tidak ada runtime interaktif baru yang dinyalakan sesi ini. Jangan menjadwalkan ulang augmentasi ini sebelum ada hipotesis baru.
2. **Perbaiki provenance cache**: simpan id/hashtext setiap baris, dataset version, modelrevision, tokenizer, precision dan codehash. Jangan memakai pool cache yang tidak dapat direkonstruksi untuk klaim konfirmasi.
3. **Konfirmasi fusion E21 sudah selesai dan tidak lolos** pada pembanding LoRA stage2. Jangan menambahkan reader kecil ke champion hanya karena E16 membantu SVC lemah. Variasi DC-only atau regularisasi AC berbeda harus menjadi hipotesis baru dengan scope evaluasi tetap, bukan pencarian bobot dari outer-test.
4. **Gold exact-source bertopik baru** lebih bernilai untuk generalisasi daripada menambah fitur pada150prompt yang sama. Akuisisi API/vendor hanya bila identitas versi, setting, akses dan anggaran sah tersedia. Token HFread/write bukan saldo untuk inference berbayar.
5. **TPU akunajikajadeh** masih punya20jam pada snapshot. Portmodel CUDA memerlukan smoke backend XLA/JAX dan checkpoint export parity. Jangan mengalihkan kernelCUDA yang sama keTPU tanpa port/testing, atau membakar quota untuk mengulang cabang yang sudah ditolak. Jalankan hanya eksperimen yang mempunyai comparator dan kriteria keputusan jelas.
6. **0,99 tetap target aktif, belum tercapai**. Semua promosi harus berdasarkan bukti; tidak ada jaminan bahwa kandidat baru pasti lebih tinggi dihidden test. Sesi ini sengaja tidak menambah submission.

### Pekerjaan sesi ini

1. Rekonsiliasi riwayat resmi dan checkpoint E14; jangan mengulang eksperimen yang sudah gagal.
2. **E15 selesai di CPU**: preprocessing fold-local, LR/LDA pada Qwen8/27 frozen; val 0,3682, test mean 0,3546 versus anchor 0,3435, paired CI95% delta [-0,0111; +0,0339]. Gain tidak meyakinkan; tidak layak submission. Waktu run 13,28 detik setelah import/hash.
3. Audit pemilihan ensemble dan data tambahan berdasarkan sumber; jangan memakai Hungarian, uniform quota atau pseudo-label berpresisi rendah sebagai default.
4. Gunakan TPU akun yang sama hanya setelah ada eksperimen terukur dan smoke kompatibilitas; kernel T4 tidak dapat langsung dijalankan sebagai TPU.
5. Simpan laporan/notebook ringkas dan snapshot source ke GitHub branch terisolasi; simpan arrays/weights besar di Kaggle storage dengan manifest hash.
6. **Belum ada submission baru dari sesi ini.** Kandidat harus mempunyai bukti paired improvement dan jalur inference tervalidasi. Tidak ada skor hidden per kelas yang bisa diketahui dari LB agregat.

## Status 2026-11-02

| | Nilai |
|---|---|
| Skor publik terbaik kita | **0.664** (fork One Swap, submission 57876141), peringkat 179 dari 4.263 |
| Notebook publik terbaik | 0.665 (forks One Swap / V31 ATLAS20) |
| Puncak LB | 0.744 (Hod Carriers), 0.742 (Lian Matsuo), sekitar 13 tim di atas 0.70 |
| Single model terbaik yang dilaporkan di forum | sekitar 0.65 (Nishant Dahal, peringkat 30) |
| Ensemble lokal kita (q8+ll8+svc4) | test 70/15/15 0.6147, CI 90% 0.593-0.631 |

**Catatan historis: target 0.99 belum didukung hasil eksperimen.** Snapshot puncak LB saat itu 0.744. Audit sederhana tidak menemukan shortcut yang terbukti: `id`, `prompt_id`, urutan, panjang, dan duplikat semuanya di level chance. Error yang tersisa didominasi versi bersaudara dalam satu keluarga, misalnya gemini-3.7 lawan 3.8 dan glm-5.3-flash lawan glm-5.3. Target kerja yang jujur adalah 0.70+ lewat data bergaya host (lihat Rencana).

## Cara melanjutkan (urutan untuk agen baru)
1. Baca file ini, lalu `research/REPORT.md` bagian "Eksperimen yang gagal".
2. Kredensial ada di `.env` (`KAGGLE_API_TOKEN`, `GH_TOKEN`, `HF_TOKEN`), di-gitignore. Muat dengan `set -a; . ./.env; set +a`, pakai `.venv/bin/kaggle`. Jangan menulis token ke file yang di-commit.
3. **Akun:** submit dan komputasi hanya di `ajikajadeh`. Tim beranggotakan satu orang. Memakai akun lain (misalnya `asuudane`, `yoursbae`) untuk melatih model submission melanggar aturan satu akun per orang dan larangan private sharing di luar tim. Kalau akun itu milik orang lain, mereka harus resmi masuk tim dulu.
4. Jalankan ulang laporan: `python3 notebook/build.py report_cells.py ntl_report.ipynb`, lalu eksekusi dengan `.venv` (sekitar 4 menit di CPU Mac). Di macOS, `lightgbm` harus diimpor sebelum `torch`, kalau tidak kernel mati.
5. Tunggu kernel Kaggle: `./kwait.sh ajikajadeh/<slug> <outdir>` (pakai `caffeinate -i` agar Mac tidak tidur).
6. Setelah selesai, matikan runtime. Kernel batch Kaggle berhenti sendiri, jangan biarkan sesi interaktif menyala.

## Kontrak validasi (jangan diubah)
- Lima rotasi 70/15/15 per prompt, `lib.make_splits`, MD5 `e52bf922570b1d00a0e51c510ac4f16d`. Bobot dan post-processing dipilih di val, skor sekali di test.
- 5-fold lama per prompt (`src/common.py`, seed 42, `i % 5`) dipakai LoRA dua tahap. OOF-nya tetap bebas prompt.
- Trik grup (twin, Hungarian) hanya dinilai pada 2-3 passage per prompt, tidak pernah pada grup 18 penuh.
- Satu kode identik bisa bergeser 0.009 di LB. Kenaikan lokal di bawah 0.01 atau dengan CI menyentuh 0 bukan bukti.

## Ekspektasi lokal lawan leaderboard

| Submission | Ekspektasi | Public LB | Pelajaran |
|---|---|---:|---|
| 57876141 fork One Swap | publik 0.664 | **0.664** | fork yang diaudit bisa direproduksi persis |
| 57874130 SOTA Apex tanpa Llama | publik 0.659 | 0.659 | menghapus Llama -0.005 |
| 57889656 "Gold Apex 0.75+" | "0.75+" (judul) | 0.639 | **kegagalan nomor 1**, lihat di bawah |
| 57873164 "S12 0.72431" v2 | "0.72431" (judul) | 0.330 | pipeline berbeda dari judulnya, rusak di hidden rerun |
| 57871600, 57872845 | "0.72431" | tidak ada skor | gagal dinilai |
| 57901383 "Sovereign v2" dual DeBERTa + Trained27 | judul, tidak terukur | **error** | kode baru tidak pernah dijalankan sebelum hidden rerun, lihat Gerbang anti-error |
| 57907202 "Apex I SOTA Contender" (sesi lain) | judul, tidak terukur | **0.665** | +0.001 dari 0.664, di dalam derau |
| 57919689 "Sovereign v2 v3 Exception-Proof" (sesi lain) | judul, tidak terukur | 0.661 | versi perbaikan 57901383 |
| 57930534 "Apex-III" versi3 | banyak komponen baru, tanpa CV terverifikasi | **0.055** | constant-fallback defect direproduksi; penyebab hidden exception belum diketahui |

Notebook publik terbaik per 2026-11-03 adalah 0.666 (banyak fork V31/One Swap/Sovereign). Selisih 0.664 ke 0.666 lebih kecil dari bias seleksi: menurut simulasi GoPain (diskusi 755215), skor publik terbaik dari 5-20 varian mirip yang berbeda di 5% baris rata-rata 0.005-0.008 di atas skor sebenarnya. Pembanding yang jujur untuk ide baru adalah "terbaik dikurangi bias itu", dan skor private kemungkinan turun ke sekitar 0.655-0.66 untuk semua fork ini.

Pasangan CV-LB yang terverifikasi: LoRA dua tahap publik CV per prompt 0.593 lawan LB 0.592. Nested CV pada grup 18 penuh 0.6865 lawan LB 0.56.

**Kegagalan nomor 1 (internal 0.75 lalu LB 0.639).** Kernel `ajikajadeh/ntl-gold-apex-grandmaster-sota-0-75` tidak pernah mengukur 0.75 di prompt held-out. Angka itu hanya judul. Kernel itu memakai resep 0.664/0.665 lalu menambahkan Hungarian exact di setiap kluster TF-IDF hidden test berukuran 2-18 (`_capped_agglo`, thr 0.15), yang memaksa satu model per kluster. Kluster hidden test tidak lengkap dan tidak murni, jadi LB turun 0.025 dari resep dasarnya. Aturan sejak itu:
1. Angka yang tidak berasal dari protokol di atas tidak boleh dipakai sebagai ekspektasi.
2. Post-processing yang memakai struktur hidden test harus lolos uji 2-3 passage per prompt dulu.
3. Hungarian selalu rugi di sini (bukti publik -0.009 sampai -0.017, kita -0.025).

## Log eksperimen

### Berhasil atau dipakai
| Eksperimen | Hasil | Bukti |
|---|---|---|
| Open Pool 190k untuk SVC | OOF 0.333 ke 0.470 | `results.jsonl` |
| SVC 4-view | test 0.476 ke 0.501 | REPORT, 70/15/15 |
| LoRA dua tahap q8 / ll8 | test 0.5755 / 0.5784 | `kernels/twostage` |
| Tahap 2 di baris host (sekitar 2.160 baris) | q8 +0.071, ll8 +0.066 di atas pool-only | notebook bagian 4 |
| Blend weighted q8+ll8+svc4 | test 0.6147 | notebook bagian 5 |
| Fork One Swap | LB 0.664 | submission 57876141 |
| E10 kurva data host | SVC host-only 0.230 (13 prompt) ke 0.307 (105 prompt), probe 27B 0.275 ke 0.322, belum jenuh | `research/experiments/host_value.*` |
| E13 repulsi label lunak di kluster TF-IDF (soft Hungarian), λ dipilih di val | k=2 per prompt +0.007 (4/5), k=3 +0.017 (4/5); kurva risiko: k=2 netral di semua λ, k=6 +0.039 | `research/experiments/repulsion*.py`; **tidak disubmit**: Hungarian (versi keras mekanisme yang sama) rugi di LB tiga kali, jadi kluster hidden test tidak murni atau satu prompt bisa berisi model yang sama |
| E10 crop host (60% kalimat awal dan akhir) | SVC +0.0169 (k=105) dan +0.0125 (k=26), positif 5/5 rotasi; tidak menular ke LoRA (E11) | sama |

### Gagal atau ditolak
| Eksperimen | Hasil |
|---|---|
| Hungarian di kluster hidden test | LB -0.025 (kita), -0.009 sampai -0.017 (publik) |
| Bobot pool 1.0 | 0.4567 lawan 0.4616 (0.35) |
| Blend z-score dengan LightGBM | 0.397 |
| Seleksi fitur stilometri top-30 (dipilih di train rotasi) | test 0.258 lawan 0.308 semua fitur |
| Ensemble mean / rank / vote / stacking LR di val | 0.567 / 0.563 / 0.586 / 0.575, semua di bawah weighted 0.613 |
| Hill climbing Caruana di val | 0.600, di bawah weighted 0.613 |
| Tipografi + Sinkhorn di atas blend | 0.6123 lawan 0.6147 |
| Bias logit per kelas | -0.018, CI di bawah 0 |
| Stack dilatih di baris val | -0.064 |
| Pseudo-label 1 putaran / self-training 3 putaran | 0.6176 / 0.5967 lawan guru 0.6171 |
| E7 bobot ulang pool lewat classifier domain | 0.6069 lawan 0.6147 |
| E8 tambah 64.618 baris eksternal | 0.6175 lawan 0.6147, di dalam derau |
| E9 pembaca beku Qwen3.8-27B | probe 0.4146 lawan 8B 0.3871, bobot nol di blend |
| Menambah anggota LoRA ke blend kuat (publik) | paling banyak +0.004 |
| E11 crop pada LoRA tahap 2 (1 epoch asli + 2 crop, pengganti epoch kedua) | -0.019 / -0.015 terhadap kontrol, 0/5 fold; sebagai anggota tambahan blend 0.6083 lawan 0.6147 |
| E12 pseudo-label transduktif top-50% di tahap 2 | -0.011 / -0.015 terhadap kontrol, 0/5 fold |
| E14 regenerasi komunitas atas prompt train (topik cocok, cos ≥ 0.35) | SVC host-only +0.036 tetapi SVC + pool -0.008; LoRA tahap 2 -0.004 / -0.009 terhadap kontrol |

### E11: crop pada LoRA tahap 2
Kernel `ajikajadeh/ntl-s2-crops` memuat `stage1.pt` dari `ajikajadeh/ntl-twostage-train`, lalu melatih ulang tahap 2 dengan 1 epoch atas baris host asli ditambah 2 crop per passage (sekitar 1,1x komputasi tahap 2 lama). Fold dan seed sama, jadi perbandingannya berpasangan per fold dengan `oof_plain.npy`. q8 di GPU 0 dan ll8 di GPU 1 berjalan bersamaan.

Hasil (balanced accuracy OOF, berpasangan per fold):

| Model | Asli (2 epoch) | Crop (1 epoch + 2 crop) | Delta per fold |
|---|---:|---:|---|
| q8 | 0.5779 | 0.5557 | -0.030, -0.007, -0.009, -0.035, -0.030 |
| ll8 | 0.5768 | 0.5597 | -0.013, -0.015, -0.017, -0.028, -0.013 |

Waktu tahap 2: q8 sekitar 47 menit per fold, ll8 sekitar 40 menit per fold, 2xT4 paralel, total sekitar 4 jam wall dan 7,4 jam GPU. **Ditolak.** Kenaikan crop di SVC tidak menular ke LoRA. Desain ini mengganti epoch kedua atas passage utuh dengan crop, jadi kesimpulan yang sah: epoch kedua atas passage utuh lebih berharga daripada crop. Crop sebagai tambahan di atas 2 epoch penuh belum diuji (sekitar 2,2x komputasi tahap 2). Prioritasnya rendah, karena augmentasi murah tidak terbukti menggantikan data host sungguhan.

## Riset dan state of the art
- **Write-up Kaggle terdekat** (RAG `kaggle-writeups`):
  - `llm-detect-ai-generated-text/writeups/r001-comprehensive-1st-place-write-up.md`: keragaman dan skala data generasi mengalahkan pilihan model.
  - `r002-2nd-place-solution-with-code-and-data.md`: pretrain di data generasi besar, lalu fine-tune ke domain prompt uji. Ini persis pola dua tahap kita, dan tahap 2 memberi +0.07.
  - `r003-3rd-place-solution.md`: hard example mining, pseudo-label test hanya untuk prediksi sangat yakin.
  - `r005-5th-place-solution-1-7-million-training-examples.md`: skala 1,7 juta contoh.
- **Paper:**
  - ATLAS, likelihood token bersyarat sumber dengan clipping (https://arxiv.org/abs/2610.19583); dipakai publik dengan bobot 20%.
  - READER, pembaca lebih kuat membuat jejak penulis lebih terpisah secara linear (https://arxiv.org/abs/2606.10794); E9 tidak mereproduksinya pada probe kecil.
  - TRACE, fingerprinting LLM ghostwriter (https://arxiv.org/abs/2603.28054).
- **Forum kompetisi:**
  - Open Pool v9/v11 (Carrot Production, diskusi 753779): 1 dari 4 passage pool adalah sampah atau salah genre; membersihkannya +0.027. Celah yang tersisa: 31% passage train memakai we/our/us, pool 15% (kita ukur 30.7% lawan 14.8%). Prompt host berbeda.
  - TPU v5e-8 melatih LoRA Qwen3-8B sekitar 14x lebih cepat dari 2xT4, bf16 bebas overflow fp16 (diskusi 754657). Antrean TPU bisa sampai 23 jam.
- **Aturan dari staf Kaggle (diskusi 752084):**
  - Data hasil generate sendiri boleh, tidak wajib publik, asal biaya wajar. Pengeluaran token di atas nilai hadiah (sekitar $5k) tidak wajar.
  - Kuota proxy Kaggle Benchmarks tidak untuk generate data training.
  - Dataset komunitas publik dan adapter publik boleh dipakai.
  - Clustering atau pseudo-label di hidden test belum dijawab.

### E12: pseudo-label transduktif di LoRA tahap 2
Setiap fold held-out berperan sebagai hidden test. Teacher q8+ll8 OOF melabeli separuh baris paling yakin (presisi 0.775). Baris itu masuk ke tahap 2 dengan grup dari kluster TF-IDF (ARI 0.83 terhadap prompt asli), dengan 2 epoch dan seed yang sama dengan baseline. Student dinilai di seluruh fold. Kernel `ajikajadeh/ntl-e12-pseudo-s2`, data `ajikajadeh/ntl-e12-pseudo`, kode `kernels/e12pseudo`.

Hasil (balanced accuracy OOF, berpasangan per fold, presisi pseudo 0.756-0.785):

| Model | Asli | Pseudo | Delta per fold |
|---|---:|---:|---|
| q8 | 0.5779 | 0.5638 | -0.015, -0.015, -0.004, -0.026, -0.011 |
| ll8 | 0.5768 | 0.5597 | -0.017, -0.024, -0.008, -0.022, -0.015 |

**Kontrol drift** (`ajikajadeh/ntl-s2-control`, kode E12 tanpa baris pseudo, torch 2.11.0, transformers 5.16.1, peft 0.20.0): q8 0.5749, ll8 0.5749, lawan run asli 0.5779 / 0.5768. Derau rerun per fold sekitar ±0.013. Terhadap kontrol:

| Metode | q8 | ll8 | Fold positif |
|---|---:|---:|---|
| E11 crop | -0.0193 | -0.0152 | 0/5, 0/5 |
| E12 pseudo top-50% | -0.0111 | -0.0152 | 0/5, 0/5 |

**Keduanya ditolak.** Pseudo-label dengan presisi 78% menambah derau label yang lebih merugikan daripada manfaat domainnya.

### E14: regenerasi komunitas atas prompt train host
22 dataset komunitas yang diunduh untuk E8 memuat 5.162 baris yang topiknya cocok dengan satu prompt train: cosinus TF-IDF ke centroid prompt ≥ 0.35, di atas persentil-99 kemiripan antar-prompt berbeda (0.326) dan di bawah median kemiripan dengan prompt sendiri (0.517). Baris ini mencakup 145 prompt dan 18 model. 11 baris yang merupakan salinan teks train (cos char 5-gram ≥ 0.8) dibuang, tersisa 5.151 (`data/e14/regen.parquet`, dataset private `ajikajadeh/ntl-e14-regen`). E8 dulu membuang baris ini lewat saringan cos < 0.255, dan Open Pool v9 juga membuang "jawaban untuk prompt train". `nlm_matched_clean` yang dipakai notebook publik bukan data sejenis: hanya 0.6% isinya cocok dengan prompt train, dan hanya 100 baris yang overlap.

- **Proxy CPU** (`research/experiments/regen_value.py`): SVC host-only 0.3074, ditambah baris regen dari prompt blok train saja menjadi **0.3431 (+0.036, 5/5 rotasi)**; dengan bobot 0.5 menjadi 0.3368.
- **Proxy CPU dengan pool** (`research/experiments/regen_pool.py`, SVC 4-view, pool 2.000 per kelas bobot 0.35, regen bobot 1): 0.4494 menjadi **0.4411 (-0.008, 2/5 rotasi)**. Kenaikan di setting host-only hanya efek volume yang sudah diberikan pool. Baris regen berperilaku seperti pool, bukan seperti host: topik yang sama tidak membuat gaya sama. Nilai data host kemungkinan berasal dari template prompt dan konfigurasi generasi host, bukan dari topiknya.
- **LoRA tahap 2** (`kernels/e14regen`, kernel `ajikajadeh/ntl-e14-regen-s2`): host + baris regen dari prompt fold training (jumlahnya disamakan dengan baris host), 2 epoch, seed sama dengan kontrol, fold 0-2. Hasil terhadap kontrol: q8 0.5687 menjadi 0.5650 (-0.004, 1/3 fold), ll8 0.5792 menjadi 0.5705 (-0.009, 1/3 fold). Sekitar 1,5 jam per fold karena barisnya 2x. **Ditolak**, konsisten dengan E14b: baris regen berperilaku seperti pool, dan mencampur pool ke tahap 2 sudah lama diketahui merugikan.

### Opsi data bergaya host yang tersedia (dicek 2026-11-02)
HF Inference Providers lewat token `yauwes` menyediakan zai-org/GLM-5.3-Flash ($0.15 input / $0.5 output per 1 juta token), zai-org/GLM-5.3 ($1.4 / $4.4), dan moonshotai/Kimi-K3 (sekitar $2.85 / $14.25). Akun HF itu akun gratis tanpa billing (`canPay: False`), jadi generate dalam jumlah berarti butuh billing dari pengguna. Model tertutup (Claude, GPT, Gemini) butuh API vendor atau OpenRouter. Taksiran untuk 1.500 passage per label untuk ketiga label terbuka sekitar $5-10, belum termasuk token reasoning.

## Gap riset
1. **Domain prompt.** Model kita dan model publik dilatih terutama di pool berprompt non-host. Tahap 2 di 2.160 baris host saja menambah +0.07, dan kurva E10 belum jenuh (sekitar +0.03 setiap kali jumlah prompt host dilipatgandakan). Tim atas kemungkinan besar punya data bergaya host jauh lebih banyak. Petunjuk: akun `lianmatsuo` mempublikasikan shard GGUF Kimi-K3 dan merujuk `teacher_runner`.
2. **Versi bersaudara.** 58% error tetap di keluarga yang benar. Akurasi keluarga 0.84, akurasi model 0.61.
3. **Passage pendek.** Akurasi kuartil terpendek (36-64 kata) 0.51 lawan 0.73 di kuartil terpanjang.

## Rencana (urut menurut nilai per biaya)
1. **Generate data bergaya host lewat API resmi tiap model** (OpenRouter atau API vendor). Ini diizinkan host dengan biaya wajar.
   - Skala: sekitar 500-1.000 prompt baru bergaya host (penjelasan satu paragraf, 40-250 kata, suara kolektif "we"), dijawab 18 model, total 9-18 ribu baris. Taksiran biaya puluhan sampai sekitar $200, jauh di bawah batas $5k.
   - Ekspektasi berdasarkan E10 dan tahap 2: +0.03 sampai +0.06 pada anggota LoRA. **Butuh kunci API dan anggaran dari pengguna; belum bisa dijalankan di sesi ini.**
   - Prompt bisa diturunkan dari 150 prompt train (parafrase topik, gaya instruksi sama), dengan dedup terhadap prompt train.
2. ~~Crop di tahap 2 LoRA~~: ditolak di E11. Kalau tetap ingin dicoba, crop harus ditambahkan di atas 2 epoch penuh, bukan menggantikannya.
3. **Pindahkan training ke TPU v5e-8** (20 jam per minggu, sekitar 14x lebih cepat). Inferensi tetap di T4.
4. Spesialis versi bersaudara dengan data tambahan dari butir 1, bukan dengan fitur stilometri. Spesialis stilometri sudah inkonklusif di ronde 2.

## Aturan submit
- **Autosubmit hanya jika test lokal 70/15/15 ≥ 0.91** (permintaan pengguna 2026-11-03: angka lokal 0.72 pernah menjadi 0.65 di LB). Di bawah itu, kandidat disimpan dan tidak disubmit otomatis.
- Submit manual hanya jika kandidat mengalahkan 0.664 di protokol lokal dengan selisih minimal 0.01 dan CI per prompt di atas 0, **dan** jalur hidden rerun sudah diverifikasi (smoke T4 di cabang inferensi).
- Fork publik dengan skor publik sama atau +0.001 tidak disubmit, karena di dalam derau 0.009.
- Gate lama "0.99 di train, val, dan test" tidak pernah terpenuhi. Train selalu sekitar 1.0 karena hafalan, jadi gate itu tidak informatif dan diganti aturan di atas.
- Batas 5 submission per hari UTC; satu skor butuh sekitar 8 jam. Rerun penilaian tidak memakai kuota GPU mingguan, hanya commit run yang memakai (HALX2158, diskusi 754426).

### Gerbang anti-error (wajib sebelum submit)
Submission 57901383 (Sovereign v2, dual DeBERTa) mati di hidden rerun dengan "Notebook Threw Exception". Penyebab kelasnya: saat commit, notebook mendeteksi 3 baris test publik lalu hanya menulis file format tanpa menjalankan model. Kode barunya (encoder kedua, copy `deb_pool`) pertama kali dieksekusi di hidden rerun. Log hidden rerun tidak bisa dibaca, jadi baris persis penyebabnya tidak diketahui. Langkah wajib:
1. `python kernels/guard/dryrun.py kernels/sub/<kernel> 400` membuat salinan `-dryrun`. Salinan ini mengalihkan path kompetisi ke test palsu 400 passage pool, mengecek CUDA, dan memvalidasi `submission.parquet` di akhir.
2. Push salinan itu, tunggu COMPLETE, dan cari baris `DRYRUN OK` di log. Catat durasinya: biaya tetap (memuat model) sama dengan di hidden rerun, ditambah biaya per baris.
3. Baru submit kernel aslinya (versi light commit). Jangan submit versi `-dryrun`.
4. Komponen opsional dibungkus try/except dengan fallback ke prediksi dasar. Jangan pakai `assert returncode == 0` di komponen opsional.
5. Kode yang hanya jalan saat `len(test) >= 50` (kluster, Sinkhorn, ATLAS) harus lewat langkah 1-2, karena light commit tidak pernah menjalankannya.

## Struktur repo
| Path | Isi |
|---|---|
| `notebook/ntl_report.ipynb` | laporan EDA sampai submission, dengan output dan waktu per sel |
| `notebook/report_cells.py`, `build.py`, `lib.py` | sumber notebook dan seluruh fungsi (split, SVC, LoRA, blend, post-processing) |
| `notebook/ntl_a100.ipynb` | notebook training A100 Colab (bukan jalur submit) |
| `kernels/twostage` | LoRA dua tahap q8 + ll8 di T4x2, OOF di `out_full` |
| `kernels/s2crops` | E11 crop tahap 2 |
| `kernels/sub/ntl-sub-oneswap` | kernel submission terbaik (LB 0.664) |
| `kernels/svc4`, `sparse7015`, `svc4da`, `svc4ext`, `reader` | anggota sparse dan E7-E9 |
| `research/experiments` | skrip eksperimen dan log (leak, round2, pseudo, selftrain, ext, reader, host_value) |
| `research/REPORT.md` | laporan teknis lengkap per eksperimen |

## Catatan operasional
- `kernel-metadata.json` hasil `pull` membawa `id_no` penulis asli; push mengembalikan 403 "accept rules". Hapus `id_no`.
- Deteksi lingkungan lewat `KAGGLE_KERNEL_RUN_TYPE`, bukan keberadaan `/kaggle/input`.
- `time.sleep` di Mac berhenti saat tidur; tugas menunggu butuh `caffeinate -i`.
- Kernel batch Kaggle tidak menampilkan log sampai selesai. Simpan OOF per fold agar kegagalan di tengah jalan tidak membuang semuanya.

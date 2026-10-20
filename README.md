# Studi Kasus Informatika: 6 Materi Pemateri

Repositori ini berisi satu studi kasus berbasis kode untuk setiap materi pemateri. Semua dapat dijalankan
dengan Python, memakai data sintetis atau dataset bawaan scikit-learn (tanpa unduhan).

| # | Materi | Studi kasus | Folder |
|---|---|---|---|
| 1 | AI untuk Pemantauan Deforestasi Hutan Tropis | Fusi optik+radar untuk klasifikasi penyebab deforestasi saat berawan | `01_deforestasi_ai` |
| 2 | HCI dan Visualisasi Data | Chartjunk vs desain bersih, Gestalt, evaluasi heuristik Nielsen | `02_imk_visualisasi` |
| 3 | Introduction to Machine Learning | Pipeline ML lengkap, overfitting, clustering, regresi | `03_machine_learning` |
| 4 | Make Sense of Data with Analytics and AI | CRISP-DM prediksi churn: jebakan join, imbalance, leakage | `04_analitik_crispdm` |
| 5 | HPC untuk Tantangan Lokal | Ensemble simulasi banjir paralel, speedup, Amdahl | `05_hpc_paralel` |
| 6 | Karier dan Kompetisi dengan GIS | Aksesibilitas fasilitas kesehatan, spatial join, GeoJSON | `06_gis_spasial` |

## Cara menjalankan
```bash
pip install -r requirements.txt
python run_all.py            # semua studi kasus (sekitar 30 detik)
python run_all.py --fast     # versi cepat
# atau satu per satu, misalnya:
cd 03_machine_learning && python ml_pipeline.py
```
Hasil (grafik, CSV, GeoJSON) ada di folder `output/` tiap studi kasus.

## Batasan yang jujur
- Data pada kasus 1, 4, 5, 6 sintetis, jadi angkanya untuk belajar konsep, bukan klaim performa dunia nyata.
- Kasus 1 memakai ekstraktor fitur konvolusi tetap + Random Forest, bukan CNN yang dilatih penuh.
- `05_hpc_paralel/mpi_ensemble.py` dan `job_ensemble.slurm` adalah templat klaster yang belum diuji.

## Upload ke GitHub
Buat repositori kosong di github.com (tanpa README/.gitignore), lalu di folder ini:
```bash
git init
git add .
git commit -m "Studi kasus 6 materi pemateri"
git branch -M main
git remote add origin https://github.com/USERNAME/NAMA-REPO.git
git push -u origin main
```
Ganti `USERNAME` dan `NAMA-REPO`. Jika diminta password, gunakan Personal Access Token GitHub.
Tambahkan nama dan NIM Anda di bagian atas README ini sebelum dikumpulkan.

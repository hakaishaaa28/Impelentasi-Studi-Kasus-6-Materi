# Studi Kasus Informatika: 6 Materi Pemateri

**Nama:** Kevin Stevino Arnovan

**NPM:** G1A025121

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

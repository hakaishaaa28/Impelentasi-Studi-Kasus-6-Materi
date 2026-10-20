"""
Studi Kasus 4 - Make Sense of Data with Analytics and AI (CRISP-DM)
===================================================================
Skenario: operator layanan berlangganan ingin mengurangi pelanggan berhenti (churn).

Mengikuti CRISP-DM dan memperagakan jebakan yang disebut di materi:
  - Business Understanding   -> pertanyaan analitik
  - Data Understanding/Prep  -> data berantakan, JOIN yang menggandakan baris
  - Modeling/Evaluation      -> data tidak seimbang, DATA LEAKAGE
  - Deployment               -> 4 tingkat analitik (deskriptif..preskriptif)

Jalankan:  python crispdm_churn.py
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def judul(t):
    print(f"\n{'=' * 8} {t} {'=' * 8}")


def buat_data(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    cust = pd.DataFrame({
        "id": np.arange(n),
        "tenure_bulan": rng.integers(1, 72, n),
        "tagihan": rng.normal(180, 60, n).clip(40, 400).round(0),
        "tiket_keluhan": rng.poisson(1.2, n),
        "kontrak": rng.choice(["bulanan", "tahunan"], n, p=[0.7, 0.3]),
    })
    logit = (-2.2 + 0.9 * (cust.kontrak == "bulanan") + 0.45 * cust.tiket_keluhan
             - 0.025 * cust.tenure_bulan + 0.006 * (cust.tagihan - 180))
    cust["churn"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    # sebagian tagihan hilang (data berantakan)
    cust.loc[rng.choice(n, 150, replace=False), "tagihan"] = np.nan
    # tabel kedua: riwayat ganti paket, 1 pelanggan bisa punya banyak baris
    riwayat = pd.DataFrame({"id": rng.choice(n, 1500), "ganti_paket": 1})
    return cust, riwayat


def siapkan(cust):
    X = cust.copy()
    X["tagihan"] = X["tagihan"].fillna(X["tagihan"].median())
    X["kontrak_bulanan"] = (X["kontrak"] == "bulanan").astype(int)
    return X.drop(columns=["kontrak"])


def main():
    argparse.ArgumentParser().parse_known_args()
    os.makedirs(OUT, exist_ok=True)

    judul("1. BUSINESS UNDERSTANDING")
    print("Masalah bisnis : biaya kehilangan pelanggan tinggi.")
    print("Pertanyaan analitik: pelanggan mana yang kemungkinan churn bulan depan, "
          "dan siapa yang layak dihubungi tim retensi?")

    judul("2. DATA UNDERSTANDING")
    cust, riwayat = buat_data()
    print(f"Pelanggan: {len(cust)} baris | missing tagihan: {cust.tagihan.isna().sum()} | "
          f"tingkat churn: {cust.churn.mean():.1%}  (tidak seimbang)")

    judul("3. DATA PREPARATION - jebakan JOIN")
    salah = cust.merge(riwayat, on="id", how="left")
    print(f"Join naif   : {len(cust)} -> {len(salah)} baris (pelanggan terduplikasi, statistik jadi bias)")
    agg = riwayat.groupby("id", as_index=False)["ganti_paket"].count().rename(columns={"ganti_paket": "jml_ganti_paket"})
    benar = cust.merge(agg, on="id", how="left", validate="one_to_one").fillna({"jml_ganti_paket": 0})
    print(f"Join benar  : agregasi dulu + validate='one_to_one' -> {len(benar)} baris (sama dengan jumlah pelanggan)")
    assert len(benar) == len(cust)
    data = siapkan(benar)

    judul("4. MODELING + EVALUATION")
    fitur = ["tenure_bulan", "tagihan", "tiket_keluhan", "kontrak_bulanan", "jml_ganti_paket"]
    X_tr, X_te, y_tr, y_te = train_test_split(data[fitur], data.churn, test_size=0.3, stratify=data.churn, random_state=0)

    dummy = np.zeros(len(y_te))
    print(f"Jebakan akurasi: model 'tebak tidak ada yang churn' akurasi={1 - y_te.mean():.3f} tapi recall=0")

    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", random_state=0, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    p = rf.predict_proba(X_te)[:, 1]
    auc_jujur = roc_auc_score(y_te, p)
    rec = recall_score(y_te, (p > 0.4).astype(int))
    print(f"Model jujur    : AUC={auc_jujur:.3f}, recall(churn) pada threshold 0.4 = {rec:.3f}")

    # DATA LEAKAGE: fitur yang baru ada SETELAH pelanggan memutuskan berhenti
    data_bocor = data.copy()
    rng = np.random.default_rng(1)
    data_bocor["tawaran_retensi_ditolak"] = np.where(
        data_bocor.churn == 1, (rng.random(len(data_bocor)) < 0.95).astype(int),
        (rng.random(len(data_bocor)) < 0.02).astype(int))
    fb = fitur + ["tawaran_retensi_ditolak"]
    Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(data_bocor[fb], data_bocor.churn, test_size=0.3, stratify=data_bocor.churn, random_state=0)
    rfb = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", random_state=0, n_jobs=-1).fit(Xb_tr, yb_tr)
    auc_bocor = roc_auc_score(yb_te, rfb.predict_proba(Xb_te)[:, 1])
    print(f"Model BOCOR    : AUC={auc_bocor:.3f}  <- 'terlalu bagus'; fitur 'tawaran_retensi_ditolak' hanya ada setelah churn.")
    print("                 Di produksi fitur ini kosong, model akan gagal. Garbage in, garbage out.")

    judul("5. DEPLOYMENT - empat tingkat analitik")
    hasil = X_te.copy()
    # probabilitas model 'balanced' menggelembung; kalibrasi dulu sebelum dipakai menghitung nilai harapan
    kal = CalibratedClassifierCV(RandomForestClassifier(n_estimators=300, min_samples_leaf=10, random_state=0, n_jobs=-1),
                                 method="isotonic", cv=3).fit(X_tr, y_tr)
    hasil["p_churn"] = kal.predict_proba(X_te)[:, 1]
    hasil["churn_aktual"] = y_te.values
    print("Deskriptif : tingkat churn per kontrak ->")
    print(data.groupby("kontrak_bulanan").churn.mean().rename({0: "tahunan", 1: "bulanan"}).round(3).to_string())
    print("Diagnostik : rata-rata fitur churn vs tidak ->")
    print(data.groupby("churn")[["tenure_bulan", "tiket_keluhan"]].mean().round(2).to_string())
    print(f"Prediktif  : model memberi skor risiko untuk {len(hasil)} pelanggan uji (AUC {auc_jujur:.2f})")

    # Preskriptif: hubungi hanya jika nilai harapan penyelamatan > biaya kontak
    nilai_pelanggan, biaya_kontak, peluang_berhasil = 2_000_000, 150_000, 0.25
    hasil["nilai_harapan"] = hasil.p_churn * peluang_berhasil * nilai_pelanggan - biaya_kontak
    target = hasil[hasil.nilai_harapan > 0].sort_values("nilai_harapan", ascending=False)
    print(f"Preskriptif: hubungi {len(target)} dari {len(hasil)} pelanggan (nilai harapan > biaya kontak)")
    print(f"             estimasi churn tertangkap pada kelompok ini: {int(target.churn_aktual.sum())} "
          f"dari {int(hasil.churn_aktual.sum())} churn aktual")
    target.head(20).to_csv(os.path.join(OUT, "daftar_kontak_retensi_top20.csv"))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Model jujur", "Model bocor"], [auc_jujur, auc_bocor], color=["#2c7fb8", "#c0392b"])
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("AUC")
    ax.set_title("AUC melonjak = curigai data leakage")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "leakage_auc.png"), dpi=140)
    print(f"\nKeluaran disimpan di: {OUT}")
    print("Keputusan akhir (siapa dihubungi, berapa anggaran) tetap perlu verifikasi dan penilaian manusia.")


if __name__ == "__main__":
    main()

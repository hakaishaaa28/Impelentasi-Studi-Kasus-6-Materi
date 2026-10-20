"""
Studi Kasus 3 - Introduction to Machine Learning
================================================
Mencakup alur kerja ML dari materi:
  1. Data mentah -> pra-pemrosesan (missing value, duplikasi, normalisasi)
  2. Split latih/uji -> pilih algoritma -> latih -> evaluasi (akurasi, presisi, recall)
  3. Overfitting vs underfitting (kurva kompleksitas)
  4. Unsupervised: clustering (KMeans) + reduksi dimensi (PCA)
  5. Regresi dengan Mean Absolute Error

Dataset bawaan scikit-learn (tanpa unduhan): breast_cancer, iris, diabetes.

Jalankan:  python ml_pipeline.py
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, adjusted_rand_score, mean_absolute_error,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def supervised_pipeline():
    print("=== 1) SUPERVISED: klasifikasi kanker payudara ===")
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    # simulasikan data kotor: missing value + duplikasi
    rng = np.random.default_rng(0)
    for col in ["mean radius", "mean texture"]:
        df.loc[rng.choice(df.index, 25, replace=False), col] = np.nan
    df = pd.concat([df, df.sample(15, random_state=1)])
    print(f"Data mentah: {df.shape}, missing={int(df.isna().sum().sum())}, duplikat={int(df.duplicated().sum())}")
    df = df.drop_duplicates()
    X, y = df.drop(columns="target"), df["target"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=0),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=0),
    }
    rows = []
    for name, m in models.items():
        # imputer & scaler di dalam pipeline -> hanya belajar dari data latih (cegah leakage)
        pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), m)
        pipe.fit(X_tr, y_tr)
        p = pipe.predict(X_te)
        rows.append({"model": name, "akurasi": accuracy_score(y_te, p),
                     "presisi": precision_score(y_te, p), "recall": recall_score(y_te, p)})
    res = pd.DataFrame(rows).round(3)
    print(res.to_string(index=False))
    res.to_csv(os.path.join(OUT, "hasil_supervised.csv"), index=False)


def overfitting_demo():
    print("\n=== 2) OVERFITTING vs UNDERFITTING (kedalaman pohon) ===")
    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=3)
    # tambah label noise agar overfitting terlihat jelas
    rng = np.random.default_rng(0)
    flip = rng.random(len(y_tr)) < 0.15
    y_noisy = np.where(flip, 1 - y_tr, y_tr)
    depths = range(1, 16)
    tr, te = [], []
    for d in depths:
        t = DecisionTreeClassifier(max_depth=d, random_state=0).fit(X_tr, y_noisy)
        tr.append(t.score(X_tr, y_noisy))
        te.append(t.score(X_te, y_te))
    best = list(depths)[int(np.argmax(te))]
    print(f"Kedalaman terbaik pada data uji: {best} (akurasi uji {max(te):.3f}); "
          f"kedalaman 15: latih {tr[-1]:.3f} vs uji {te[-1]:.3f}")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(depths, tr, label="latih", marker="o")
    ax.plot(depths, te, label="uji", marker="s")
    ax.axvline(best, ls="--", color="gray")
    ax.set_xlabel("Kedalaman pohon (kompleksitas)")
    ax.set_ylabel("Akurasi")
    ax.set_title("Makin kompleks: akurasi latih naik, akurasi uji tidak ikut naik")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "overfitting.png"), dpi=140)
    plt.close(fig)


def unsupervised_demo():
    print("\n=== 3) UNSUPERVISED: KMeans + PCA pada Iris (label tidak dipakai saat latih) ===")
    X, y = load_iris(return_X_y=True)
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(Xs)
    ari = adjusted_rand_score(y, km.labels_)
    print(f"Adjusted Rand Index vs spesies sebenarnya: {ari:.3f}")
    Z = PCA(n_components=2).fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(*Z.T, c=km.labels_, cmap="Set2", s=35)
    ax.set_title("Cluster KMeans pada ruang PCA 2D")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "clustering_pca.png"), dpi=140)
    plt.close(fig)


def regression_demo():
    print("\n=== 4) REGRESI: progres diabetes, metrik MAE ===")
    X, y = load_diabetes(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
    base = np.full_like(y_te, y_tr.mean(), dtype=float)
    lr = LinearRegression().fit(X_tr, y_tr)
    print(f"MAE baseline (tebak rata-rata): {mean_absolute_error(y_te, base):.1f}")
    print(f"MAE regresi linear            : {mean_absolute_error(y_te, lr.predict(X_te)):.1f}")


def main():
    argparse.ArgumentParser().parse_known_args()
    os.makedirs(OUT, exist_ok=True)
    supervised_pipeline()
    overfitting_demo()
    unsupervised_demo()
    regression_demo()
    print(f"\nKeluaran disimpan di: {OUT}")
    print("Etika: periksa bias data & transparansi model sebelum dipakai pada keputusan kesehatan nyata.")


if __name__ == "__main__":
    main()

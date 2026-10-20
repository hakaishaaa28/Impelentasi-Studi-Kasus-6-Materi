"""
Studi Kasus 1 - AI untuk Pemantauan Deforestasi Hutan Tropis
=============================================================
Materi: fusi citra optik Sentinel-2 (terhalang awan) + radar Sentinel-1
(menembus awan) untuk mengklasifikasikan PENYEBAB kehilangan tutupan hutan.

Karena data satelit asli berukuran besar, studi kasus ini memakai patch
sintetis 16x16 piksel dengan 4 kanal:
    0 = NDVI (optik, S2)   1 = NIR (optik, S2)
    2 = VV   (radar, S1)   3 = VH  (radar, S1)

Pipeline (gaya CNN): konvolusi filter tetap -> ReLU -> pooling -> classifier.
Filter tidak dilatih (random-feature / fixed-filter), hanya "kepala"
classifier yang dilatih. Untuk CNN sungguhan lihat bagian Pengembangan di README.

Eksperimen utama: apakah fusi optik+radar lebih tangguh saat berawan?

Jalankan:  python deforestasi.py [--fast]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split

CLASSES = ["alami", "pertanian", "jalan", "penebangan", "tambang", "kebakaran"]
S = 16
FOREST = np.array([0.80, 0.70, 0.50, 0.35])  # NDVI, NIR, VV, VH hutan sehat
TARGETS = {
    "pertanian": np.array([0.45, 0.50, 0.30, 0.20]),
    "jalan": np.array([0.15, 0.30, 0.65, 0.20]),
    "penebangan": np.array([0.35, 0.45, 0.40, 0.25]),
    "tambang": np.array([0.05, 0.20, 0.85, 0.60]),
    "kebakaran": np.array([0.15, 0.10, 0.45, 0.20]),
}
SENSORS = {"optik": [0, 1], "radar": [2, 3], "fusi": [0, 1, 2, 3]}


def smooth_noise(rng, k=3):
    n = rng.random((S, S))
    return convolve2d(n, np.ones((k, k)) / k**2, mode="same", boundary="symm")


def make_mask(cls, rng):
    yy, xx = np.mgrid[0:S, 0:S]
    m = np.zeros((S, S))
    if cls == "pertanian":  # blok persegi panjang rapi
        h, w = rng.integers(6, 11, 2)
        y0, x0 = rng.integers(0, S - h), rng.integers(0, S - w)
        m[y0:y0 + h, x0:x0 + w] = 1
    elif cls == "jalan":  # garis tipis dengan sudut acak
        th = rng.uniform(0, np.pi)
        cx, cy = rng.uniform(5, 11, 2)
        d = np.abs((xx - cx) * np.sin(th) - (yy - cy) * np.cos(th))
        m = (d < rng.uniform(0.8, 1.4)).astype(float)
    elif cls == "penebangan":  # titik-titik kecil tersebar (selektif)
        for _ in range(rng.integers(6, 12)):
            y, x = rng.integers(0, S - 1, 2)
            m[y:y + 2, x:x + 2] = 1
    elif cls == "tambang":  # blob tak beraturan, satu gumpalan besar
        r = rng.uniform(3, 5.5)
        cx, cy = rng.uniform(5, 11, 2)
        m = (((xx - cx) ** 2 + (yy - cy) ** 2) < r**2 + 6 * (smooth_noise(rng) - 0.5) * r).astype(float)
    elif cls == "kebakaran":  # beberapa bercak luas menyebar
        m = (smooth_noise(rng, 5) > 0.55).astype(float)
    else:  # alami: celah kecil lemah (pohon tumbang)
        m = (smooth_noise(rng, 5) > 0.62).astype(float) * 0.3
    return m


def make_patch(cls, rng, cloudy):
    x = np.zeros((4, S, S))
    m = make_mask(cls, rng) if cls != "alami" else make_mask("alami", rng)
    tgt = TARGETS.get(cls, FOREST)
    for c in range(4):
        sigma = 0.05 if c < 2 else 0.09  # radar lebih berbintik (speckle)
        x[c] = FOREST[c] * (1 - m) + tgt[c] * m + rng.normal(0, sigma, (S, S))
    if cloudy:  # awan menghapus informasi optik, radar tetap utuh
        x[0:2] = rng.normal(0.6, 0.2, (2, S, S))
    return x


def gen_dataset(n_per_class, cloud_frac, seed):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for i, cls in enumerate(CLASSES):
        for _ in range(n_per_class):
            X.append(make_patch(cls, rng, rng.random() < cloud_frac))
            y.append(i)
    return np.array(X), np.array(y)


# ---- ekstraktor fitur gaya CNN: konvolusi -> ReLU -> global pooling ----
FILTERS = [
    np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]),   # Sobel-x
    np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]),   # Sobel-y
    np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),     # Laplacian
    np.ones((3, 3)) / 9,                              # blur
]


def extract_features(X, channels):
    feats = []
    for patch in X:
        f = []
        for c in channels:
            ch = patch[c]
            f += [ch.mean(), ch.std(), ch.min(), ch.max()]
            f += [ch[:8, :8].mean(), ch[:8, 8:].mean(), ch[8:, :8].mean(), ch[8:, 8:].mean()]
            for k in FILTERS:
                r = np.maximum(convolve2d(ch, k, mode="valid"), 0)  # ReLU
                f += [r.mean(), r.max()]
        feats.append(f)
    return np.array(feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="dataset lebih kecil")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    n = 120 if a.fast else 350

    X_tr, y_tr = gen_dataset(n, cloud_frac=0.3, seed=1)
    tests = {"cerah (0% awan)": gen_dataset(n // 2, 0.0, 2), "berawan (100% awan)": gen_dataset(n // 2, 1.0, 3)}

    print("Akurasi klasifikasi penyebab deforestasi (6 kelas)")
    print(f"{'sensor':<8}" + "".join(f"{k:>24}" for k in tests))
    results, best_model = {}, None
    for name, ch in SENSORS.items():
        clf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
        clf.fit(extract_features(X_tr, ch), y_tr)
        row = {}
        for tname, (Xt, yt) in tests.items():
            pred = clf.predict(extract_features(Xt, ch))
            row[tname] = accuracy_score(yt, pred)
            if name == "fusi" and tname.startswith("berawan"):
                best_model = (yt, pred)
        results[name] = row
        print(f"{name:<8}" + "".join(f"{row[k]:>24.3f}" for k in tests))

    # grafik perbandingan sensor
    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.35
    for i, tname in enumerate(tests):
        ax.bar(np.arange(3) + i * w, [results[s][tname] for s in SENSORS], w, label=tname)
    ax.set_xticks(np.arange(3) + w / 2)
    ax.set_xticklabels(SENSORS.keys())
    ax.set_ylabel("Akurasi")
    ax.set_ylim(0, 1)
    ax.set_title("Optik vs Radar vs Fusi saat cerah dan berawan")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "perbandingan_sensor.png"), dpi=140)

    # confusion matrix fusi saat berawan
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(*best_model, display_labels=CLASSES, ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title("Confusion matrix - fusi, kondisi berawan")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "confusion_fusi_berawan.png"), dpi=140)

    # contoh patch tiap kelas (NDVI)
    rng = np.random.default_rng(7)
    fig, axs = plt.subplots(2, 6, figsize=(12, 4.2))
    for i, cls in enumerate(CLASSES):
        p = make_patch(cls, rng, False)
        axs[0, i].imshow(p[0], cmap="YlGn", vmin=0, vmax=1)
        axs[0, i].set_title(cls)
        axs[1, i].imshow(p[2], cmap="gray", vmin=0, vmax=1)
        for r in (0, 1):
            axs[r, i].axis("off")
    axs[0, 0].text(-4, 8, "NDVI", rotation=90, va="center")
    axs[1, 0].text(-4, 8, "Radar VV", rotation=90, va="center")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "contoh_patch.png"), dpi=140)
    print(f"\nGrafik disimpan di: {a.out}")
    print("Catatan: data sintetis; hasil nyata butuh validasi lapangan dan peninjauan manusia.")


if __name__ == "__main__":
    main()

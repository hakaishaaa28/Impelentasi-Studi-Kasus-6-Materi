"""
Studi Kasus 6 - Karier dan Kompetisi di Era Industri dengan GIS
===============================================================
Mahasiswa informatika bisa menjadi PENGGUNA sekaligus PENGEMBANG alat analitik spasial.
Studi kasus portofolio: analisis aksesibilitas fasilitas kesehatan di sebuah kota fiktif
(manajemen aset kota pintar / smart city).

Diimplementasikan murni dengan NumPy (tanpa GeoPandas) agar mudah dijalankan; keluarannya
GeoJSON yang otomatis ter-render sebagai peta di GitHub.

Analisis spasial:
  1. Jarak haversine rumah tangga -> fasilitas terdekat
  2. Cakupan layanan (buffer 2 km)
  3. Spatial join point-in-polygon: rumah tangga di zona rawan banjir (ray casting)
  4. Grid kepadatan + prioritas lokasi fasilitas baru (maximal coverage sederhana)

Jalankan:  python gis_analisis.py
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
R_BUMI = 6371.0  # km
PUSAT = (-2.50, 112.00)  # lat, lon kota fiktif
RADIUS_LAYANAN_KM = 2.0


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R_BUMI * np.arcsin(np.sqrt(a))


def titik_dalam_poligon(lons, lats, poly):
    """Ray casting vectorized. poly: array (m,2) berisi (lon,lat)."""
    inside = np.zeros(len(lons), bool)
    x1, y1 = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    for xa, ya, xb, yb in zip(x1, y1, x2, y2):
        cond = (ya > lats) != (yb > lats)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (xb - xa) * (lats - ya) / (yb - ya) + xa
        inside ^= cond & (lons < xint)
    return inside


def buat_data(n_rumah=3000, seed=4):
    rng = np.random.default_rng(seed)
    # rumah tangga: dua klaster padat + sebaran
    pusat_klaster = np.array([[-2.48, 111.98], [-2.53, 112.04], [-2.50, 112.00]])
    idx = rng.choice(3, n_rumah, p=[0.4, 0.35, 0.25])
    lat = pusat_klaster[idx, 0] + rng.normal(0, 0.018, n_rumah)
    lon = pusat_klaster[idx, 1] + rng.normal(0, 0.020, n_rumah)
    fasilitas = np.array([[-2.485, 111.985], [-2.52, 112.03], [-2.47, 112.03]])  # 3 puskesmas
    zona_banjir = np.array([[111.97, -2.46], [112.02, -2.47], [112.03, -2.50], [111.99, -2.52], [111.96, -2.49]])  # (lon,lat)
    return lat, lon, fasilitas, zona_banjir


def main():
    argparse.ArgumentParser().parse_known_args()
    os.makedirs(OUT, exist_ok=True)
    lat, lon, fas, zona = buat_data()
    n = len(lat)

    # 1) jarak ke fasilitas terdekat
    d = np.stack([haversine_km(lat, lon, f[0], f[1]) for f in fas], axis=1)
    terdekat, jarak = d.argmin(1), d.min(1)
    print(f"1) Jarak median ke fasilitas terdekat: {np.median(jarak):.2f} km | P90: {np.percentile(jarak, 90):.2f} km")

    # 2) cakupan layanan
    tercakup = jarak <= RADIUS_LAYANAN_KM
    print(f"2) Rumah tangga dalam radius {RADIUS_LAYANAN_KM:.0f} km: {tercakup.sum()} dari {n} ({tercakup.mean():.1%})")

    # 3) spatial join zona banjir
    banjir = titik_dalam_poligon(lon, lat, zona)
    rawan_tak_tercakup = banjir & ~tercakup
    print(f"3) Rumah tangga di zona rawan banjir: {banjir.sum()} | di antaranya di luar cakupan fasilitas: {rawan_tak_tercakup.sum()}")

    # 4) kandidat lokasi baru: pusat sel grid dengan rumah tak tercakup terbanyak (greedy max coverage)
    tak = ~tercakup
    lat_k = np.linspace(lat.min(), lat.max(), 25)
    lon_k = np.linspace(lon.min(), lon.max(), 25)
    best, best_gain = None, -1
    for la in lat_k:
        for lo in lon_k:
            gain = int(((haversine_km(lat[tak], lon[tak], la, lo) <= RADIUS_LAYANAN_KM)).sum())
            if gain > best_gain:
                best, best_gain = (la, lo), gain
    cakupan_baru = (tercakup.sum() + best_gain) / n
    print(f"4) Lokasi fasilitas baru usulan: lat={best[0]:.4f}, lon={best[1]:.4f} -> +{best_gain} rumah, cakupan jadi {cakupan_baru:.1%}")

    # ---- peta statis ----
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.fill(zona[:, 0], zona[:, 1], color="#3498db", alpha=0.2, label="Zona rawan banjir")
    ax.scatter(lon[tercakup], lat[tercakup], s=4, c="#7f8c8d", label="Tercakup")
    ax.scatter(lon[tak], lat[tak], s=5, c="#e67e22", label="Tidak tercakup")
    ax.scatter(lon[rawan_tak_tercakup], lat[rawan_tak_tercakup], s=7, c="#c0392b", label="Rawan banjir & tidak tercakup")
    ax.scatter(fas[:, 1], fas[:, 0], marker="P", s=160, c="#27ae60", edgecolor="k", label="Fasilitas ada")
    ax.scatter(best[1], best[0], marker="*", s=300, c="gold", edgecolor="k", label="Usulan fasilitas baru")
    ax.set_xlabel("Bujur")
    ax.set_ylabel("Lintang")
    ax.set_title("Aksesibilitas fasilitas kesehatan (kota fiktif)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_aspect(1 / np.cos(np.radians(PUSAT[0])))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "peta_aksesibilitas.png"), dpi=140)
    plt.close(fig)

    # ---- GeoJSON (render otomatis di GitHub) ----
    fitur = [{"type": "Feature", "properties": {"nama": "Zona rawan banjir"},
              "geometry": {"type": "Polygon", "coordinates": [zona.tolist() + [zona[0].tolist()]]}}]
    for i, f in enumerate(fas):
        fitur.append({"type": "Feature", "properties": {"nama": f"Fasilitas {i + 1}", "jenis": "ada"},
                      "geometry": {"type": "Point", "coordinates": [float(f[1]), float(f[0])]}})
    fitur.append({"type": "Feature", "properties": {"nama": "Usulan fasilitas baru", "jenis": "usulan"},
                  "geometry": {"type": "Point", "coordinates": [float(best[1]), float(best[0])]}})
    contoh = np.flatnonzero(rawan_tak_tercakup)[:200]
    for i in contoh:
        fitur.append({"type": "Feature", "properties": {"jenis": "rawan_banjir_tak_tercakup", "jarak_km": round(float(jarak[i]), 2)},
                      "geometry": {"type": "Point", "coordinates": [float(lon[i]), float(lat[i])]}})
    with open(os.path.join(OUT, "hasil.geojson"), "w", encoding="utf-8") as fh:
        json.dump({"type": "FeatureCollection", "features": fitur}, fh)
    print(f"\nPeta PNG dan hasil.geojson disimpan di: {OUT}")
    print("Data sintetis. Portofolio nyata: pakai data terbuka (OpenStreetMap, BIG, BPS) dan dokumentasikan metodenya.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
PNG auf n Farben quantisieren (K-Means), Palette plotten,
Vorher/Nachher anzeigen und Ergebnis speichern.

Nutzung:
    python quantize_png_kmeans.py --input pfad/zum/bild.png --n_colors 8 --output output.png
"""

import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def load_image_rgba(path):
    img = Image.open(path).convert("RGBA")
    return img

def image_to_arrays(img_rgba):
    """Gibt (rgb_array(H,W,3), alpha(H,W)) zurück, beide uint8."""
    arr = np.array(img_rgba, dtype=np.uint8)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]
    return rgb, alpha

def kmeans_quantize(rgb, n_colors=8, sample_max=200_000, random_state=42):
    """
    rgb: (H,W,3) uint8
    -> quant_rgb (H,W,3) uint8, palette (n,3) uint8, labels (H*W,) int
    """
    h, w, _ = rgb.shape
    pixels = rgb.reshape(-1, 3).astype(np.float32)

    # Optional: auf sichtbare Pixel beschränken kann man, wenn Alpha vorliegt.
    # Hier nehmen wir alle RGB-Pixel.

    # Für sehr große Bilder: nur Stichprobe zum Fitten
    if pixels.shape[0] > sample_max:
        idx = np.random.RandomState(random_state).choice(pixels.shape[0], sample_max, replace=False)
        sample = pixels[idx]
    else:
        sample = pixels

    km = KMeans(n_clusters=n_colors, n_init="auto", random_state=random_state)
    km.fit(sample)

    # Zugehörigkeit aller Pixel bestimmen
    labels = km.predict(pixels)
    palette = np.clip(km.cluster_centers_.round(), 0, 255).astype(np.uint8)

    quant_pixels = palette[labels].reshape(h, w, 3)
    return quant_pixels, palette, labels

def save_rgba(rgb, alpha, path):
    rgba = np.dstack([rgb, alpha]).astype(np.uint8)
    out = Image.fromarray(rgba, mode="RGBA")
    out.save(path)

def plot_palette(palette, title="Palette (K-Means)"):
    """
    Zeigt die Farben als große quadratische Marker.
    """
    fig, ax = plt.subplots(figsize=(max(6, len(palette) * 0.7), 2.5))
    x = np.arange(len(palette))
    for i, c in enumerate(palette):
        # Farbe in [0,1] normalisieren
        ax.scatter([i], [0], s=3000, marker='s', c=[c / 255.0], edgecolors='black', linewidths=0.5)
        ax.text(i, -0.4, f"#{c[0]:02X}{c[1]:02X}{c[2]:02X}", ha='center', va='top', fontsize=9)
    ax.set_xlim(-0.5, len(palette) - 0.5)
    ax.set_ylim(-1, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    return fig, ax

def side_by_side(original_rgba, quant_rgba, title_left="Original", title_right=f"Quantisiert"):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(original_rgba)
    axs[0].set_title(title_left)
    axs[0].axis('off')

    axs[1].imshow(quant_rgba)
    axs[1].set_title(title_right)
    axs[1].axis('off')

    plt.tight_layout()
    return fig, axs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Pfad zur PNG-Datei")
    parser.add_argument("--n_colors", "-n", type=int, default=8, help="Anzahl Farben (Cluster)")
    parser.add_argument("--output", "-o", default="quantized.png", help="Ausgabedatei (PNG)")
    parser.add_argument("--no_show", action="store_true", help="Plots nicht anzeigen (nur speichern)")
    args = parser.parse_args()

    # Laden
    img = load_image_rgba(args.input)
    rgb, alpha = image_to_arrays(img)

    # K-Means
    quant_rgb, palette, labels = kmeans_quantize(rgb, n_colors=args.n_colors)

    # Alpha beibehalten (transparente Pixel bleiben transparent)
    # Optional: transparente Pixel unverändert lassen
    original_rgba = np.dstack([rgb, alpha]).astype(np.uint8)
    quant_rgba = np.dstack([quant_rgb, alpha]).astype(np.uint8)

    # Speichern
    save_rgba(quant_rgb, alpha, args.output)
    print(f"[OK] Quantisiertes Bild gespeichert als: {args.output}")

    # Debug-Palette
    fig_palette, _ = plot_palette(palette, title=f"Palette ({args.n_colors} Farben)")
    # Vorher/Nachher
    fig_compare, _ = side_by_side(original_rgba, quant_rgba, title_left="Original", title_right=f"Quantisiert ({args.n_colors})")

    if not args.no_show:
        plt.show()
    else:
        fig_palette.savefig("palette_debug.png", dpi=150, bbox_inches="tight")
        fig_compare.savefig("compare_debug.png", dpi=150, bbox_inches="tight")
        print("[OK] Debug-Plots gespeichert: palette_debug.png, compare_debug.png")

if __name__ == "__main__":
    main()

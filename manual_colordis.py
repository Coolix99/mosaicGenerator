#!/usr/bin/env python3
"""
Quantize PNG to a user-specified HEX palette.

- Loads PNG (with alpha)
- Maps each pixel to the nearest palette color (RGB Euclidean)
- Saves the result (alpha preserved)
- Plots palette debug and before/after comparison

Usage:
  python quantize_with_manual_palette.py \
      --input input.png \
      --output quantized.png \
      --palette "#0F0F0F,#F94144,#577590,#F9C74F" \
      [--no_show]
  # or load from file (one HEX per line):
  python quantize_with_manual_palette.py -i input.png -o out.png --palette_file palette.txt
"""

import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# ---------- Helpers ----------

def parse_hex_color(s: str) -> np.ndarray:
    """Parse '#RRGGBB' or 'RRGGBB' into uint8 RGB np.array([r,g,b])."""
    s = s.strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) != 6:
        raise ValueError(f"Invalid HEX color '{s}'. Expected 6 hex digits.")
    try:
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
    except ValueError as e:
        raise ValueError(f"Invalid HEX color '{s}'.") from e
    return np.array([r, g, b], dtype=np.uint8)

def parse_palette(palette_arg: str = None, palette_file: str = None) -> np.ndarray:
    """Return palette as (K,3) uint8 RGB array."""
    colors = []
    if palette_arg:
        for token in palette_arg.split(","):
            if token.strip():
                colors.append(parse_hex_color(token))
    if palette_file:
        txt = Path(palette_file).read_text(encoding="utf-8")
        for line in txt.splitlines():
            line = line.strip()
            if not line or line.startswith("#") and len(line) != 7:
                # allow comments or empty lines
                if not line or line.startswith("//") or line.startswith(";"):
                    continue
            if line:
                colors.append(parse_hex_color(line))
    if not colors:
        raise ValueError("Palette is empty. Provide --palette or --palette_file.")
    # Remove duplicates while preserving order
    uniq = []
    seen = set()
    for c in colors:
        key = (int(c[0]), int(c[1]), int(c[2]))
        if key not in seen:
            uniq.append(c)
            seen.add(key)
    return np.vstack(uniq).astype(np.uint8)

def load_image_rgba(path: str) -> Image.Image:
    return Image.open(path).convert("RGBA")

def image_to_arrays(img_rgba: Image.Image):
    arr = np.array(img_rgba, dtype=np.uint8)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]
    return rgb, alpha

def assign_to_nearest_palette(rgb: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """
    Vectorized nearest-neighbor in RGB.
    rgb: (H,W,3) uint8
    palette: (K,3) uint8
    returns quant_rgb: (H,W,3) uint8
    """
    h, w, _ = rgb.shape
    pixels = rgb.reshape(-1, 3).astype(np.float64)  # int16 to avoid overflow in diffs
    pal = palette.astype(np.float64)

    # Compute squared distances to all palette colors: (N, K)
    # (pixels[:, None, :] - pal[None, :, :]) -> (N,K,3)
    diffs = pixels[:, None, :] - pal[None, :, :]
    print(diffs.shape)
    d2 = np.sum(diffs * diffs, axis=2)
    nearest_idx = np.argmin(d2, axis=1)  # (N,)
    print(d2)
    print(nearest_idx)
    quant_pixels = palette[nearest_idx].reshape(h, w, 3)
    return quant_pixels

def save_rgba(rgb: np.ndarray, alpha: np.ndarray, path: str):
    out_rgba = np.dstack([rgb, alpha]).astype(np.uint8)
    Image.fromarray(out_rgba, mode="RGBA").save(path)

def plot_palette(palette: np.ndarray, title="Palette"):
    fig, ax = plt.subplots(figsize=(max(6, len(palette) * 0.7), 2.5))
    x = np.arange(len(palette))
    for i, c in enumerate(palette):
        ax.scatter([i], [0], s=3000, marker='s', c=[c / 255.0], edgecolors='black', linewidths=0.5)
        ax.text(i, -0.45, f"#{c[0]:02X}{c[1]:02X}{c[2]:02X}", ha='center', va='top', fontsize=9)
    ax.set_xlim(-0.5, len(palette) - 0.5)
    ax.set_ylim(-1, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    return fig, ax

def side_by_side(original_rgba: np.ndarray, quant_rgba: np.ndarray,
                 title_left="Original", title_right="Quantized"):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(original_rgba)
    axs[0].set_title(title_left)
    axs[0].axis('off')

    axs[1].imshow(quant_rgba)
    axs[1].set_title(title_right)
    axs[1].axis('off')

    plt.tight_layout()
    return fig, axs

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Path to input PNG")
    ap.add_argument("-o", "--output", default="quantized_manual_palette.png", help="Output PNG")
    ap.add_argument("-p", "--palette", default=None,
                    help='Comma-separated HEX list, e.g. "#112233,#AABBCC,#FF8800"')
    ap.add_argument("--palette_file", default=None, help="Text file with one HEX per line")
    ap.add_argument("--no_show", action="store_true", help="Do not show plots; just save files")
    ap.add_argument("--show_palette_first", action="store_true",
                    help="Show only palette preview and exit (useful to verify colors)")
    args = ap.parse_args()

    try:
        palette = parse_palette(args.palette, args.palette_file)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    # Optional: just preview the palette and exit
    if args.show_palette_first:
        plot_palette(palette, title=f"Palette ({len(palette)} colors)")
        if args.no_show:
            plt.savefig("palette_preview.png", dpi=150, bbox_inches="tight")
            print("[OK] Saved palette_preview.png")
        else:
            plt.show()
        sys.exit(0)

    # Load image
    img = load_image_rgba(args.input)
    rgb, alpha = image_to_arrays(img)

    # Assign to nearest palette color
    quant_rgb = assign_to_nearest_palette(rgb, palette)

    # Keep alpha as-is
    original_rgba = np.dstack([rgb, alpha]).astype(np.uint8)
    quant_rgba = np.dstack([quant_rgb, alpha]).astype(np.uint8)

    # Save result
    save_rgba(quant_rgb, alpha, args.output)
    print(f"[OK] Wrote: {args.output}")

    # Debug plots
    fig_palette, _ = plot_palette(palette, title=f"Palette ({len(palette)} colors)")
    fig_compare, _ = side_by_side(original_rgba, quant_rgba,
                                  title_left="Original", title_right="Quantized (manual palette)")

    if args.no_show:
        fig_palette.savefig("palette_debug.png", dpi=150, bbox_inches="tight")
        fig_compare.savefig("compare_debug.png", dpi=150, bbox_inches="tight")
        print("[OK] Saved: palette_debug.png, compare_debug.png")
    else:
        plt.show()

if __name__ == "__main__":
    main()

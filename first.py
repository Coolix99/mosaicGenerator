import os
import random
import argparse
from PIL import Image
from PIL import Image, ImageOps
import numpy as np
from tqdm import tqdm

def load_tiles(tile_folder, tile_size, max_tiles=None):
    tiles = []
    image_files = os.listdir(tile_folder)
    if max_tiles is not None:
        image_files = image_files[:max_tiles]

    for fname in tqdm(image_files, desc="Loading tiles"):
        path = os.path.join(tile_folder, fname)
        try:
            img = Image.open(path)
            img = ImageOps.exif_transpose(img)    # <-- apply EXIF rotation
            img = img.convert('RGB')
        except Exception:
            continue
        img = img.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        avg = np.array(img).mean(axis=(0, 1))
        tiles.append((img, avg))
    if not tiles:
        raise ValueError(f"No images found in {tile_folder}")
    return tiles


def get_target_pixels(target_path, scale=1.0):
    tgt = Image.open(target_path)
    tgt = ImageOps.exif_transpose(tgt)           # <-- apply here too
    tgt = tgt.convert('RGB')
    if scale != 1.0:
        w, h = tgt.size
        new_size = (int(w * scale), int(h * scale))
        tgt = tgt.resize(new_size, Image.Resampling.LANCZOS)
    pixels = np.array(tgt)
    w, h = tgt.size
    return pixels, w, h

def match_nearest(target_pixels, tiles):
    """
    For each target pixel, pick the tile whose average color is nearest (Euclidean).
    Returns a 2D array of tile indices.
    """
    tile_avgs = np.array([avg for _, avg in tiles])
    h, w, _ = target_pixels.shape
    indices = np.zeros((h, w), dtype=int)
    # Flatten for speed
    t_flat = target_pixels.reshape(-1, 3)
    # Compute distances to all tile averages
    dists = np.linalg.norm(t_flat[:, None, :] - tile_avgs[None, :, :], axis=2)
    closest = np.argmin(dists, axis=1)
    indices = closest.reshape((h, w))
    return indices


def match_weighted(target_pixels, tiles, top_k=10, epsilon=1e-6):
    """
    For each target pixel, pick among the top_k nearest tiles randomly, weighted by inverse distance.
    Returns a 2D array of tile indices.
    """
    tile_avgs = np.array([avg for _, avg in tiles])
    h, w, _ = target_pixels.shape
    indices = np.zeros((h, w), dtype=int)
    t_flat = target_pixels.reshape(-1, 3)
    dists = np.linalg.norm(t_flat[:, None, :] - tile_avgs[None, :, :], axis=2)
    # For each pixel, pick among top_k
    for i, dist_row in enumerate(tqdm(dists, desc="Matching tiles (weighted)")):
        # get indices of top_k smallest distances
        k = min(top_k, dist_row.shape[0])
        nearest = np.argpartition(dist_row, k)[:k]
        # compute weights
        weights = 1 / (dist_row[nearest] + epsilon)
        # normalize
        probs = weights / weights.sum()
        choice = np.random.choice(nearest, p=probs)
        indices.flat[i] = choice
    return indices.reshape((h, w))


def build_mosaic(target_pixels, tiles, indices, tile_size):
    """
    Assemble the mosaic image using tile assignments with a single flat loop.
    Avoids nested for-loops for better performance.
    """
    
    h, w = indices.shape
    mosaic_array = np.zeros((h * tile_size, w * tile_size, 3), dtype=np.uint8)

    # Pre-cache the tile images as numpy arrays
    tile_arrays = [np.array(tile[0]) for tile in tiles]

    for idx in tqdm(range(h * w), desc="Assembling mosaic"):
        y = idx // w
        x = idx % w
        tile_idx = indices[y, x]
        tile = tile_arrays[tile_idx]
        y0, y1 = y * tile_size, (y + 1) * tile_size
        x0, x1 = x * tile_size, (x + 1) * tile_size
        mosaic_array[y0:y1, x0:x1] = tile

    return Image.fromarray(mosaic_array)


def create_mosaic(tile_folder, target_path, output_path, tile_size, algorithm, top_k, scale=1.0, max_tiles=None):
    tiles = load_tiles(tile_folder, tile_size, max_tiles)
    pixels, w, h = get_target_pixels(target_path, scale)
    print(f"Target image size: {w}x{h}, Tile size: {tile_size}x{tile_size}")
    if algorithm == 'nearest':
        indices = match_nearest(pixels, tiles)
    elif algorithm == 'weighted':
        indices = match_weighted(pixels, tiles, top_k=top_k)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    mosaic = build_mosaic(pixels, tiles, indices, tile_size)
    mosaic.save(output_path)
    print(f"Mosaic saved to {output_path}")



def parse_args():
    parser = argparse.ArgumentParser(description='Create a photo mosaic from image tiles')
    parser.add_argument('--tiles', required=True, help='Folder containing tile images')
    parser.add_argument('--target', required=True, help='Target image path')
    parser.add_argument('--out', required=True, help='Output mosaic image path')
    parser.add_argument('--tile-size', type=int, default=64, help='Size (in px) for each tile (default: 64)')
    parser.add_argument('--algorithm', choices=['nearest', 'weighted'], default='nearest',
                        help='Tile matching algorithm: nearest or weighted (default: nearest)')
    parser.add_argument('--top-k', type=int, default=10,
                        help='Number of nearest tiles to consider for weighted algorithm (default: 10)')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Scale factor to resize target image before processing (default: 1.0)')
    return parser.parse_args()


if __name__ == '__main__':
    create_mosaic(
        tile_folder=r"C:\Users\kotzm\Downloads\drive-download-20250720T102006Z-1-001",
        target_path=r"C:\Users\kotzm\Downloads\drive-download-20250720T102006Z-1-001\IMG_20250718_210351.jpg",
        output_path=r"C:\Users\kotzm\Downloads\mosaic_output.png",
        tile_size=64,
        algorithm="weighted",
        top_k=15,
        scale=0.02,
        max_tiles=100
    )
    # args = parse_args()
    # create_mosaic(
    #     tile_folder=args.tiles,
    #     target_path=args.target,
    #     output_path=args.out,
    #     tile_size=args.tile_size,
    #     algorithm=args.algorithm,
    #     top_k=args.top_k
    # )

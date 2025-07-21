import os
import random
import argparse
from PIL import Image, ImageOps
import numpy as np
from tqdm import tqdm


def load_tiles(tile_folder, tile_size, max_tiles=None):
    """
    Load images from folder, resize to multiples of tile_size preserving aspect ratio
    (n*tile_size x tile_size or tile_size x n*tile_size), compute their average color,
    and store grid cell dimensions.
    Returns a list of tuples: (tile_image, avg_color, cell_h, cell_w).
    """
    tiles = []
    image_files = os.listdir(tile_folder)
    if max_tiles is not None:
        image_files = image_files[:max_tiles]

    for fname in tqdm(image_files, desc="Loading tiles"):
        path = os.path.join(tile_folder, fname)
        try:
            img = Image.open(path)
            img = ImageOps.exif_transpose(img)
            img = img.convert('RGB')
        except Exception:
            continue
        w0, h0 = img.size
        # Compute number of grid cells along each dimension
        ratio = w0 / h0
        if ratio >= 1:
            # wide image: height = 1 cell, width = round(ratio) cells
            cell_h, cell_w = 1, max(1, int(round(ratio)))
        else:
            # tall image: width = 1 cell, height = round(1/ratio) cells
            cell_w, cell_h = 1, max(1, int(round(1/ratio)))
        # Resize to actual pixels
        new_size = (cell_w * tile_size, cell_h * tile_size)
        img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
        avg = np.array(img_resized).mean(axis=(0, 1))
        tiles.append((img_resized, avg, cell_h, cell_w))

    if not tiles:
        raise ValueError(f"No images found in {tile_folder}")
    return tiles


def compute_integral(image_array):
    """
    Compute 2D integral image with zero padding for fast region sum queries.
    """
    # image_array shape (H, W, C)
    H, W, C = image_array.shape
    integral = np.zeros((H+1, W+1, C), dtype=np.float64)
    integral[1:, 1:, :] = np.cumsum(np.cumsum(image_array, axis=0), axis=1)
    return integral


def region_average(integral, x, y, h, w):
    """
    Compute average color in rectangle at pixel coords (x,y) with height h and width w.
    """
    y1, x1 = y, x
    y2, x2 = y + h, x + w
    total = (integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1])
    return total / (h * w)


def create_mosaic(tile_folder, target_path, output_path, tile_size,
                  top_k_pos, pitch, scale, max_tiles=None,sample_size=100):
    # Load and prepare tiles
    tiles = load_tiles(tile_folder, tile_size, max_tiles)
    # Load target, apply EXIF, scale, and compute integral
    tgt = Image.open(target_path)
    tgt = ImageOps.exif_transpose(tgt).convert('RGB')
    if scale != 1.0:
        w0, h0 = tgt.size
        tgt = tgt.resize((int(w0 * scale), int(h0 * scale)), Image.Resampling.LANCZOS)
    target_arr = np.array(tgt)
    H, W, _ = target_arr.shape
    integral = compute_integral(target_arr)
    
    occupancy = np.zeros((H, W), dtype=bool)
    # Canvas to paste mosaic
    mosaic = Image.new('RGB', (W * tile_size, H * tile_size))

    total_cells = H * W
    filled = 0
    pbar = tqdm(total=total_cells, desc="Coverage", unit="cell")

    while True:
        placed = 0

        # Shuffle tile order for extra randomness each pass
        random.shuffle(tiles)
        for img_resized, avg_tile, cell_h, cell_w in tiles:
            # Build free positions list once per tile
            free = np.logical_not(occupancy)
            # A fast list comprehension of all non‐overlapping starts:
            all_positions = [
                (gy, gx)
                for gy in range(0, H - cell_h + 1)
                for gx in range(0, W - cell_w + 1)
                if free[gy:gy+cell_h, gx:gx+cell_w].all()
            ]
            
            if not all_positions:
                continue

            # Sample S = min(sample_size, len(all_positions))
            S = min(sample_size, len(all_positions))
            sampled = random.sample(all_positions, S)
    
            # Evaluate only the sampled positions
            candidates = []
            for gy, gx in sampled:
                avg_region = region_average(integral, gx, gy, cell_h, cell_w)
                diff = np.linalg.norm(avg_tile - avg_region)
                candidates.append((diff, gx, gy, avg_region))

            candidates.sort(key=lambda x: x[0])
            top = candidates[:min(top_k_pos, len(candidates))]
            _, gx, gy, avg_region = random.choice(top)

            # Place tile
            occupancy[gy:gy+cell_h, gx:gx+cell_w] = True
            delta = cell_h * cell_w
            filled += delta
            pbar.update(delta)

            arr = np.array(img_resized, dtype=np.float32)
            arr = arr * (1 - pitch) + avg_region.reshape(1,1,3) * pitch
            tile_adj = Image.fromarray(arr.clip(0,255).astype(np.uint8))
            mosaic.paste(tile_adj, (gx * tile_size, gy * tile_size))
            placed += 1

        # Termination
        if placed == 0 or filled >= total_cells:
            break

    pbar.close()
    mosaic.save(output_path)
    print(f"Mosaic saved to {output_path}") 


def parse_args():
    parser = argparse.ArgumentParser(description='Create a stochastic block-based photo mosaic')
    parser.add_argument('--tiles', required=True, help='Folder containing tile images')
    parser.add_argument('--target', required=True, help='Target image path')
    parser.add_argument('--out', required=True, help='Output mosaic image path')
    parser.add_argument('--tile-size', type=int, default=64,
                        help='Base cell size in pixels (default: 64)')
    parser.add_argument('--algorithm', choices=['grid'], default='grid',
                        help='(grid placement only)')
    parser.add_argument('--top-k-pos', type=int, default=5,
                        help='Number of best positions to sample from (default: 5)')
    parser.add_argument('--pitch', type=float, default=0.0,
                        help='Blend factor between tile and region average (0-1)')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Scale factor for target image before processing')
    parser.add_argument('--max-tiles', type=int, default=None,
                        help='Limit number of tiles loaded (for debugging)')
    return parser.parse_args()


if __name__ == '__main__':
    #args = parse_args()
    create_mosaic(
        tile_folder=r"C:\Users\kotzm\Downloads\drive-download-20250720T102006Z-1-001",
        target_path=r"C:\Users\kotzm\Downloads\drive-download-20250720T102006Z-1-001\IMG_20250718_210351.jpg",
        output_path=r"C:\Users\kotzm\Downloads\mosaic_output.png",
        tile_size=64,
        top_k_pos=15,
        pitch=0.1,
        scale=0.02,
        max_tiles=100
    )

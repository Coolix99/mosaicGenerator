import os
import random
import argparse
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm


def load_tiles(tile_folder, tile_size, max_tiles=None):
    """
    Load images from folder, resize to multiples of tile_size preserving aspect ratio
    (n * tile_size x tile_size or tile_size x n * tile_size), compute their average color,
    and store grid cell dimensions.
    Returns: list of (img_resized, avg_color, cell_h, cell_w)
    """
    tiles = []
    image_files = os.listdir(tile_folder)
    if max_tiles is not None:
        image_files = image_files[:max_tiles]

    for fname in tqdm(image_files, desc="Loading tiles"):
        path = os.path.join(tile_folder, fname)
        try:
            img = Image.open(path)
            img = ImageOps.exif_transpose(img).convert('RGB')
        except Exception:
            continue
        w0, h0 = img.size
        ratio = w0 / h0
        if ratio >= 1:
            # wide image: height = 1, width = round(ratio)
            cell_h, cell_w = 1, max(1, int(round(ratio)))
        else:
            # tall image: width = 1, height = round(1/ratio)
            cell_w, cell_h = 1, max(1, int(round(1 / ratio)))
        new_size = (cell_w * tile_size, cell_h * tile_size)
        img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
        avg = np.array(img_resized).mean(axis=(0, 1))
        tiles.append((img_resized, avg, cell_h, cell_w))

    if not tiles:
        raise ValueError(f"No images found in {tile_folder}")
    return tiles


def compute_integral(image_arr):
    """Compute 2D integral image with zero padding"""
    H, W, C = image_arr.shape
    integral = np.zeros((H + 1, W + 1, C), dtype=np.float64)
    integral[1:, 1:, :] = np.cumsum(np.cumsum(image_arr, axis=0), axis=1)
    return integral


def region_average(integral, gx, gy, cell_h, cell_w):
    """Average color in rectangle starting at (gx, gy) spanning cell_h x cell_w"""
    y1, x1 = gy, gx
    y2, x2 = gy + cell_h, gx + cell_w
    total = (integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1])
    return total / (cell_h * cell_w)


def is_duplicate_nearby(tile_id, gx, gy, cell_h, cell_w, placement_matrix):
    """Check if same tile_id is adjacent to this block in placement_matrix"""
    H, W = placement_matrix.shape
    # iterate neighbors around the block
    for dy in range(-1, cell_h + 1):
        for dx in range(-1, cell_w + 1):
            ny = gy + dy
            nx = gx + dx
            if (0 <= ny < H) and (0 <= nx < W):
                if placement_matrix[ny, nx] == tile_id:
                    return True
    return False


def create_mosaic(tile_folder, target_path, output_path, tile_size,
                  top_k_pos=5, beta=1000.0, gamma=0.1, scale=1.0, max_tiles=None):
    # Load tiles
    tiles = load_tiles(tile_folder, tile_size, max_tiles)
    num_tiles = len(tiles)

    # Load target image and compute integral
    tgt = Image.open(target_path)
    tgt = ImageOps.exif_transpose(tgt).convert('RGB')
    if scale != 1.0:
        w0, h0 = tgt.size
        tgt = tgt.resize((int(w0 * scale), int(h0 * scale)), Image.Resampling.LANCZOS)
    target_arr = np.array(tgt)
    H, W, _ = target_arr.shape
    integral = compute_integral(target_arr)

    # Initialize occupancy and placement matrices
    occupancy = np.zeros((H, W), dtype=bool)
    placement = -1 * np.ones((H, W), dtype=int)
    usage_count = np.zeros(num_tiles, dtype=int)
    total_cells = H * W

    # Prepare output canvas
    mosaic = Image.new('RGB', (W * tile_size, H * tile_size))
    filled = 0

    # Shuffle tile iteration order
    tile_indices = list(range(num_tiles))
    random.shuffle(tile_indices)

    # Greedy placement loop
    for tid in tqdm(tile_indices, desc="Placing tiles"):
        img_resized, avg_tile, cell_h, cell_w = tiles[tid]
        # find all free anchor positions where tile fits
        free = ~occupancy
        positions = [(gy, gx)
                     for gy in range(0, H - cell_h + 1)
                     for gx in range(0, W - cell_w + 1)
                     if free[gy:gy+cell_h, gx:gx+cell_w].all()]
        if not positions:
            continue

        # sample positions if too many
        sample_size = min(len(positions), 200)
        sampled = random.sample(positions, sample_size)

        # evaluate candidates
        candidates = []
        usage_mean = filled / max(1, num_tiles)
        for gy, gx in sampled:
            avg_region = region_average(integral, gx, gy, cell_h, cell_w)
            diff = np.linalg.norm(avg_tile - avg_region)
            dup = 1.0 if is_duplicate_nearby(tid, gx, gy, cell_h, cell_w, placement) else 0.0
            usage_pen = (usage_count[tid] - usage_mean) ** 2
            score = diff + beta * dup + gamma * usage_pen
            candidates.append((score, gy, gx, avg_region))

        # pick best
        best = min(candidates, key=lambda x: x[0])
        _, gy, gx, avg_region = best

        # place tile
        occupancy[gy:gy+cell_h, gx:gx+cell_w] = True
        placement[gy:gy+cell_h, gx:gx+cell_w] = tid
        usage_count[tid] += 1
        filled += cell_h * cell_w

        # paste blended tile
        arr = np.array(img_resized, dtype=np.float32)
        tile_out = (arr * (1 - gamma) + avg_region.reshape(1,1,3) * gamma).clip(0,255).astype(np.uint8)
        mosaic.paste(Image.fromarray(tile_out), (gx * tile_size, gy * tile_size))

    # Save result
    mosaic.save(output_path)
    print(f"Mosaic saved to {output_path}")


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Efficient photo mosaic generator')
    # parser.add_argument('--tiles', required=True, help='Folder with tile images')
    # parser.add_argument('--target', required=True, help='Target image path')
    # parser.add_argument('--out', required=True, help='Output mosaic path')
    # parser.add_argument('--tile-size', type=int, default=64, help='Base cell size in px')
    # parser.add_argument('--top-k-pos', type=int, default=5, help='Sample best positions (default: 5)')
    # parser.add_argument('--beta', type=float, default=1000.0, help='Penalty for duplicate neighbors')
    # parser.add_argument('--gamma', type=float, default=0.1, help='Weight for usage fairness penalty')
    # parser.add_argument('--scale', type=float, default=1.0, help='Scale factor for target')
    # parser.add_argument('--max-tiles', type=int, default=None, help='Limit number of tiles')
    # args = parser.parse_args()
    create_mosaic(tile_folder=r"C:\Users\kotzm\Downloads\drive-download-20250720T102006Z-1-001",
                  target_path=r"C:\Users\kotzm\Downloads\drive-download-20250720T102006Z-1-001\IMG_20250718_210351.jpg",
                  output_path=r"C:\Users\kotzm\Downloads\mosaic_output.png",
                  tile_size=64,
                  top_k_pos=15,
                  scale=0.02,
                  max_tiles=100)


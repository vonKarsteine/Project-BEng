"""Convert a FAST-LIO .pcd map into a 2D occupancy raster for RL training.

Replaces the thesis's lost 'modified random_forest_sensing.cpp' path: crop the
cloud to the flight band, histogram into a grid, save as .npy + .png.

Supports ascii and binary (uncompressed) PCD v0.7 with x,y,z float32 fields.
For binary_compressed clouds, convert first on the ROS machine:
    pcl_convert_pcd_ascii_binary in.pcd out.pcd 0

Usage:
    python pcd_to_raster.py scans.pcd --out map_raster --resolution 0.1 \
        --zmin 0.3 --zmax 1.8 --min-points 3
"""
import argparse
import os
import re
import struct

import numpy as np


def load_pcd_xyz(path):
    with open(path, 'rb') as f:
        header = {}
        while True:
            line = f.readline().decode('ascii', errors='replace').strip()
            if not line or line.startswith('#'):
                continue
            key, _, value = line.partition(' ')
            header[key.upper()] = value
            if key.upper() == 'DATA':
                break
        fields = header['FIELDS'].split()
        sizes = list(map(int, header['SIZE'].split()))
        types = header['TYPE'].split()
        counts = list(map(int, header.get('COUNT', ' '.join(['1'] * len(fields))).split()))
        n_points = int(header['POINTS'])
        data_mode = header['DATA'].split()[0]

        if data_mode == 'ascii':
            body = f.read().decode('ascii', errors='replace')
            rows = [r.split() for r in body.strip().splitlines() if r.strip()]
            arr = np.array(rows, dtype=np.float64)
            col = 0
            cols = {}
            for name, cnt in zip(fields, counts):
                cols[name] = col
                col += cnt
            return arr[:, [cols['x'], cols['y'], cols['z']]].astype(np.float32)

        if data_mode == 'binary':
            point_step = sum(s * c for s, c in zip(sizes, counts))
            raw = f.read(point_step * n_points)
            offsets = {}
            off = 0
            for name, size, cnt in zip(fields, sizes, counts):
                offsets[name] = off
                off += size * cnt
            pts = np.empty((n_points, 3), dtype=np.float32)
            for i in range(n_points):
                base = i * point_step
                pts[i, 0] = struct.unpack_from('<f', raw, base + offsets['x'])[0]
                pts[i, 1] = struct.unpack_from('<f', raw, base + offsets['y'])[0]
                pts[i, 2] = struct.unpack_from('<f', raw, base + offsets['z'])[0]
            return pts

    raise ValueError('unsupported PCD DATA mode: {} (convert binary_compressed '
                     'with pcl_convert_pcd_ascii_binary first)'.format(data_mode))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pcd', help='input .pcd (e.g. fast_lio/PCD/scans.pcd)')
    parser.add_argument('--out', default='map_raster', help='output basename')
    parser.add_argument('--resolution', type=float, default=0.1, help='cell size [m]')
    parser.add_argument('--zmin', type=float, default=0.3,
                        help='lower crop bound [m] (drop the floor)')
    parser.add_argument('--zmax', type=float, default=1.8,
                        help='upper crop bound [m] (drop the ceiling)')
    parser.add_argument('--min-points', type=int, default=3,
                        help='points per cell to mark it occupied')
    args = parser.parse_args()

    pts = load_pcd_xyz(args.pcd)
    pts = pts[np.isfinite(pts).all(axis=1)]
    band = pts[(pts[:, 2] >= args.zmin) & (pts[:, 2] <= args.zmax)]
    print('loaded {} points, {} inside z=[{}, {}]'.format(
        len(pts), len(band), args.zmin, args.zmax))

    x0, y0 = band[:, 0].min(), band[:, 1].min()
    xi = ((band[:, 0] - x0) / args.resolution).astype(np.int64)
    yi = ((band[:, 1] - y0) / args.resolution).astype(np.int64)
    W, H = xi.max() + 1, yi.max() + 1
    counts = np.zeros((W, H), dtype=np.int64)
    np.add.at(counts, (xi, yi), 1)
    occupancy = (counts >= args.min_points).astype(np.uint8)

    np.save(args.out + '.npy', occupancy)
    meta = dict(origin_x=float(x0), origin_y=float(y0),
                resolution=args.resolution, zmin=args.zmin, zmax=args.zmax)
    with open(args.out + '.meta.txt', 'w') as f:
        for k, v in meta.items():
            f.write('{}: {}\n'.format(k, v))

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 6))
        plt.imshow(occupancy.T, origin='lower', cmap='gray_r',
                   extent=[x0, x0 + W * args.resolution, y0, y0 + H * args.resolution])
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.title('occupancy raster ({} m/cell)'.format(args.resolution))
        plt.savefig(args.out + '.png', dpi=150, bbox_inches='tight')
        print('saved {0}.npy, {0}.meta.txt, {0}.png ({1}x{2} cells)'.format(
            args.out, W, H))
    except ImportError:
        print('saved {0}.npy, {0}.meta.txt ({1}x{2} cells; matplotlib not '
              'available for the PNG preview)'.format(args.out, W, H))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""检查并打印 HDF5 文件结构与数据摘要的脚本。

用法:
  python scripts/check_hdf5.py /path/to/file.hdf5

脚本会递归列出组/数据集、shape、dtype，并对数值数组打印最小/最大/均值（如可打印）。
"""

import sys
import argparse
import h5py
import numpy as np


def summarize_dataset(name, ds):
    try:
        data = ds[()]
    except Exception:
        print(f"  - {name}: <unreadable> (non-array or complex type)")
        return
    shape = getattr(data, 'shape', None)
    dtype = getattr(data, 'dtype', None)
    print(f"  - {name}: shape={shape}, dtype={dtype}")
    if np.issubdtype(np.dtype(dtype), np.number):
        # compute simple stats on a sampled subset if large
        flat = data.ravel()
        if flat.size > 100000:
            idx = np.random.choice(flat.size, 100000, replace=False)
            flat = flat[idx]
        print(f"      min={flat.min()}, max={flat.max()}, mean={flat.mean():.6f}")


def walk_hdf5(f):
    """Yield (path, obj) for datasets and groups."""
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name}")
            summarize_dataset(name, obj)
        elif isinstance(obj, h5py.Group):
            print(f"Group: {name}")

    f.visititems(visitor)


def print_top_level(f):
    print("Top-level keys:")
    for k in f.keys():
        obj = f[k]
        kind = 'Group' if isinstance(obj, h5py.Group) else 'Dataset'
        print(f"- {k} ({kind})")


def main():
    parser = argparse.ArgumentParser(description='Inspect HDF5 file')
    parser.add_argument('hdf5_path', help='Path to HDF5 file')
    parser.add_argument('--list', action='store_true', help='Only list top-level keys')
    args = parser.parse_args()

    path = args.hdf5_path
    try:
        with h5py.File(path, 'r') as f:
            print(f"Opened HDF5: {path}\n")
            print_top_level(f)
            print('\nDetailed traversal:')
            if args.list:
                return
            walk_hdf5(f)
    except Exception as e:
        print(f"Failed to open or parse HDF5: {e}")
        sys.exit(2)


if __name__ == '__main__':
    main()
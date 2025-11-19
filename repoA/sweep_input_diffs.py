"""Sweep HW=1 input differences and compute PCA/KMeans metrics.

Exposes function:
- sweep_input_differences(...): returns list of dict rows and side-effect write CSVs if paths provided.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from sklearn.metrics import silhouette_score

from utils.cipher_utils import build_generator_for_diff
from utils.pca_utils import EigenValueDecomposition, DimensionReduction
from utils.cluster_utils import kmeans_clustering, calculate_silhouette


def sweep_input_differences(
    cipher_mod,
    *,
    nr: int,
    pairs: int,
    datasize: int,
    clusters: int,
    max_bits: Optional[int],
    use_gpu: bool,
    lambda_base: float,
    t0: float,
    results_csv: Optional[Path] = None,
    save_eigen_csv: bool = False,
    eigen_dir: Optional[Path] = None,
) -> List[Dict]:
    total_bits = cipher_mod.plain_bits
    num_diffs = max_bits if max_bits is not None else total_bits

    if results_csv:
        results_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(results_csv, 'w', newline='', encoding='utf-8') as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow([
                'bit_pos', 'input_diff_hex', 'biased_pcs', 'max_diff',
                'silhouette_clusters', 'silhouette_true', 'elapsed_sec'
            ])

    rows: List[Dict] = []

    for bit_pos in range(num_diffs):
        tick = time.time()
        input_diff_int = 1 << bit_pos
        input_diff_hex = f"0x{input_diff_int:016X}"

        gen = build_generator_for_diff(
            cipher_mod,
            input_diff_int=input_diff_int,
            nr=nr,
            pairs=pairs,
            n_samples=datasize,
            batch_size=datasize,
            use_gpu=use_gpu,
        )
        X_batch, Y_batch = gen[0]

        ev_ratio, ev_vec = EigenValueDecomposition(dataset=X_batch)
        num_biased = int(np.sum(np.abs(ev_ratio - lambda_base) > t0))
        max_diff = float(np.max(ev_ratio) - lambda_base)

        projected = DimensionReduction(dataset=X_batch, n_components=3)

        if datasize < clusters:
            raise ValueError(f'datasize ({datasize}) must be >= clusters ({clusters})')
        labels_k = kmeans_clustering(projected, clusters, n_init=10)
        sil_k = calculate_silhouette(projected, labels_k)

        sil_true = None
        try:
            sil_true = silhouette_score(projected, Y_batch)
        except Exception:
            pass

        row = {
            'bit_pos': bit_pos,
            'input_diff_hex': input_diff_hex,
            'biased_pcs': num_biased,
            'max_diff': max_diff,
            'silhouette_clusters': sil_k,
            'silhouette_true': sil_true,
            'elapsed_sec': (time.time() - tick),
        }
        rows.append(row)

        if results_csv:
            with open(results_csv, 'a', newline='', encoding='utf-8') as fcsv:
                writer = csv.writer(fcsv)
                writer.writerow([
                    bit_pos, input_diff_hex, num_biased,
                    f"{max_diff:.6f}", f"{sil_k:.6f}",
                    (f"{sil_true:.6f}" if sil_true is not None else ''),
                    f"{(time.time() - tick):.3f}",
                ])

        if save_eigen_csv and eigen_dir is not None:
            eigen_dir.mkdir(parents=True, exist_ok=True)
            np.savetxt(eigen_dir / f'eigen_ratios_bit_{bit_pos:02d}.csv', ev_ratio.reshape(1, -1), delimiter=',')

    return rows

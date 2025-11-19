"""Delta-key search utilities.

Exposes function:
- select_best_delta_key(...): search best delta_key bit using PCA(2D)+silhouette
- run_delta_key_search_for_topK(...): orchestrate over top-K input differences
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from utils.cipher_utils import integer_to_binary_array
from make_data_train import NDCMultiPairGenerator


def select_best_delta_key(
    encryption_function,
    *,
    input_difference: int,
    plain_bits: int,
    key_bits: int,
    n_round: int,
    pairs: int,
    n_samples: int = 100_000,
    batch_size: int = 5_000,
    use_gpu: bool = True,
    random_seed: int = 42,
):
    delta_plain = integer_to_binary_array(input_difference, plain_bits)
    best_score = -1.0
    best_bit = -1
    all_scores: Dict[int, float] = {}

    for bit in range(key_bits):
        delta_key = np.zeros(key_bits, dtype=np.uint8)
        delta_key[bit] = 1

        gen = NDCMultiPairGenerator(
            encryption_function=encryption_function,
            plain_bits=plain_bits, key_bits=key_bits, nr=n_round,
            delta_state=delta_plain, delta_key=delta_key,
            pairs=pairs,
            n_samples=n_samples, batch_size=batch_size,
            use_gpu=use_gpu, to_float32=True,
            
        )
        X_val, Y_val = gen[0]

        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_val)
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(X_scaled)
            score = silhouette_score(pca_result, Y_val)
        except Exception:
            score = -1.0

        all_scores[bit] = float(score)
        if score > best_score:
            best_score = float(score)
            best_bit = bit

    return best_bit, best_score, all_scores


def run_delta_key_search_for_topK(
    cipher_mod,
    picked_rows: List[Dict],
    *,
    nr: int,
    pairs: int,
    n_samples: int,
    batch_size: int,
    use_gpu: bool,
    out_dir: Path,
):
    summary_csv = out_dir / 'best_delta_key_summary.csv'
    with open(summary_csv, 'w', newline='', encoding='utf-8') as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(['bit_pos', 'input_diff_hex', 'best_delta_key_bit', 'best_score'])

    for rec in picked_rows:
        bit_pos = rec['bit_pos']
        input_diff_hex = rec['input_diff_hex']
        input_diff_int = int(input_diff_hex, 16)

        best_bit, best_score, all_scores = select_best_delta_key(
            encryption_function=cipher_mod.encrypt,
            input_difference=input_diff_int,
            plain_bits=cipher_mod.plain_bits,
            key_bits=cipher_mod.key_bits,
            n_round=nr,
            pairs=pairs,
            n_samples=n_samples,
            batch_size=batch_size,
            use_gpu=use_gpu,
        )

        # Write per-input-diff scores
        per_csv = out_dir / f'delta_key_scores_for_{input_diff_hex}.csv'
        with open(per_csv, 'w', newline='', encoding='utf-8') as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(['delta_key_bit', 'silhouette_score'])
            for kbit, score in all_scores.items():
                writer.writerow([kbit, f'{score:.6f}'])

        # Append summary
        with open(summary_csv, 'a', newline='', encoding='utf-8') as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow([bit_pos, input_diff_hex, best_bit, f'{best_score:.6f}'])

    return summary_csv

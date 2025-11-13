"""Orchestrator for:
1. Sweep HW=1 input differences (PCA + clustering metrics)
2. Optional delta_key search on top-K ranked results

Refactored: helper logic moved to modules: cipher_utils, pca_utils, cluster_utils,
sweep_input_diffs, delta_key_search.
"""

from cipher import present80 as present
from utils.cipher_utils import resolve_cipher_module, _USING_CUPY
from sweep_input_diffs import sweep_input_differences

import argparse
import json
import time
import csv
from datetime import datetime
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Sweep HW=1 input differences and analyze with PCA/KMeans."
    )
    parser.add_argument(
        "--cipher-module",
        default="cipher.present80",
        help="Dotted path to cipher module (default: cipher.present80)",
    )
    parser.add_argument("--nr", type=int, default=7, help="Number of rounds to test")
    parser.add_argument(
        "--nr-sweep",
        type=str,
        default=None,
        help="Sweep rounds as CSV (e.g., '5,6,7') or range 'start:end[:step]' (e.g., '5:9' or '5:10:2'). Overrides --nr.",
    )
    parser.add_argument(
        "--pairs", type=int, default=1, help="Pairs per sample (feature dimension = pairs*3*plain_bits)"
    )
    parser.add_argument(
        "--datasize", type=int, default=50000, help="Samples per input difference"
    )
    parser.add_argument(
        "--clusters", type=int, default=27, help="Number of clusters for KMeans"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--max-bits",
        type=int,
        default=None,
        help="Limit sweep to first N bit positions (default: all plain bits)",
    )
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument(
        "--use-gpu", dest="use_gpu", action="store_true", help="Force GPU if available"
    )
    gpu_group.add_argument(
        "--no-gpu", dest="use_gpu", action="store_false", help="Disable GPU path"
    )
    parser.set_defaults(use_gpu=True)
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for logs/CSV (default: differences_findings/logs/<cipher>/<timestamp>)",
    )
    parser.add_argument(
        "--save-eigen-csv",
        action="store_true",
        help="Also save eigenvalue ratios per diff to CSV",
    )
    # (Delta-key functionality moved to finding_delta_key.py)

    args = parser.parse_args()

    # Resolve cipher module
    try:
        cipher_mod = resolve_cipher_module(args.cipher_module)
        cipher_name = args.cipher_module.split('.')[-1]
    except Exception:
        cipher_mod = present
        cipher_name = 'present80'
        print(f"[warn] Cannot import '{args.cipher_module}', falling back to default '{cipher_name}'.")

    # GPU availability guard
    if args.use_gpu and not _USING_CUPY:
        print("[warn] CuPy not available; switching to CPU mode.")
        args.use_gpu = False

    # Rounds sweep parsing
    def _parse_nr_sweep(spec: str):
        if not spec:
            return None
        s = spec.strip()
        if "," in s:
            vals = [int(x) for x in s.split(",") if x.strip() != ""]
            if not vals:
                raise ValueError("--nr-sweep CSV is empty")
            return vals
        parts = [int(x) for x in s.split(":") if x.strip() != ""]
        if len(parts) not in (2, 3):
            raise ValueError("--nr-sweep expects CSV or 'start:end[:step]'")
        start, end = parts[0], parts[1]
        step = parts[2] if len(parts) == 3 else 1
        if step == 0:
            raise ValueError("--nr-sweep step cannot be 0")
        if (end - start) * step <= 0:
            raise ValueError("--nr-sweep range/step inconsistent")
        return list(range(start, end, step))

    nr_list = _parse_nr_sweep(args.nr_sweep) or [args.nr]

    # Derived parameters and setup (invariant across rounds)
    pairs_sweep = args.pairs
    datasize_sweep = args.datasize
    batch_size_sweep = datasize_sweep
    lambda_base = 1 / (2 * cipher_mod.plain_bits)
    t0 = 0.1 * lambda_base

    input_dim_sweep = pairs_sweep * 3 * cipher_mod.plain_bits
    total_bits = cipher_mod.plain_bits
    num_diffs = args.max_bits if args.max_bits is not None else total_bits

    # Prepare parent output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parent_out = (
        Path(args.out_dir)
        if args.out_dir
        else Path("differences_findings/logs") / cipher_name / timestamp
    )
    parent_out.mkdir(parents=True, exist_ok=True)

    # Save global configuration
    base_config = {
        "cipher_module": args.cipher_module,
        "resolved_cipher": cipher_name,
        "pairs": pairs_sweep,
        "datasize": datasize_sweep,
        "clusters": args.clusters,
        "max_bits": num_diffs,
        "use_gpu": args.use_gpu,
        "input_dim": input_dim_sweep,
        "lambda_base": lambda_base,
        "t0": t0,
        "nr_list": nr_list,
    }
    with open(parent_out / "config.global.json", "w", encoding="utf-8") as f:
        json.dump(base_config, f, indent=2)

    # (delta-key-only path removed; use finding_delta_key.py for delta-key search)

    # Run sweep per rounds value
    for nr_sweep in nr_list:
        out_dir = parent_out if len(nr_list) == 1 else (parent_out / f"nr{nr_sweep}")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save per-round configuration
        config = {
            "cipher_module": args.cipher_module,
            "resolved_cipher": cipher_name,
            "nr": nr_sweep,
            "pairs": pairs_sweep,
            "datasize": datasize_sweep,
            "clusters": args.clusters,
            "max_bits": num_diffs,
            "use_gpu": args.use_gpu,
            "input_dim": input_dim_sweep,
            "lambda_base": lambda_base,
            "t0": t0,
        }
        with open(out_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        # CSV setup
        results_csv = out_dir / "sweep_results.csv"
        with open(results_csv, "w", newline="", encoding="utf-8") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(
                [
                    "bit_pos",
                    "input_diff_hex",
                    "biased_pcs",
                    "max_diff",
                    "silhouette_clusters",
                    "silhouette_true",
                    "elapsed_sec",
                ]
            )

        print("🔧 Setting up systematic input difference analysis (generator-based)...")
        print(f"Setup complete (generator-based):")
        print(f"   - Cipher: {cipher_name}")
        print(f"   - Testing {num_diffs} input differences (HW=1)")
        print(f"   - Rounds: {nr_sweep}")
        print(f"   - Samples per diff: {datasize_sweep}")
        print(f"   - Pairs per sample: {pairs_sweep}")
        print(f"   - Feature dimension: {input_dim_sweep}")
        print(f"   - Lambda base: {lambda_base:.6f}")
        print(f"   - Threshold t0: {t0:.6f}")
        print(f"   - Output dir: {out_dir}")

        print("\n" + "=" * 70)
        print("SWEEPING INPUT DIFFERENCES (HW=1) WITH GENERATOR")
        print("=" * 70)
        print(
            f"Note: Runtime depends on cipher and GPU. This run will iterate {num_diffs} diffs."
        )
        print()

        start_sweep = time.time()

        sweep_rows = sweep_input_differences(
            cipher_mod,
            nr=nr_sweep,
            pairs=pairs_sweep,
            datasize=datasize_sweep,
            clusters=args.clusters,
            max_bits=num_diffs,
            use_gpu=args.use_gpu,
            lambda_base=lambda_base,
            t0=t0,
            results_csv=results_csv,
            save_eigen_csv=args.save_eigen_csv,
            eigen_dir=out_dir if args.save_eigen_csv else None,
        )

        elapsed_sweep = time.time() - start_sweep
        print("=" * 70)
        print(
            f"✅ Sweep complete (nr={nr_sweep})! Total time: {elapsed_sweep:.1f}s ({elapsed_sweep/60:.1f} minutes)"
        )
        print("Results saved to:", results_csv)
        print("=" * 70)

        # === Also export sorted CSVs by key metrics (descending) ===
        def _metric_value(row, key):
            val = row.get(key)
            return val if val is not None else float('-inf')

        sorted_specs = [
            ("biased_pcs", out_dir / "sweep_sorted_by_biased_pcs.csv"),
            ("max_diff", out_dir / "sweep_sorted_by_max_diff.csv"),
            ("silhouette_clusters", out_dir / "sweep_sorted_by_silhouette_clusters.csv"),
            ("silhouette_true", out_dir / "sweep_sorted_by_silhouette_true.csv"),
        ]

        for key, path in sorted_specs:
            ranked = sorted(sweep_rows, key=lambda r: _metric_value(r, key), reverse=True)
            with open(path, "w", newline="", encoding="utf-8") as fcsv:
                writer = csv.writer(fcsv)
                writer.writerow([
                    "bit_pos",
                    "input_diff_hex",
                    "biased_pcs",
                    "max_diff",
                    "silhouette_clusters",
                    "silhouette_true",
                    "elapsed_sec",
                ])
                for r in ranked:
                    writer.writerow([
                        r["bit_pos"],
                        r["input_diff_hex"],
                        r["biased_pcs"],
                        f"{r['max_diff']:.6f}",
                        f"{r['silhouette_clusters']:.6f}",
                        (f"{r['silhouette_true']:.6f}" if r.get('silhouette_true') is not None else ""),
                        f"{r['elapsed_sec']:.3f}",
                    ])
            print(f"Sorted by {key}: {path}")

    # (Delta-key search phase removed; use finding_delta_key.py)


 # (Delta-key utilities moved to delta_key_search module)

if __name__ == "__main__":
    main()
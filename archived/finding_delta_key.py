"""Delta-key search entrypoint.

Select top-K input differences from a sweep_results.csv and search best delta_key bit
for each, writing per-diff score CSVs and a summary CSV.

Relies on:
- utils.cipher_utils.resolve_cipher_module, _USING_CUPY
- delta_key_search.run_delta_key_search_for_topK
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from cipher import present80 as present
from utils.cipher_utils import resolve_cipher_module, _USING_CUPY
from delta_key_search import run_delta_key_search_for_topK


def _find_latest_sweep_csv(base_dir: Path, cipher_name: str) -> Path | None:
    root = base_dir / cipher_name
    if not root.exists():
        return None
    candidates = list(root.glob("**/sweep_results.csv"))
    if not candidates:
        return None
    # Pick most recently modified
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(
        description="Run delta_key search using top-K entries from an existing sweep CSV."
    )
    parser.add_argument(
        "--cipher-module",
        default="cipher.present80",
        help="Dotted path to cipher module (default: cipher.present80)",
    )
    parser.add_argument("--nr", type=int, default=7, help="Number of rounds to test")
    parser.add_argument(
        "--pairs",
        type=int,
        default=1,
        help="Pairs per sample (feature dimension = pairs*3*plain_bits)",
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

    # Source selection
    parser.add_argument(
        "--sweep-csv",
        default=None,
        help="Path to an existing sweep_results.csv",
    )
    parser.add_argument(
        "--auto-latest",
        action="store_true",
        help="If --sweep-csv is not provided, auto-pick the latest sweep_results.csv under differences_findings/logs/<cipher>",
    )

    # Ranking and delta-key search controls
    parser.add_argument(
        "--top-k",
        type=int,
        required=True,
        help="Select top-K input differences to search best delta_key",
    )
    parser.add_argument(
        "--dk-metric",
        choices=["biased_pcs", "max_diff", "silhouette_clusters", "silhouette_true"],
        default="biased_pcs",
        help="Metric to rank input differences for delta_key search",
    )
    parser.add_argument("--dk-datasize", type=int, default=100000, help="n_samples for delta_key search")
    parser.add_argument("--dk-batch-size", type=int, default=5000, help="batch_size for delta_key search")

    args = parser.parse_args()

    # Resolve cipher module
    try:
        cipher_mod = resolve_cipher_module(args.cipher_module)
        cipher_name = args.cipher_module.split(".")[-1]
    except Exception:
        cipher_mod = present
        cipher_name = "present80"
        print(
            f"[warn] Cannot import '{args.cipher_module}', falling back to default '{cipher_name}'."
        )

    # GPU availability guard
    if args.use_gpu and not _USING_CUPY:
        print("[warn] CuPy not available; switching to CPU mode.")
        args.use_gpu = False

    # Prepare output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path("differences_findings/logs") / cipher_name / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load sweep CSV
    source_csv: Path | None = None
    if args.sweep_csv:
        source_csv = Path(args.sweep_csv)
        if not source_csv.exists():
            print(f"[error] Provided --sweep-csv not found: {source_csv}")
            return
    elif args.auto_latest:
        source_csv = _find_latest_sweep_csv(Path("differences_findings/logs"), cipher_name)
        if not source_csv:
            print("[error] --auto-latest could not find any sweep_results.csv. Provide --sweep-csv explicitly.")
            return
    else:
        print("[error] Provide --sweep-csv or use --auto-latest to locate a sweep CSV.")
        return

    print("\n" + "-" * 70)
    print(f"Loading sweep results from: {source_csv}")
    sweep_rows = []
    with open(source_csv, "r", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        for r in reader:
            try:
                row = {
                    "bit_pos": int(r.get("bit_pos", 0)),
                    "input_diff_hex": r.get("input_diff_hex", "").strip(),
                    "biased_pcs": float(r.get("biased_pcs", 0) or 0),
                    "max_diff": float(r.get("max_diff", 0) or 0),
                    "silhouette_clusters": float(r.get("silhouette_clusters", 0) or 0),
                    "silhouette_true": (
                        float(r.get("silhouette_true")) if r.get("silhouette_true") not in (None, "") else None
                    ),
                    "elapsed_sec": float(r.get("elapsed_sec", 0) or 0),
                }
                sweep_rows.append(row)
            except Exception:
                continue

    if not sweep_rows:
        print("[error] No valid rows parsed from sweep CSV; aborting delta-key run.")
        return

    print(f"Ranking by metric: {args.dk_metric} and selecting top-{args.top_k}...")

    def metric_value(row):
        val = row.get(args.dk_metric)
        return val if val is not None else float("-inf")

    ranked = sorted(sweep_rows, key=metric_value, reverse=True)
    picked = ranked[: args.top_k]

    # Save run configuration
    config = {
        "mode": "delta-key-only",
        "cipher_module": args.cipher_module,
        "resolved_cipher": cipher_name,
        "nr": args.nr,
        "pairs": args.pairs,
        "top_k": args.top_k,
        "dk_metric": args.dk_metric,
        "dk_datasize": args.dk_datasize,
        "dk_batch_size": args.dk_batch_size,
        "use_gpu": args.use_gpu,
        "sweep_csv": str(source_csv),
    }
    with open(out_dir / "config.delta_key.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    summary_csv = run_delta_key_search_for_topK(
        cipher_mod,
        picked,
        nr=args.nr,
        pairs=args.pairs,
        n_samples=args.dk_datasize,
        batch_size=args.dk_batch_size,
        use_gpu=args.use_gpu,
        out_dir=out_dir,
    )
    print(f"Delta-key search completed. Summary saved to: {summary_csv}")


if __name__ == "__main__":
    main()

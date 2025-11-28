import argparse
import importlib
import os
import logging
import time
import numpy as np
import optimizer
from analysis.pca_helper import compute_pca
from analysis.clustering_helper import kmeans_cluster, compute_silhouette
from utils.cipher_utils import resolve_cipher_module
from make_data_train import NDCMultiPairGenerator


def parse_args():
    ap = argparse.ArgumentParser("Find input differences using AutoND optimizer")
    ap.add_argument("--cipher", default="speck3264", help="cipher module under cipher/, e.g. speck3264")
    ap.add_argument("--scenario", choices=["single-key", "related-key"], default="single-key")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--pairs", type=int, default=1, help="số cặp (pairs) dùng khi tạo dữ liệu")
    ap.add_argument("--method", type=str, default="evo", choices=["evo", "pso", "ga", "de", "aco", "sa", "gwo"], help="optimizer method")
    ap.add_argument("--epsilon", type=float, default=0.1, help="bias threshold epsilon for optimizer")
    ap.add_argument("--out", default="", help="base output dir for optimizer logs (default: ./results)")
    ap.add_argument("--log-file", default="", help="optional path to write log output")
    ap.add_argument("--verbose", action="store_true", help="enable debug-level logging")
    # Optional: provide or sweep delta-key using a chosen difference
    ap.add_argument("--difference", default="", help="hex or int; if empty, use optimizer top difference")
    ap.add_argument("--sweep-dk", action="store_true", help="explicitly sweep delta-key single-bit positions and report scores")
    ap.add_argument("--dk-start", type=int, default=0, help="start bit index for delta-key sweep (inclusive)")
    ap.add_argument("--dk-end", type=int, default=-1, help="end bit index for delta-key sweep (inclusive); -1 means key_bits-1")
    ap.add_argument("--sweep-samples", type=int, default=50000, help="samples to use per delta-key bit during sweep")
    ap.add_argument("--kmeans-k", type=int, default=2, help="clusters for KMeans during sweep silhouette computation")
    return ap.parse_args()


def integer_to_binary_array(int_val, num_bits):
    return np.array([int(i) for i in bin(int_val)[2:].zfill(num_bits)], dtype=np.uint8).reshape(1, num_bits)


def _print_top_differences_details(diffs_hex_list, *, cipher_mod, scenario: str, top_k: int = 5):
    if not diffs_hex_list:
        return
    plain_bits = int(cipher_mod.plain_bits)
    key_bits = int(cipher_mod.key_bits)
    print("\nChi tiết Top differences (tách delta_plain / delta_key):")
    for i, hx in enumerate(diffs_hex_list[:top_k], start=1):
        try:
            diff_int = int(hx, 16)
        except Exception:
            try:
                diff_int = int(hx, 0)
            except Exception:
                print(f"  [{i}] {hx} -> không parse được")
                continue
        if scenario == "related-key":
            bits = integer_to_binary_array(diff_int, plain_bits + key_bits)
            dp = bits[:, :plain_bits]
            dk = bits[:, plain_bits:]
        else:
            dp = integer_to_binary_array(diff_int, plain_bits)
            dk = np.zeros((1, key_bits), dtype=np.uint8)

        dp_pos = np.where(dp[0] == 1)[0].tolist()
        dk_pos = np.where(dk[0] == 1)[0].tolist()
        print(f"  [{i}] {hx}")
        print(f"      delta_plain: HW={len(dp_pos)} | pos={dp_pos}")
        print(f"      delta_key  : HW={len(dk_pos)} | pos={dk_pos}")


def find_input_differences_with_cli(args, out_dir: str | None = None):
    """
    Find good input differences using the optimizer, wired to this script's CLI flags.

    Returns (best_differences, highest_round, log_prefix)
    """
    # Load cipher module (repo uses 'cipher.<name>')
    try:
        cipher = importlib.import_module(f"cipher.{args.cipher}")
    except ModuleNotFoundError:
        # Fallback if user provides full path or legacy 'ciphers.'
        try:
            cipher = importlib.import_module(args.cipher)
        except ModuleNotFoundError:
            # last attempt legacy
            cipher = importlib.import_module(f"ciphers.{args.cipher}")
    plain_bits = cipher.plain_bits
    key_bits = cipher.key_bits
    encrypt = cipher.encrypt

    # Prepare output/log path
    s_tag = f"{args.cipher}_{args.scenario}_r{args.rounds}"
    base_out = out_dir if out_dir else (args.out if args.out else os.path.join(os.path.dirname(__file__), "results"))
    os.makedirs(base_out, exist_ok=True)
    log_prefix = os.path.join(base_out, s_tag)

    logging.info(
        "Running optimizer | cipher=%s scenario=%s rounds=%d method=%s epsilon=%.6f log_prefix=%s",
        args.cipher, args.scenario, args.rounds, args.method, args.epsilon, log_prefix,
    )

    best_differences, highest_round = optimizer.optimize(
        plain_bits,
        key_bits,
        encrypt,
        scenario=args.scenario,
        log_file=log_prefix,
        epsilon=args.epsilon,
        method=args.method,
        rounds=args.rounds,
    )

    logging.info(
        "Optimizer finished | found=%d highest_round=%s top_diff=%s",
        len(best_differences) if best_differences is not None else 0,
        str(highest_round),
        (hex(int(best_differences[0])) if best_differences else "None"),
    )

    return best_differences, highest_round, log_prefix

def compute_deltas(diff_int: int, plain_bits: int, key_bits: int, scenario: str):
    if scenario == "related-key":
        delta = integer_to_binary_array(diff_int, plain_bits + key_bits)
        delta_key = delta[:, plain_bits:]
    else:
        delta = integer_to_binary_array(diff_int, plain_bits)
        delta_key = 0
    delta_plain = delta[:, :plain_bits]
    return delta_plain, delta_key


def sweep_delta_key(
    *,
    args,
    encrypt,
    plain_bits: int,
    key_bits: int,
    delta_plain: np.ndarray,
    base_out: str,
    diff_token: str,
):
    """
    Sweep single-bit delta_key positions, compute PCA(2) and silhouette over true labels.
    Saves a CSV summary and returns (best_bit, best_score, rows, sweep_dir, sweep_duration_seconds).
    """
    dk_start = max(0, int(args.dk_start))
    dk_end = key_bits - 1 if int(args.dk_end) < 0 else min(key_bits - 1, int(args.dk_end))
    if dk_start > dk_end:
        dk_start, dk_end = 0, key_bits - 1

    sweep_dir = os.path.join(base_out, "sweep_dk", f"{args.cipher}_r{args.rounds}_{diff_token}_{args.scenario}")
    os.makedirs(sweep_dir, exist_ok=True)
    csv_path = os.path.join(sweep_dir, "delta_key_sweep.csv")

    import time
    sweep_start = time.time()
    logging.info(
        "[SWEEP] Start delta-key sweep | cipher=%s scenario=%s rounds=%d range=[%d,%d] samples/bit=%d",
        args.cipher, args.scenario, args.rounds, dk_start, dk_end, args.sweep_samples,
    )

    records = []
    best_score = -1.0
    best_bit = -1

    for bit in range(dk_start, dk_end + 1):
        bit_start = time.time()
        dk = np.zeros((1, key_bits), dtype=np.uint8)
        dk[0, bit] = 1
        # Generate dataset for this delta-key
        Xs, Ys = make_train_data(
            encrypt,
            plain_bits,
            key_bits,
            args.sweep_samples,
            args.rounds,
            delta_state=delta_plain,
            delta_key=dk,
            pairs=args.pairs,
        )
        # PCA to 2D for silhouette
        proj2, evr2, _ = compute_pca(Xs.astype(np.float32), n_components=2)
        try:
            sil_true = compute_silhouette(proj2, Ys)
        except Exception:
            sil_true = float("nan")

        # Optionally also KMeans on 2D
        try:
            k_labels, k_inertia, _, _ = kmeans_cluster(proj2, n_clusters=args.kmeans_k)
            try:
                sil_k = compute_silhouette(proj2, k_labels)
            except Exception:
                sil_k = float("nan")
        except Exception:
            k_labels, k_inertia, sil_k = None, float("nan"), float("nan")

        evr_sum2 = float(np.sum(evr2))
        records.append({
            "bit": bit,
            "silhouette_true": float(sil_true),
            "silhouette_kmeans": float(sil_k),
            "kmeans_inertia": float(k_inertia),
            "evr_sum2": evr_sum2,
        })

        if isinstance(sil_true, float) and not np.isnan(sil_true) and sil_true > best_score:
            best_score = sil_true
            best_bit = bit

        bit_dur = time.time() - bit_start
        logging.debug(
            "[SWEEP] bit=%d | sil_true=%.5f sil_k=%.5f inertia=%.6f evr2=%.5f dur=%.3fs",
            bit,
            sil_true if isinstance(sil_true, float) else float("nan"),
            sil_k if isinstance(sil_k, float) else float("nan"),
            k_inertia if isinstance(k_inertia, float) else float("nan"),
            evr_sum2,
            bit_dur,
        )
        if (bit - dk_start) % 16 == 0:
            logging.info(
                "[SWEEP] progress bit=%d/%d | best_bit=%s best_sil=%.5f",
                bit, dk_end, (str(best_bit) if best_bit >= 0 else "-"),
                best_score if isinstance(best_score, float) else float("nan"),
            )

    # Save CSV
    try:
        import csv
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["bit", "silhouette_true", "silhouette_kmeans", "kmeans_inertia", "evr_sum2"])
            writer.writeheader()
            writer.writerows(records)
        logging.info("[SWEEP] CSV saved: %s (rows=%d)", csv_path, len(records))
    except Exception as e:
        logging.warning("[SWEEP] Failed to write CSV: %s", e)
    sweep_dur = time.time() - sweep_start
    logging.info(
        "[SWEEP] Done | best_bit=%s best_sil=%.5f dir=%s duration=%.3fs",
        (str(best_bit) if best_bit >= 0 else "-"),
        best_score if isinstance(best_score, float) else float("nan"),
        sweep_dir,
        sweep_dur,
    )
    return best_bit, best_score, records, sweep_dir, sweep_dur


def make_train_data(encryption_function, plain_bits, key_bits, n, nr, delta_state=0, delta_key=0, pairs=1):
    """
    Generate dataset using project's NDCMultiPairGenerator for consistency.
    Allows user to specify rounds (nr) and pairs.

    Returns (X, Y) where X are concatenated ciphertext pairs and Y in {0,1}.
    """
    gen = NDCMultiPairGenerator(
        encryption_function=encryption_function,
        plain_bits=plain_bits,
        key_bits=key_bits,
        nr=nr,
        delta_state=delta_state,
        delta_key=delta_key,
        n_samples=int(n),
        batch_size=int(n),
        pairs=int(pairs),
        use_gpu=True,
        to_float32=True,
    )
    X, Y = gen[0]
    return X, Y



def main():
    args = parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_fmt = "[%(asctime)s] %(levelname)s: %(message)s"
    if args.log_file:
        logging.basicConfig(level=log_level, format=log_fmt, handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.log_file, encoding="utf-8")
        ])
    else:
        logging.basicConfig(level=log_level, format=log_fmt)

    # Log cipher summary
    try:
        cipher = importlib.import_module(f"cipher.{args.cipher}")
    except ModuleNotFoundError:
        try:
            cipher = importlib.import_module(args.cipher)
        except ModuleNotFoundError:
            cipher = importlib.import_module(f"ciphers.{args.cipher}")
    logging.info("Loaded cipher.%s | plain_bits=%d key_bits=%d", args.cipher, cipher.plain_bits, cipher.key_bits)

    _t0_opt = time.time()
    best_differences, highest_round, log_prefix = find_input_differences_with_cli(args)
    _t1_opt = time.time()
    _opt_secs = _t1_opt - _t0_opt
    diffs_hex = [hex(int(x)) for x in (best_differences or [])]

    print("\n=== Input Difference Search Result ===")
    print(f"Cipher: {args.cipher} | Scenario: {args.scenario} | Rounds: {args.rounds}")
    print(f"Method: {args.method} | Epsilon: {args.epsilon}")
    print(f"Top differences: {diffs_hex[:10]}")
    print(f"Highest round: {highest_round}")
    print(f"Optimizer logs at: {log_prefix}*")
    print(f"Summary: rounds={args.rounds} | highest_non_random_round={highest_round} | optimizer_time={_opt_secs:.3f}s")
    # In ra chi tiết delta_plain / delta_key cho các khác biệt tốt đầu tiên
    _print_top_differences_details(diffs_hex, cipher_mod=cipher, scenario=args.scenario, top_k=5)

    # Optional delta-key sweep (use provided difference or the top optimizer difference)
    if args.sweep_dk:
        # Load cipher details
        plain_bits = cipher.plain_bits
        key_bits = cipher.key_bits
        encrypt = cipher.encrypt

        # Determine difference integer
        if str(args.difference).strip():
            try:
                diff_int = int(args.difference, 0)
            except Exception:
                diff_int = int(args.difference)
            diff_token = (args.difference if isinstance(args.difference, str) else str(args.difference)).lower().strip()
        else:
            diff_int = int(best_differences[0]) if best_differences else 1
            diff_token = hex(diff_int)

        delta_plain, _delta_key = compute_deltas(diff_int, plain_bits, key_bits, args.scenario)
        base_out_for_sweep = args.out if args.out else os.path.join(os.path.dirname(__file__), "analysis_results")

        best_bit, best_score, records, sweep_dir, sweep_secs = sweep_delta_key(
            args=args,
            encrypt=encrypt,
            plain_bits=plain_bits,
            key_bits=key_bits,
            delta_plain=delta_plain,
            base_out=base_out_for_sweep,
            diff_token=(diff_token if isinstance(diff_token, str) else str(diff_token)),
        )
        print("\n=== Delta-Key Sweep Result ===")
        print(f"Best bit: {best_bit}  | silhouette_true={best_score:.5f}")
        print(f"Sweep CSV: {os.path.join(sweep_dir, 'delta_key_sweep.csv')}")
        print(f"Summary: rounds={args.rounds} | pairs={args.pairs} | sweep_time={sweep_secs:.3f}s")


if __name__ == "__main__":
    main()

# --- Optional utilities mirrored from main_pcaClustering.py ---

# python main_findingDiffereces.py --cipher speck3264 --scenario single-key --rounds 8 --verbose
# python main_findingDiffereces.py --cipher speck3264 --scenario related-key --rounds 8 --sweep-dk --dk-start 0 --dk-end 64 --sweep-samples 5000 --verbose

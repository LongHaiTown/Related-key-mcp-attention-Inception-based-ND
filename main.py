from train_nets import (
    update_checkpoint_in_callbacks, select_best_delta_key,
    integer_to_binary_array, NDCMultiPairGenerator, make_model_inception, callbacks
)
import argparse
import importlib
import numpy as np
from train_nets import (
    update_checkpoint_in_callbacks, select_best_delta_key,
    integer_to_binary_array, NDCMultiPairGenerator, make_model_inception, callbacks
)
import argparse
import importlib
import numpy as np
import tensorflow as tf
import os
import json
import datetime as dt
from eval_nets import evaluate_with_statistics
import csv
from pathlib import Path
from typing import Tuple


def parse_args():
    parser = argparse.ArgumentParser(description="Neural distinguisher training")
    parser.add_argument(
        "--cipher",
        type=str,
        default=os.getenv("CIPHER_NAME", "present80"),
        help="Cipher module under 'cipher' package (e.g., present80, simon3264, speck3264, speck64128, simmeck3264, simmeck4896)"
    )
    parser.add_argument(
        "--rounds", "-r",
        type=int,
        default=int(os.getenv("CIPHER_ROUNDS", 7)),
        help="Number of cipher rounds to use (default: 7)"
    )
    parser.add_argument(
        "--pairs", "-p",
        type=int,
        default=int(os.getenv("PAIRS", 8)),
        help="Number of plaintext-ciphertext pairs per sample (default: 8)"
    )
    # Optional: choose input_difference from sweep CSV
    parser.add_argument(
        "--input-diff",
        type=str,
        default=os.getenv("INPUT_DIFF", "0x00000080"),
        help="Input difference hex (e.g., 0x80). Ignored when --sweep-csv is provided."
    )
    parser.add_argument(
        "--sweep-csv",
        type=str,
        default=os.getenv("SWEEP_CSV", None),
        help="Path to sweep_results.csv to auto-pick best input difference."
    )
    # Multi-round sweep parent (directories with nr<round>/sweep_results.csv)
    parser.add_argument(
        "--sweep-parent",
        type=str,
        default=os.getenv("SWEEP_PARENT", None),
        help="Path to a parent sweep directory containing per-round subfolders (nr<round>)."
    )
    parser.add_argument(
        "--auto-latest-sweep",
        action="store_true",
        help="Use the latest sweep run under differences_findings/logs/<cipher> (multi-round)."
    )
    parser.add_argument(
        "--diff-metric",
        type=str,
        choices=["biased_pcs", "max_diff", "silhouette_clusters", "silhouette_true"],
        default=os.getenv("DIFF_METRIC", "biased_pcs"),
        help="Metric to rank input differences when --sweep-csv is provided."
    )
    parser.add_argument(
        "--delta-key-bit",
        type=int,
        default=os.getenv("DELTA_KEY_BIT", None),
        help="Manually specify delta key bit index to use (skip automatic delta-key search)."
    )
    parser.add_argument(
        "--difference",
        type=str,
        default=os.getenv("DIFFERENCE", None),
        help="Optional hex/int difference. If bit-length > plain_bits or --combined-diff set, treated as concatenated (plain_bits + key_bits)."
    )
    parser.add_argument(
        "--combined-diff",
        action="store_true",
        help="Force --difference to be interpreted as combined (plain_bits + key_bits)."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(os.getenv("CHUNK_SIZE", 10**4)),
        help="Chunk size for training-by-chunks (default: 10000)"
    )
    parser.add_argument(
        "--num-samples-train",
        type=int,
        default=int(os.getenv("NUM_SAMPLES_TRAIN", 10**6)),
        help="Total number of training samples (default: 1_000_000)"
    )
    args, _ = parser.parse_known_args()
    return args


def import_cipher_module(cipher_name: str):
    try:
        return importlib.import_module(f"cipher.{cipher_name}")
    except Exception as e:
        raise RuntimeError(f"Failed to import cipher module 'cipher.{cipher_name}': {e}")


def _metric_value(row: dict, key: str) -> float:
    v = row.get(key, "")
    if v in ("", None):
        return float("-inf")
    try:
        return float(v)
    except Exception:
        return float("-inf")


def pick_best_input_diff_from_csv(csv_path: Path, metric: str) -> Tuple[int, int]:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    best = max(rows, key=lambda r: _metric_value(r, metric))
    hex_str = best.get("input_diff_hex", "0x0").strip()
    bit_pos = int(best.get("bit_pos", "-1"))
    input_diff_int = int(hex_str, 16)
    return bit_pos, input_diff_int


def _get_best_row_from_csv(csv_path: Path, metric: str):
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    best = max(rows, key=lambda r: _metric_value(r, metric))
    return best, _metric_value(best, metric)


def _find_latest_sweep_parent(cipher_name: str) -> Path:
    base = Path("differences_findings") / "logs" / cipher_name
    if not base.exists():
        raise FileNotFoundError(f"Base sweep directory not found: {base}")
    # Timestamped directories sort lexicographically in chronological order
    candidates = [p for p in base.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No sweep runs found under: {base}")
    latest = sorted(candidates, key=lambda p: p.name)[-1]
    return latest


def pick_best_round_and_input_diff(parent_dir: Path, metric: str) -> Tuple[int, int, int, Path]:
    """
    Aggregate per-round sweep CSVs under parent_dir to choose the best round and input difference.
    Returns (best_nr, best_bit_pos, best_input_diff_int, csv_path_for_best_round)
    """
    # Discover per-round subfolders
    subdirs = [d for d in parent_dir.iterdir() if d.is_dir() and d.name.lower().startswith("nr")]

    # If no subdirs, treat parent as a single-round sweep
    if not subdirs:
        csv_path = parent_dir / "sweep_results.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"sweep_results.csv not found in {parent_dir}")
        # Read nr from config.json
        cfg_path = parent_dir / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"config.json not found in {parent_dir}")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        best_row, _ = _get_best_row_from_csv(csv_path, metric)
        bit_pos = int(best_row.get("bit_pos", "-1"))
        input_diff_int = int(best_row.get("input_diff_hex", "0x0"), 16)
        return int(cfg.get("nr", -1)), bit_pos, input_diff_int, csv_path

    # Iterate per-round subfolders
    global_best_val = float("-inf")
    best_nr = None
    best_bit_pos = None
    best_input_diff = None
    best_csv_path = None

    for sd in subdirs:
        # Determine rounds from folder name or config.json as fallback
        nr_val = None
        name = sd.name.lower()
        if name.startswith("nr"):
            try:
                nr_val = int(name[2:])
            except Exception:
                nr_val = None
        if nr_val is None:
            cfgp = sd / "config.json"
            if cfgp.exists():
                with open(cfgp, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    nr_val = int(cfg.get("nr", -1))

        csv_path = sd / "sweep_results.csv"
        if not csv_path.exists():
            continue

        try:
            row, val = _get_best_row_from_csv(csv_path, metric)
        except Exception:
            continue

        if val > global_best_val:
            global_best_val = val
            best_nr = nr_val
            best_bit_pos = int(row.get("bit_pos", "-1"))
            best_input_diff = int(row.get("input_diff_hex", "0x0"), 16)
            best_csv_path = csv_path

    if best_nr is None:
        raise ValueError(f"No valid sweep CSVs found under {parent_dir}")

    return best_nr, best_bit_pos, best_input_diff, best_csv_path


def choose_input_difference(args) -> int:
    if args.sweep_csv:
        bit_pos_best, input_difference = pick_best_input_diff_from_csv(Path(args.sweep_csv), args.diff_metric)
        print(
            f"[auto] Picked best input difference from sweep ({args.diff_metric}): "
            f"bit_pos={bit_pos_best}, hex=0x{input_difference:016X}"
        )
        return input_difference
    return int(str(args.input_diff), 16)


def _int_to_bits(int_val: int, num_bits: int) -> np.ndarray:
    return np.array([int(b) for b in bin(int_val)[2:].zfill(num_bits)], dtype=np.uint8)


def _split_combined_difference(diff_int: int, plain_bits: int, key_bits: int):
    bits = _int_to_bits(diff_int, plain_bits + key_bits)
    delta_plain_bits = bits[:plain_bits]
    delta_key_bits = bits[plain_bits:]
    plain_int = int(''.join(str(x) for x in delta_plain_bits.tolist()), 2)
    return plain_int, delta_plain_bits, delta_key_bits


def choose_delta_key(encrypt, plain_bits: int, key_bits: int, n_round: int, pairs: int, input_difference: int):
    best_bit, best_score, all_scores = select_best_delta_key(
        encryption_function=encrypt,
        input_difference=input_difference,
        plain_bits=plain_bits,
        key_bits=key_bits,
        n_round=n_round,
        pairs=pairs,
        use_gpu=True,
    )
    delta_plain = integer_to_binary_array(input_difference, plain_bits)
    delta_key = np.zeros(key_bits, dtype=np.uint8)
    delta_key[best_bit] = 1
    return best_bit, best_score, delta_plain, delta_key


def make_generators(encrypt, plain_bits: int, key_bits: int, n_round: int, pairs: int,
                    delta_plain, delta_key, train_samples: int, val_samples: int,
                    batch_size: int, val_batch_size: int):
    gen = NDCMultiPairGenerator(
        encryption_function=encrypt,
        plain_bits=plain_bits, key_bits=key_bits, nr=n_round,
        delta_state=delta_plain,
        delta_key=delta_key,
        pairs=pairs,
        n_samples=train_samples, batch_size=batch_size,
        use_gpu=True,
    )
    gen_val = NDCMultiPairGenerator(
        encryption_function=encrypt,
        plain_bits=plain_bits, key_bits=key_bits, nr=n_round,
        delta_state=delta_plain,
        delta_key=delta_key,
        pairs=pairs,
        n_samples=val_samples, batch_size=val_batch_size,
        use_gpu=True,
    )
    return gen, gen_val


def build_and_train_model(gen, gen_val, pairs: int, plain_bits: int, cb, epochs: int, batch_size: int):
    X_train, Y_train = gen[0]
    print("Sample training data shapes:", X_train.shape, Y_train.shape)
    model = make_model_inception(pairs=pairs, plain_bits=plain_bits)
    optimizer = tf.keras.optimizers.Adam(amsgrad=True)
    model.compile(optimizer=optimizer, loss='mse', metrics=['acc'])
    history = model.fit(
        gen,
        epochs=epochs,
        validation_data=gen_val,
        batch_size=batch_size,
        callbacks=cb,
        verbose=True,
    )
    return model, history
def train_by_chunks(
    model,
    encrypt,
    plain_bits,
    key_bits,
    n_round,
    pairs,
    delta_plain,
    delta_key,
    total_samples,
    chunk_size,
    batch_size,
    epochs,
    val_samples=1_000_000,
    patience=3,
    callbacks=None,
):

    n_chunks = (total_samples + chunk_size - 1) // chunk_size

    from types import SimpleNamespace
    history_acc = {}

    # fixed validation generator
    val_gen = NDCMultiPairGenerator(
        encrypt,
        plain_bits,
        key_bits,
        n_round,
        delta_state=delta_plain,
        delta_key=delta_key,
        n_samples=val_samples,
        batch_size=batch_size,
        pairs=pairs,
    )

    best_val_loss = float("inf")
    wait = 0

    for epoch in range(epochs):
        print(f"\n========== Global Epoch {epoch+1}/{epochs} ==========")

        # ---- training ----
        for c in range(n_chunks):
            print(f"--- Chunk {c+1}/{n_chunks} ---")

            gen_chunk = NDCMultiPairGenerator(
                encrypt,
                plain_bits,
                key_bits,
                n_round,
                delta_state=delta_plain,
                delta_key=delta_key,
                n_samples=chunk_size,
                batch_size=batch_size,
                pairs=pairs,
                start_idx=c * chunk_size,
            )

            h = model.fit(
                gen_chunk,
                epochs=1,
                callbacks=callbacks,
                verbose=1,
            )

            if hasattr(h, "history"):
                for k, v in h.history.items():
                    history_acc.setdefault(k, []).extend(v)

        # ---- validation ----
        val_loss, val_acc = model.evaluate(val_gen, verbose=0)
        print(f"[Validation] loss={val_loss:.5f}, acc={val_acc:.5f}")

        history_acc.setdefault("val_loss", []).append(val_loss)
        history_acc.setdefault("val_acc", []).append(val_acc)

        # ---- early stopping ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            print("✓ Validation improved")
        else:
            wait += 1
            print(f"✗ No improvement (patience {wait}/{patience})")

        if wait >= patience:
            print("Early stopping triggered (global epoch level)")
            break

    return SimpleNamespace(
        history=history_acc,
        epochs_completed=epoch + 1,
        n_chunks=n_chunks,
    )


def save_artifacts(model, history, cipher_name: str, n_round: int, run_id: str):
    ckpt_dir = os.path.join("checkpoints", cipher_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    weights_path = os.path.join(ckpt_dir, f"{cipher_name}_final_{n_round}r.weights.h5")
    model.save_weights(weights_path)
    arch_path = os.path.join(ckpt_dir, f"{cipher_name}_final_{n_round}r_architecture.json")
    with open(arch_path, 'w') as f:
        f.write(model.to_json())
    print(f"Saved final model weights: {weights_path}")
    print(f"Saved model architecture: {arch_path}")

    final_path = os.path.join(ckpt_dir, f"{cipher_name}_final_{n_round}r.keras")
    model.save(final_path)
    log_dir = os.path.join("logs", cipher_name, run_id)
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, f"history_{n_round}r.json"), "w", encoding="utf-8") as f:
        json.dump(history.history, f, ensure_ascii=False, indent=2)


def evaluate_model(model, encrypt, plain_bits: int, key_bits: int, input_difference: int, delta_key, pairs: int, n_round: int):
    stats = evaluate_with_statistics(
        model,
        round_number=n_round,
        encryption_function=encrypt,
        plain_bits=plain_bits,
        key_bits=key_bits,
        input_difference=input_difference,
        delta_key=delta_key,
        pairs=pairs,
    )
    print("Evaluation statistics:", stats)
    return stats


def run():
    args = parse_args()
    cipher = import_cipher_module(args.cipher)

    # Rounds / pairs validation
    n_round = int(args.rounds)
    pairs = int(args.pairs)
    if n_round <= 0:
        raise ValueError("--rounds must be a positive integer")
    if pairs <= 0:
        raise ValueError("--pairs must be a positive integer")

    plain_bits = cipher.plain_bits
    key_bits = cipher.key_bits
    encrypt = cipher.encrypt
    cipher_name = getattr(cipher, "cipher_name", args.cipher)
    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Constants (tune as needed)
    BATCH_SIZE = 5000
    VAL_BATCH_SIZE = 20000
    EPOCHS = 2
    # `NUM_SAMPLES_TRAIN` and `CHUNK_SIZE` can be overridden by CLI flags
    NUM_SAMPLES_TRAIN = int(args.num_samples_train)
    NUM_SAMPLES_TEST = 10**5

    # Callbacks
    cb = update_checkpoint_in_callbacks(callbacks, rounds=n_round, cipher_name=cipher_name, run_id=run_id)

    # === Input difference handling (extended) ===
    input_difference = None
    combined_from_user = False
    user_delta_plain = None
    user_delta_key = None

    if args.difference:
        # Direct difference provided
        try:
            user_diff_int = int(args.difference, 0)
        except Exception:
            user_diff_int = int(args.difference)
        if args.combined_diff or user_diff_int.bit_length() > plain_bits:
            combined_from_user = True
            input_difference, d_plain_bits, d_key_bits = _split_combined_difference(
                user_diff_int, plain_bits, key_bits
            )
            # Shape to (plain_bits,) & (key_bits,)
            user_delta_plain = d_plain_bits.astype(np.uint8)
            user_delta_key = d_key_bits.astype(np.uint8)
            print(f"[difference] Using combined difference (plain+key): 0x{user_diff_int:X}")
            print(f"            delta_plain HW={int(user_delta_plain.sum())} bits={[i for i,v in enumerate(user_delta_plain) if v]}")
            print(f"            delta_key  HW={int(user_delta_key.sum())} bits={[i for i,v in enumerate(user_delta_key) if v]}")
        else:
            input_difference = user_diff_int
            print(f"[difference] Using provided plain input difference: 0x{input_difference:X}")
    else:
        # Existing sweep / manual logic
        if args.sweep_parent or args.auto_latest_sweep:
            parent_dir = Path(args.sweep_parent) if args.sweep_parent else _find_latest_sweep_parent(cipher_name)
            best_nr, best_bit_pos, best_input_difference, best_csv = pick_best_round_and_input_diff(parent_dir, args.diff_metric)
            print(
                f"[auto] Picked best round and input difference from multi-sweep ({args.diff_metric}): "
                f"nr={best_nr}, bit_pos={best_bit_pos}, hex=0x{best_input_difference:016X}\n"
                f"       source CSV: {best_csv}"
            )
            n_round = int(best_nr)
            input_difference = int(best_input_difference)
        else:
            input_difference = choose_input_difference(args)

    # === Delta-key determination ===
    if combined_from_user:
        # Build arrays from user supplied combined diff
        delta_plain = user_delta_plain
        delta_key = user_delta_key
        # Override delta-key if --delta-key-bit passed
        if args.delta_key_bit is not None:
            mb = int(args.delta_key_bit)
            if mb < 0 or mb >= key_bits:
                raise ValueError(f"--delta-key-bit {mb} out of range (0..{key_bits-1})")
            delta_key = np.zeros(key_bits, dtype=np.uint8)
            delta_key[mb] = 1
            print(f"[override] Replacing combined delta_key with manual bit={mb}")
        best_bit = [i for i,v in enumerate(delta_key) if v]
        best_score = None
    else:
        if args.delta_key_bit is not None:
            manual_bit = int(args.delta_key_bit)
            if manual_bit < 0 or manual_bit >= key_bits:
                raise ValueError(f"--delta-key-bit {manual_bit} out of range (0..{key_bits-1})")
            delta_plain = integer_to_binary_array(input_difference, plain_bits)
            delta_key = np.zeros(key_bits, dtype=np.uint8)
            delta_key[manual_bit] = 1
            best_bit = manual_bit
            best_score = None
            print(f"[manual] Using provided delta key bit: {best_bit}")
        else:
            best_bit, best_score, delta_plain, delta_key = choose_delta_key(
                encrypt, plain_bits, key_bits, n_round, pairs, input_difference
            )

    # Ensure delta arrays are 1D uint8 for generator
    if delta_plain.ndim > 1:
        delta_plain = delta_plain.reshape(-1)
    if delta_key.ndim > 1:
        delta_key = delta_key.reshape(-1)

    # # Data generators
    # gen, gen_val = make_generators(
    #     encrypt, plain_bits, key_bits, n_round, pairs,
    #     delta_plain, delta_key, NUM_SAMPLES_TRAIN, NUM_SAMPLES_TEST,
    #     BATCH_SIZE, VAL_BATCH_SIZE,
    # )

    # # Train
    # model, history = build_and_train_model(
    #     gen, gen_val, pairs, plain_bits, cb, EPOCHS, BATCH_SIZE
    # )

    TOTAL_SAMPLES = NUM_SAMPLES_TRAIN
    CHUNK_SIZE    = int(args.chunk_size)
    EPOCHS_PER_CHUNK = 1

    model = make_model_inception(pairs=pairs, plain_bits=plain_bits)
    optimizer = tf.keras.optimizers.Adam(amsgrad=True)
    model.compile(optimizer=optimizer, loss='mse', metrics=['acc'])

    history = train_by_chunks(
        model,
        encrypt,
        plain_bits,
        key_bits,
        n_round,
        pairs,
        delta_plain,
        delta_key,
        TOTAL_SAMPLES,
        CHUNK_SIZE,
        BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=cb,
    )


    # Save artifacts
    save_artifacts(model, history, cipher_name, n_round, run_id)

    # Evaluate
    evaluate_model(model, encrypt, plain_bits, key_bits, input_difference, delta_key, pairs, n_round)


# python finding_input.py --cipher-module cipher.present80 --nr 7 --pairs 1 --datasize 50000 --clusters 27 --max-bits 64

# python main.py --cipher present80 --rounds 7 --pairs 8 --sweep-csv "differences_findings\logs\present80\20251112-145640\sweep_results.csv" --diff-metric biased_pcs

# python main.py --cipher present80 --rounds 7 --pairs 8 

# python main.py --cipher present80 --rounds 7 --pairs 8 --input-diff 0x00000080

# Manual delta-key bit example (skip automatic search):
# python main.py --cipher present80 --rounds 7 --pairs 8 --input-diff 0x00000080 --delta-key-bit 107

# Multi-round auto selection examples:
# python main.py --cipher present80 --auto-latest-sweep --pairs 8 --diff-metric biased_pcs
# python main.py --cipher present80 --sweep-parent "differences_findings\logs\present80\20251113-101500" --pairs 8 --diff-metric max_diff

if __name__ == "__main__":
    run()
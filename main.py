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
    parser.add_argument(
        "--diff-metric",
        type=str,
        choices=["biased_pcs", "max_diff", "silhouette_clusters", "silhouette_true"],
        default=os.getenv("DIFF_METRIC", "biased_pcs"),
        help="Metric to rank input differences when --sweep-csv is provided."
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


def choose_input_difference(args) -> int:
    if args.sweep_csv:
        bit_pos_best, input_difference = pick_best_input_diff_from_csv(Path(args.sweep_csv), args.diff_metric)
        print(
            f"[auto] Picked best input difference from sweep ({args.diff_metric}): "
            f"bit_pos={bit_pos_best}, hex=0x{input_difference:016X}"
        )
        return input_difference
    return int(str(args.input_diff), 16)


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
    NUM_SAMPLES_TRAIN = 10**6
    NUM_SAMPLES_TEST = 10**5

    # Callbacks
    cb = update_checkpoint_in_callbacks(callbacks, rounds=n_round, cipher_name=cipher_name, run_id=run_id)

    # Input difference selection
    input_difference = choose_input_difference(args)

    # Delta-key selection
    best_bit, best_score, delta_plain, delta_key = choose_delta_key(
        encrypt, plain_bits, key_bits, n_round, pairs, input_difference
    )

    # Data generators
    gen, gen_val = make_generators(
        encrypt, plain_bits, key_bits, n_round, pairs,
        delta_plain, delta_key, NUM_SAMPLES_TRAIN, NUM_SAMPLES_TEST,
        BATCH_SIZE, VAL_BATCH_SIZE,
    )

    # Train
    model, history = build_and_train_model(
        gen, gen_val, pairs, plain_bits, cb, EPOCHS, BATCH_SIZE
    )

    # Save artifacts
    save_artifacts(model, history, cipher_name, n_round, run_id)

    # Evaluate
    evaluate_model(model, encrypt, plain_bits, key_bits, input_difference, delta_key, pairs, n_round)


# python finding_input.py --cipher-module cipher.present80 --nr 7 --pairs 1 --datasize 50000 --clusters 27 --max-bits 64

# python main.py --cipher present80 --rounds 7 --pairs 8 --sweep-csv "differences_findings\logs\present80\20251112-145640\sweep_results.csv" --diff-metric biased_pcs
# python main.py --cipher present80 --rounds 7 --pairs 8 
# python main.py ^
#   --cipher present80 ^
#   --rounds 7 ^
#   --pairs 8 ^
#   --input-diff 0x00000080

if __name__ == "__main__":
    run()
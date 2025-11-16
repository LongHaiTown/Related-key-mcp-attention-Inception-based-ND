import argparse
from pathlib import Path
import numpy as np
from scipy.stats import norm
from tqdm import tqdm
import tensorflow as tf
from make_data_train import NDCMultiPairGenerator
from train_nets import integer_to_binary_array
import importlib

def evaluate_with_statistics(
    model,
    round_number,
    n_repeat=20,
    log_path=None,
    encryption_function=None,
    plain_bits=64,
    key_bits=80,
    input_difference=None,
    delta_key=None,
    pairs=8,
    test_samples: int = 1_000_000,
    batch_size: int = 10_000,
    use_gpu: bool = True,
):
    """
    Evaluate the model multiple times and compute statistics to check if the result is statistically significant.

    Parameters:
        model: Trained model.
        round_number: Number of PRESENT rounds for test data generation.
        n_repeat: Number of test set generations and evaluations (recommended: 20–30).
        log_path: If provided, save log results to this file.
        encryption_function: Cipher encryption function.
        plain_bits: Number of bits in plaintext.
        key_bits: Number of bits in key.
        input_difference: Input difference (int or array).
        delta_key: Key difference (array).
        pairs: Number of pairs per sample.

    Returns:
        dict: Contains avg_acc, std_acc, z_score, p_value.
    """
    print(f"\n📊 Evaluating model on {n_repeat} fresh test sets for round {round_number}...")

    if encryption_function is None or input_difference is None or delta_key is None:
        raise ValueError("encryption_function, input_difference, and delta_key must be provided.")

    test_accs = []
    for i in tqdm(range(n_repeat)):
        test_gen = NDCMultiPairGenerator(
            encryption_function=encryption_function,
            plain_bits=plain_bits,
            key_bits=key_bits,
            nr=round_number,
            delta_state=integer_to_binary_array(input_difference, plain_bits),
            delta_key=delta_key,
            n_samples=test_samples,
            batch_size=batch_size,
            pairs=pairs,
            use_gpu=use_gpu,
        )
        _, acc = model.evaluate(test_gen, verbose=0)
        test_accs.append(acc)

    # Compute statistics
    accs = np.array(test_accs)
    avg_acc = np.mean(accs)
    std_acc = np.std(accs)

    # Z-score: deviation from random guessing (50%)
    mean_random = 0.5
    std_random = 0.0005  # For each test set of 1 million samples
    std_mean = std_random / np.sqrt(n_repeat)
    z_score = (avg_acc - mean_random) / std_mean
    p_value = 1 - norm.cdf(z_score)

    print(f"\n✅ Average Accuracy: {avg_acc:.5f} ± {std_acc:.5f}")
    print(f"📐 Z-score = {z_score:.2f},  P-value = {p_value:.4e}")
    if p_value < 0.01:
        print("✨ Statistically significant improvement over random.")
    else:
        print("⚠️  Accuracy may still be due to random guessing.")

    # Save log if needed
    if log_path:
        with open(log_path, 'w') as f:
            for i, acc in enumerate(accs):
                f.write(f"Test {i+1}: {acc:.6f}\n")
            f.write(f"\nAverage: {avg_acc:.6f}, Std: {std_acc:.6f}\n")
            f.write(f"Z-score: {z_score:.2f}, P-value: {p_value:.4e}\n")

    return {
        'avg_acc': avg_acc,
        'std_acc': std_acc,
        'z_score': z_score,
        'p_value': p_value
    }


def _parse_delta_key_from_hex(hex_str: str, key_bits: int) -> np.ndarray:
    mask = int(hex_str, 16)
    arr = np.zeros(key_bits, dtype=np.uint8)
    for i in range(key_bits):
        arr[i] = (mask >> i) & 1
    return arr


def _import_cipher_module(module_path: str):
    try:
        return importlib.import_module(module_path)
    except Exception as e:
        raise RuntimeError(f"Failed to import cipher module '{module_path}': {e}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained distinguisher with statistical testing")
    parser.add_argument(
        "--cipher-module",
        type=str,
        default="cipher.present80",
        help="Dotted path to cipher module (e.g., cipher.present80)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model: .keras (full model) or .h5/.hdf5. If it's a weights file like *.weights.h5, the script rebuilds the model via RKmcp.make_model_inception(pairs, plain_bits) and loads weights.",
    )
    parser.add_argument("--rounds", "-r", type=int, default=7, help="Number of cipher rounds for evaluation")
    parser.add_argument("--pairs", "-p", type=int, default=8, help="Pairs per sample")
    parser.add_argument("--input-diff", type=str, default="0x00000080", help="Input difference hex (e.g., 0x80)")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--delta-key-bit",
        type=int,
        help="Delta key bit index (0-based). If omitted and --delta-key-hex not provided, uses delta_key=0 (no related-key).",
    )
    group.add_argument(
        "--delta-key-hex",
        type=str,
        help="Delta key bitmask as hex (e.g., 0x00000000000000000001). If omitted and --delta-key-bit not provided, uses delta_key=0 (no related-key).",
    )
    parser.add_argument("--n-repeat", type=int, default=20, help="Number of repeated test evaluations")
    parser.add_argument("--test-samples", type=int, default=1_000_000, help="Samples per test set")
    parser.add_argument("--batch-size", type=int, default=10_000, help="Batch size for test generator")
    # Always use GPU path in generator (if available); no CPU fallback flag
    parser.add_argument("--log-path", type=str, default=None, help="Optional path to save evaluation log")

    args = parser.parse_args()

    # Load cipher
    cipher_mod = _import_cipher_module(args.cipher_module)
    encrypt = cipher_mod.encrypt
    plain_bits = cipher_mod.plain_bits
    key_bits = cipher_mod.key_bits

    # Prepare deltas
    input_difference = int(args.input_diff, 16)
    if args.delta_key_bit is not None:
        if args.delta_key_bit < 0 or args.delta_key_bit >= key_bits:
            raise ValueError(f"--delta-key-bit out of range (0..{key_bits-1})")
        delta_key = np.zeros(key_bits, dtype=np.uint8)
        delta_key[int(args.delta_key_bit)] = 1
    elif args.delta_key_hex:
        delta_key = _parse_delta_key_from_hex(args.delta_key_hex, key_bits)
    else:
        # Default: no related-key difference
        delta_key = np.zeros(key_bits, dtype=np.uint8)
        print("[info] No --delta-key-bit/--delta-key-hex provided; using delta_key = 0 (no related-key).")

    # Load model
    mp = Path(args.model_path)
    if not mp.exists():
        raise FileNotFoundError(f"Model path not found: {mp}")

    suffix = mp.suffix.lower()
    model = None
    if suffix == ".keras":
        model = tf.keras.models.load_model(str(mp))
    elif suffix in (".h5", ".hdf5"):
        # Case 1: full model saved in H5
        try:
            model = tf.keras.models.load_model(str(mp))
        except Exception:
            # Case 2: weights-only H5 — rebuild model from RKmcp.make_model_inception
            try:
                from RKmcp import make_model_inception
            except Exception as e:
                raise RuntimeError("Failed to import make_model_inception from RKmcp: " + str(e))
            model = make_model_inception(pairs=int(args.pairs), plain_bits=plain_bits)
            model.load_weights(str(mp))
    else:
        raise ValueError(f"Unsupported model file type: {suffix}. Use .keras or .h5/.hdf5")
    
    if model.compiled_loss is None:
        print("[info] Compiling model for evaluation (binary_crossentropy, accuracy).")
        model.compile(
            loss="binary_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            metrics=["accuracy"],
        )

    # Run evaluation
    stats = evaluate_with_statistics(
        model,
        round_number=int(args.rounds),
        n_repeat=int(args.n_repeat),
        log_path=args.log_path,
        encryption_function=encrypt,
        plain_bits=plain_bits,
        key_bits=key_bits,
        input_difference=input_difference,
        delta_key=delta_key,
        pairs=int(args.pairs),
        test_samples=int(args.test_samples),
        batch_size=int(args.batch_size),
        use_gpu=True,
    )

    print("\nSummary:", stats)


if __name__ == "__main__":
    main()

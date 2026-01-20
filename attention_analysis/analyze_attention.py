import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from train_nets import integer_to_binary_array
from make_data_train import NDCMultiPairGenerator
import importlib
from scipy.stats import entropy
from datetime import datetime


# ==============================
#  Utilities
# ==============================
def _import_cipher_module(module_path: str):
    return importlib.import_module(module_path)


def _parse_delta_key_from_hex(hex_str: str, key_bits: int):
    mask = int(hex_str, 16)
    arr = np.zeros(key_bits, dtype=np.uint8)
    for i in range(key_bits):
        arr[i] = (mask >> i) & 1
    return arr


# ==============================
#  ECA Attention Utilities
# ==============================
def find_eca_activation_layer(model):
    """
    Heuristically locate the ECA sigmoid activation layer.
    Priority:
      1) name contains "eca_sigmoid"
      2) Activation with shape (None, C, 1)
      3) Conv1D(filters=1) followed by Activation
    """
    # Strategy 1
    for layer in model.layers:
        if "eca_sigmoid" in layer.name.lower():
            return layer

    # Strategy 2
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Activation):
            try:
                shape = layer.output_shape
            except Exception:
                shape = None
            if shape and len(shape) == 3 and shape[-1] == 1:
                return layer

    # Strategy 3
    for i, l in enumerate(model.layers[:-1]):
        if isinstance(l, tf.keras.layers.Conv1D) and getattr(l, "filters", None) == 1:
            nxt = model.layers[i + 1]
            if isinstance(nxt, tf.keras.layers.Activation):
                return nxt

    raise RuntimeError("Could not find ECA activation layer")


def build_eca_model(model):
    act = find_eca_activation_layer(model)
    return tf.keras.Model(inputs=model.input, outputs=act.output)


# ==============================
#  Main Attention Extraction
# ==============================
def extract_attention(
    model,
    encrypt_func,
    plain_bits,
    key_bits,
    nr,
    delta_state,
    delta_key,
    pairs,
    n_samples,
    batch_size,
    use_gpu=True,
):
    """
    Generate a dataset and extract ECA attention weights.
    Return:
        att   : (N, C)
        y_true: (N,)
    """
    gen = NDCMultiPairGenerator(
        encryption_function=encrypt_func,
        plain_bits=plain_bits,
        key_bits=key_bits,
        nr=nr,
        delta_state=delta_state,
        delta_key=delta_key,
        n_samples=n_samples,
        batch_size=batch_size,
        pairs=pairs,
        use_gpu=use_gpu,
    )

    X_all, y_all = [], []

    for i in range(len(gen)):
        Xb, yb = gen[i]
        X_all.append(Xb)
        y_all.append(yb)

    X = np.vstack(X_all)[:n_samples]
    y = np.hstack(y_all)[:n_samples]

    att_model = build_eca_model(model)
    att = att_model.predict(X, batch_size=batch_size, verbose=0)
    att = np.squeeze(att, axis=-1)

    return att, y


# ==============================
#  Statistics
# ==============================
def summarize_attention(att, y):
    """
    Compute:
      - per-channel mean (real, random)
      - delta means
      - entropy distribution
    """
    att_real = att[y == 1]
    att_rand = att[y == 0]

    mu_real = att_real.mean(axis=0)
    mu_rand = att_rand.mean(axis=0)
    delta = mu_real - mu_rand

    H_real = entropy(att_real.T + 1e-9, base=2).mean()
    H_rand = entropy(att_rand.T + 1e-9, base=2).mean()

    return {
        "mu_real": mu_real,
        "mu_rand": mu_rand,
        "delta": delta,
        "H_real": H_real,
        "H_rand": H_rand,
    }


# ==============================
#  CLI
# ==============================
def main():
    parser = argparse.ArgumentParser(description="Extract & analyze ECA attention weights")
    parser.add_argument("--cipher-module", type=str, default="cipher.present80")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--rounds", "-r", type=int, default=7)
    parser.add_argument("--pairs", "-p", type=int, default=8)
    parser.add_argument("--input-diff", type=str, default="0x00000080")
    parser.add_argument("--delta-key-bit", type=int, default=None)
    parser.add_argument("--delta-key-hex", type=str, default=None)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--no-eca", action="store_true", help="Model is the no-ECA ablation variant; skip/disable attention extraction if set or if model contains 'noECA' in filename")
    parser.add_argument("--save-npy", action="store_true", help="Save attention & stats to attention_analysis/{cipher}/ with auto-generated filename")

    args = parser.parse_args()

    # Cipher
    cipher = _import_cipher_module(args.cipher_module)
    encrypt = cipher.encrypt
    plain_bits = cipher.plain_bits
    key_bits = cipher.key_bits

    delta_state = integer_to_binary_array(int(args.input_diff, 16), plain_bits)

    if args.delta_key_bit is not None:
        delta_key = np.zeros(key_bits, dtype=np.uint8)
        delta_key[args.delta_key_bit] = 1
    elif args.delta_key_hex:
        delta_key = _parse_delta_key_from_hex(args.delta_key_hex, key_bits)
    else:
        delta_key = np.zeros(key_bits, dtype=np.uint8)

    # Model
    mp = Path(args.model_path)
    if mp.suffix == ".keras":
        model = tf.keras.models.load_model(str(mp))
    else:
        try:
            model = tf.keras.models.load_model(str(mp))
        except Exception:
            # weights-only H5 — rebuild using requested factory
            try:
                from RKmcp import make_model_inception, make_model_inception_no_eca
            except Exception:
                raise RuntimeError("Failed to import model factories from RKmcp for rebuilding weights-only file")

            use_no_eca = bool(args.no_eca) or ('noeca' in mp.stem.lower()) or ('no_eca' in mp.stem.lower())
            factory = make_model_inception_no_eca if use_no_eca else make_model_inception
            model = factory(pairs=args.pairs, plain_bits=plain_bits)
            model.load_weights(str(mp))
            if use_no_eca:
                print("[info] Rebuilt model using no-ECA variant (make_model_inception_no_eca)")

    # Check presence of ECA layer before attempting extraction
    def _has_eca_layer(m):
        try:
            _ = find_eca_activation_layer(m)
            return True
        except Exception:
            return False

    if args.no_eca or not _has_eca_layer(model):
        print("[info] No ECA attention layer found (or --no-eca set). analyze_attention requires an ECA model to extract weights. Exiting.")
        return

    # Extract
    att, y = extract_attention(
        model,
        encrypt,
        plain_bits,
        key_bits,
        args.rounds,
        delta_state,
        delta_key,
        args.pairs,
        args.samples,
        args.batch_size,
        use_gpu=True,
    )

    stats = summarize_attention(att, y)

    print("\n=== Attention Summary ===")
    print(f"Channels: {att.shape[1]}")
    print(f"Samples : {att.shape[0]}")
    print(f"\nEntropy(real)  : {stats['H_real']:.4f}")
    print(f"Entropy(random): {stats['H_rand']:.4f}")

    print("\nTop-10 |Δ mean attention| channels:")
    idx = np.argsort(-np.abs(stats["delta"]))[:10]
    for i in idx:
        print(f"ch {i:3d}: real={stats['mu_real'][i]:.4f} rand={stats['mu_rand'][i]:.4f} Δ={stats['delta'][i]:+.4f}")

    if args.save_npy:
        # Auto-generate output directory and filename: attention_analysis/{cipher}/{modelstem}_r{rounds}_p{pairs}_{timestamp}.npz
        cipher_name = args.cipher_module.split('.')[-1]
        out_dir = Path('attention_analysis') / cipher_name
        out_dir.mkdir(parents=True, exist_ok=True)

        model_stem = Path(args.model_path).stem
        
        base = f"{model_stem}_r{args.rounds}_p{args.pairs}"
        out_name = f"{base}.npz"
        out_path = out_dir / out_name

        np.savez(
            str(out_path),
            att=att,
            y=y,
            mu_real=stats["mu_real"],
            mu_rand=stats["mu_rand"],
            delta=stats["delta"],
        )
        print(f"\nSaved attention & stats to: {out_path}")


if __name__ == "__main__":
    main()

# !python analyze_attention.py --cipher-module cipher.present80 --model-path D:\#PROJECT_RKNDIncECA\Related-key-mcp-attention-Inception-based-ND\checkpoints\present80\0x00..80_diff-bit_56\16_pairs\present80_best_7r.weights.h5 --rounds 8 --pairs 16 --input-diff 0x80 --delta-key-bit 56 --samples 1000 --save-npy
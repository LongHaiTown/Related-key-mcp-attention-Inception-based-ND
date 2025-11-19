import argparse
import os
import json
import datetime as dt
import numpy as np
import tensorflow as tf
from pathlib import Path

TRAIN_NUM_SAMPLES = 1_000_000
VAL_NUM_SAMPLES = 250_000
# CuPy compatibility check
try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    cp = np
    _CUPY_AVAILABLE = False
    print("[WARNING] CuPy not available, falling back to NumPy (CPU mode)")

from train_nets import (
    update_checkpoint_in_callbacks,
    integer_to_binary_array,
    NDCMultiPairGenerator,
    make_model_inception,
)


def parse_args():
    ap = argparse.ArgumentParser(description="Multi-stage training for neural distinguisher")
    ap.add_argument("--cipher", type=str, default="present80", help="Cipher module under cipher/ (e.g. present80, simon3264)")
    ap.add_argument("--pairs", type=int, default=8, help="Pairs per sample")
    ap.add_argument("--input-diff", type=str, default="0x00000080", help="Hex input difference (e.g. 0x80)")
    ap.add_argument("--delta-key-bit", type=int, default=None, help="Optional delta key bit index (0-based). If omitted -> all zeros")
    ap.add_argument("--stages-rounds", type=str, default="7,7,7", help="Comma list of rounds per stage")
    ap.add_argument("--stages-epochs", type=str, default="20,10,10", help="Comma list of epochs per stage")
    ap.add_argument("--stages-lrs", type=str, default="1e-3,5e-4,1e-4", help="Comma list of learning rates per stage")
    ap.add_argument("--batch-size", type=int, default=5000, help="Train batch size for generator")
    ap.add_argument("--val-batch-size", type=int, default=20000, help="Validation batch size")
    ap.add_argument("--use-gpu", action="store_true", default=True, help="Use GPU data generation (CuPy) if available")
    ap.add_argument("--out-dir", type=str, default="staged_runs", help="Output directory for final artifacts")
    ap.add_argument("--save-final", action="store_true", help="Save final full model (.keras) additionally to weights")
    ap.add_argument("--init-model", type=str, default=None, help="Path to full .keras model to start from (takes priority over --init-weights)")
    ap.add_argument("--init-weights", type=str, default=None, help="Path to weights file (.h5/.weights.h5) to initialize model")
    return ap.parse_args()


def _comma_list_to_numbers(s, cast_fn):
    return [cast_fn(x.strip()) for x in s.split(",") if x.strip()]


def build_delta_key(bit_idx, key_bits, use_gpu=False):
    """Build delta key array compatible with the chosen backend (CuPy/NumPy)"""
    lib = cp if (use_gpu and _CUPY_AVAILABLE) else np
    dk = lib.zeros(key_bits, dtype=lib.uint8)
    if bit_idx is not None:
        if bit_idx < 0 or bit_idx >= key_bits:
            raise ValueError(f"--delta-key-bit {bit_idx} out of range (0..{key_bits-1})")
        dk[bit_idx] = 1
    return dk


def build_or_load_initial_model(pairs, plain_bits, init_model_path=None, init_weights_path=None, lr=1e-3):
    """
    Build or load the initial model for staged training.
    Priority: init_model_path > init_weights_path > new model
    """
    if init_model_path:
        print(f"[INFO] Loading full model from: {init_model_path}")
        model = tf.keras.models.load_model(init_model_path)
        # Ensure model is compiled and set initial learning rate
        if model.compiled_loss is None:
            print("[INFO] Model not compiled, compiling with MSE loss...")
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr, amsgrad=True), 
                         loss="mse", metrics=["acc"])
        else:
            print(f"[INFO] Model already compiled, updating learning rate to {lr}")
            model.optimizer.learning_rate.assign(lr)
        return model
    
    # Build fresh model
    print(f"[INFO] Building new model with pairs={pairs}, plain_bits={plain_bits}")
    model = make_model_inception(pairs=pairs, plain_bits=plain_bits)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr, amsgrad=True), 
                 loss="mse", metrics=["acc"])
    
    # Load weights if provided
    if init_weights_path:
        print(f"[INFO] Loading weights from: {init_weights_path}")
        model.load_weights(init_weights_path)
    
    return model


def run_stage_training(args):
    # Import cipher dynamically
    import importlib
    cipher_mod = importlib.import_module(f"cipher.{args.cipher}")
    encrypt = cipher_mod.encrypt
    plain_bits = cipher_mod.plain_bits
    key_bits = cipher_mod.key_bits

    input_diff_int = int(args.input_diff, 16)
    delta_plain = integer_to_binary_array(input_diff_int, plain_bits)
    
    # Ensure delta_plain uses correct backend
    if args.use_gpu and _CUPY_AVAILABLE:
        if isinstance(delta_plain, np.ndarray):
            delta_plain = cp.asarray(delta_plain)
    else:
        if hasattr(delta_plain, '__cuda_array_interface__'):
            delta_plain = cp.asnumpy(delta_plain)
    
    delta_key = build_delta_key(args.delta_key_bit, key_bits, use_gpu=args.use_gpu)

    rounds_list = _comma_list_to_numbers(args.stages_rounds, int)
    epochs_list = _comma_list_to_numbers(args.stages_epochs, int)
    lrs_list = _comma_list_to_numbers(args.stages_lrs, float)

    n_stages = len(rounds_list)
    train_samples_list = [TRAIN_NUM_SAMPLES] * n_stages  # 1M samples per stage
    val_samples_list = [VAL_NUM_SAMPLES] * n_stages     # 250K samples per stage
    if not (len(epochs_list) == len(lrs_list) == len(train_samples_list) == len(val_samples_list) == n_stages):
        # Provide detailed error message to help debug
        raise ValueError(
            f"Stage parameter lists must have equal length.\n"
            f"  rounds: {len(rounds_list)} values {rounds_list}\n"
            f"  epochs: {len(epochs_list)} values {epochs_list}\n"
            f"  lrs: {len(lrs_list)} values {lrs_list}\n"
            f"  train_samples: {len(train_samples_list)} values {train_samples_list}\n"
            f"  val_samples: {len(val_samples_list)} values {val_samples_list}"
        )
    
    # Skip first stage if loading from init weights/model with matching rounds
    start_stage_idx = 0
    if (args.init_weights or args.init_model) and n_stages > 0:
        # Extract round number from init file if possible
        init_file = args.init_model or args.init_weights
        # Pattern: *_<N>r.* (e.g., speck3264_last_6r.weights.h5)
        import re
        match = re.search(r'_(\d+)r[._]', init_file)
        if match:
            init_rounds = int(match.group(1))
            if rounds_list[0] == init_rounds:
                print(f"[INFO] Skipping stage 1 (rounds={init_rounds}) - already trained in init weights/model")
                start_stage_idx = 1

    os.makedirs(args.out_dir, exist_ok=True)
    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Build or load initial model
    model = build_or_load_initial_model(
        pairs=args.pairs, 
        plain_bits=plain_bits,
        init_model_path=args.init_model,
        init_weights_path=args.init_weights,
        lr=lrs_list[0]
    )

    all_stage_histories = []

    for idx in range(start_stage_idx, n_stages):
        stage_nr = rounds_list[idx]
        stage_epochs = epochs_list[idx]
        stage_lr = lrs_list[idx]
        stage_train_samples = train_samples_list[idx]
        stage_val_samples = val_samples_list[idx]

        # Adjust learning rate for this stage
        model.optimizer.learning_rate.assign(stage_lr)

        print(f"\n[Stage {idx+1}/{n_stages}] rounds={stage_nr} lr={stage_lr} epochs={stage_epochs} train_samples={stage_train_samples}")
        if start_stage_idx > 0 and idx == start_stage_idx:
            print(f"[INFO] Continuing from pre-trained weights (skipped {start_stage_idx} stage(s))")

        gen_train = NDCMultiPairGenerator(
            encryption_function=encrypt,
            plain_bits=plain_bits, key_bits=key_bits, nr=stage_nr,
            delta_state=delta_plain, delta_key=delta_key,
            pairs=args.pairs,
            n_samples=stage_train_samples, batch_size=args.batch_size,
            use_gpu=args.use_gpu,
        )
        gen_val = NDCMultiPairGenerator(
            encryption_function=encrypt,
            plain_bits=plain_bits, key_bits=key_bits, nr=stage_nr,
            delta_state=delta_plain, delta_key=delta_key,
            pairs=args.pairs,
            n_samples=stage_val_samples, batch_size=args.val_batch_size,
            use_gpu=args.use_gpu,
        )

        # Prepare callbacks for this stage
        stage_callbacks = update_checkpoint_in_callbacks(
            [], rounds=stage_nr, cipher_name=args.cipher, run_id=f"{run_id}_stage{idx+1}", save_dir="checkpoints"
        )

        hist = model.fit(
            gen_train,
            validation_data=gen_val,
            epochs=stage_epochs,
            callbacks=stage_callbacks,
            verbose=True,
        )

        all_stage_histories.append(hist.history)

        # Save intermediate weights
        weights_path = os.path.join(args.out_dir, f"{args.cipher}_stage{idx+1}_{stage_nr}r_last.weights.h5")
        model.save_weights(weights_path)
        print(f"Saved stage {idx+1} weights: {weights_path}")

    # Final artifacts
    final_weights = os.path.join(args.out_dir, f"{args.cipher}_staged_final.weights.h5")
    model.save_weights(final_weights)
    print(f"Saved final staged weights: {final_weights}")

    if args.save_final:
        final_model_path = os.path.join(args.out_dir, f"{args.cipher}_staged_final.keras")
        model.save(final_model_path)
        print(f"Saved final staged full model: {final_model_path}")

    # Save combined history
    history_path = os.path.join(args.out_dir, f"{args.cipher}_staged_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(all_stage_histories, f, ensure_ascii=False, indent=2)
    print(f"Saved staged training history: {history_path}")


if __name__ == "__main__":
    args = parse_args()
    run_stage_training(args)

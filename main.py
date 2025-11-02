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

if __name__ == "__main__":

    # === CLI / Environment: dynamic cipher selection ===
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
    args, _ = parser.parse_known_args()

    # Dynamically import the cipher module and bind to variable 'cipher'
    try:
        cipher = importlib.import_module(f"cipher.{args.cipher}")
    except Exception as e:
        raise RuntimeError(f"Failed to import cipher module 'cipher.{args.cipher}': {e}")

    # === Config ===
    input_difference = 0x00000080
    # Use CLI-provided rounds/pairs
    n_round = int(args.rounds)
    pairs = int(args.pairs)
    if n_round <= 0:
        raise ValueError("--rounds must be a positive integer")
    if pairs <= 0:
        raise ValueError("--pairs must be a positive integer")
    plain_bits = cipher.plain_bits
    key_bits = cipher.key_bits
    encrypt = cipher.encrypt
    batch_size = 5000
    val_batch_size = 20000
    EPOCHS = 2
    NUM_SAMPLES_TRAIN = 10**6
    NUM_SAMPLES_TEST = 10**5
    # Use cipher's declared name if available
    cipher_name = getattr(cipher, "cipher_name", args.cipher)
    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")

    # === Update callbacks for checkpointing ===
    cb = update_checkpoint_in_callbacks(callbacks, rounds=n_round, cipher_name=cipher_name, run_id=run_id)

    # === Select best delta_key ===
    best_bit, best_score, all_scores = select_best_delta_key(
        encryption_function=encrypt,
        input_difference=input_difference,
        plain_bits=plain_bits,
        key_bits=key_bits,
        n_round=n_round,
        pairs=pairs,
        use_gpu=True
    )

    # === Prepare delta_plain and delta_key ===
    delta_plain = integer_to_binary_array(input_difference, plain_bits)
    delta_key = np.zeros(key_bits, dtype=np.uint8)
    delta_key[best_bit] = 1

    # === Data generators ===
    gen = NDCMultiPairGenerator(
        encryption_function=encrypt,
        plain_bits=plain_bits, key_bits=key_bits, nr=n_round,
        delta_state=delta_plain,
        delta_key=delta_key,
        pairs=pairs,
        n_samples=NUM_SAMPLES_TRAIN, batch_size=batch_size,
        use_gpu=True
    )

    gen_val = NDCMultiPairGenerator(
        encryption_function=encrypt,
        plain_bits=plain_bits, key_bits=key_bits, nr=n_round,
        delta_state=delta_plain,
        delta_key=delta_key,
        pairs=pairs,
        n_samples=NUM_SAMPLES_TEST, batch_size=val_batch_size,
        use_gpu=True
    )
    X_train, Y_train = gen[0]
    print("Sample training data shapes:", X_train.shape, Y_train.shape)
    # === Build and train model ===
    model = make_model_inception(pairs=pairs, plain_bits=plain_bits)
    optimizer = tf.keras.optimizers.Adam(amsgrad=True)
    model.compile(optimizer=optimizer, loss='mse', metrics=['acc'])
    
    history = model.fit(
        gen,
        epochs=EPOCHS,
        validation_data=gen_val,
        batch_size=batch_size,
        callbacks=cb,
        verbose=True
    )

    # === Save final model weights and architecture separately ===
    ckpt_dir = os.path.join("checkpoints", cipher_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Save weights (small file)
    weights_path = os.path.join(ckpt_dir, f"{cipher_name}_final_{n_round}r.weights.h5")
    model.save_weights(weights_path)
    
    # Save architecture (tiny JSON file)
    arch_path = os.path.join(ckpt_dir, f"{cipher_name}_final_{n_round}r_architecture.json")
    with open(arch_path, 'w') as f:
        f.write(model.to_json())
    
    print(f"Saved final model weights: {weights_path}")
    print(f"Saved model architecture: {arch_path}")

    # Persist in-memory best model and training history
    ckpt_dir = os.path.join("checkpoints", cipher_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    final_path = os.path.join(ckpt_dir, f"{cipher_name}_final_{n_round}r.keras")
    model.save(final_path)
    log_dir = os.path.join("logs", cipher_name, run_id)
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, f"history_{n_round}r.json"), "w", encoding="utf-8") as f:
        json.dump(history.history, f, ensure_ascii=False, indent=2)

    # === Evaluate model with statistics ===
    stats = evaluate_with_statistics(
        model,
        round_number=n_round,
        encryption_function=encrypt,
        plain_bits=plain_bits,
        key_bits=key_bits,
        input_difference=input_difference,
        delta_key=delta_key,
        pairs=pairs
    )
    print("Evaluation statistics:", stats)
from train_nets import (
    update_checkpoint_in_callbacks, select_best_delta_key,
    integer_to_binary_array, NDCMultiPairGenerator, make_model_inception_present80, callbacks
)
from cipher import present80
import numpy as np
import tensorflow as tf
from eval_nets import evaluate_with_statistics

if __name__ == "__main__":
    # === Config ===
    input_difference = 0x00000080
    n_round = 7
    pairs = 8
    plain_bits = present80.plain_bits
    key_bits = present80.key_bits
    encrypt = present80.encrypt
    batch_size = 5000
    val_batch_size = 20000
    EPOCHS = 20

    # === Update callbacks for checkpointing ===
    cb = update_checkpoint_in_callbacks(callbacks, rounds=n_round)

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
        n_samples=10**7, batch_size=batch_size,
        use_gpu=True
    )

    gen_val = NDCMultiPairGenerator(
        encryption_function=encrypt,
        plain_bits=plain_bits, key_bits=key_bits, nr=n_round,
        delta_state=delta_plain,
        delta_key=delta_key,
        pairs=pairs,
        n_samples=10**6, batch_size=val_batch_size,
        use_gpu=True
    )

    # === Build and train model ===
    model = make_model_inception_present80(pairs=pairs)
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
import os
import datetime
import numpy as np
import cupy as cp
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, LearningRateScheduler, LambdaCallback,
    EarlyStopping, TerminateOnNaN, TensorBoard
)
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from RKmcp import make_model_inception_present80
from make_data_train import NDCMultiPairGenerator


def integer_to_binary_array(int_val, num_bits):
    return cp.array([int(i) for i in bin(int_val)[2:].zfill(num_bits)], dtype=cp.uint8).reshape(1, num_bits)

def cyclic_lr(num_epochs, high_lr, low_lr):
    return lambda i: low_lr + ((num_epochs - 1) - i % num_epochs) / (num_epochs - 1) * (high_lr - low_lr)

def print_lr(epoch, logs):
    # Note: 'model' must be in global scope or passed in via closure if used in callbacks
    lr = tf.keras.backend.get_value(tf.keras.backend.get_value(tf.keras.backend.learning_rate(tf.keras.backend.get_session())))
    print(f"Epoch {epoch}: Learning Rate = {lr:.6f}")

lr_logger = LambdaCallback(on_epoch_end=print_lr)
lr_scheduler = LearningRateScheduler(cyclic_lr(num_epochs=10, high_lr=0.002, low_lr=0.0001))

checkpoint_cb = ModelCheckpoint(
    filepath='best_model.h5',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

earlystop_cb = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

terminate_nan_cb = TerminateOnNaN()

log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)

callbacks = [
    checkpoint_cb,
    lr_scheduler,
    lr_logger,
    earlystop_cb,
    terminate_nan_cb,
    tensorboard_cb
]

def update_checkpoint_in_callbacks(callbacks, rounds, save_dir='checkpoints'):
    from tensorflow.keras.callbacks import ModelCheckpoint

    os.makedirs(save_dir, exist_ok=True)
    filename = f"{save_dir}/best_model_{rounds}rounds.h5"
    new_ckpt = ModelCheckpoint(
        filepath=filename,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    for i, cb in enumerate(callbacks):
        if isinstance(cb, ModelCheckpoint):
            callbacks[i] = new_ckpt
            break
    else:
        callbacks.append(new_ckpt)
    return callbacks

def select_best_delta_key(
    encryption_function, input_difference,
    plain_bits, key_bits, n_round, pairs,
    n_samples=100_000, batch_size=5000, use_gpu=True
):
    delta_plain = integer_to_binary_array(input_difference, plain_bits)
    best_score = -1.0
    best_bit = -1
    all_scores = {}

    print("🔍 Searching for best delta_key (Hamming weight = 1):")

    for bit in range(key_bits):
        delta_key = np.zeros(key_bits, dtype=cp.uint8)
        delta_key[bit] = 1

        gen = NDCMultiPairGenerator(
            encryption_function=encryption_function,
            plain_bits=plain_bits, key_bits=key_bits, nr=n_round,
            delta_state=delta_plain,
            delta_key=delta_key,
            pairs=pairs,
            n_samples=n_samples, batch_size=batch_size,
            use_gpu=use_gpu, to_float32=True
        )

        X_val, Y_val = gen[0]

        try:
            pca = PCA(n_components=2).fit_transform(X_val)
            score = silhouette_score(pca, Y_val)
        except Exception:
            score = -1.0
        all_scores[bit] = score

        if score > best_score:
            best_score = score
            best_bit = bit

    print(f"\n✅ Best delta_key bit: {best_bit} with score = {best_score:.5f}")
    return best_bit, best_score, all_scores
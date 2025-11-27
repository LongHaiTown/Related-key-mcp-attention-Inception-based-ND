from tensorflow.keras.utils import Sequence
import numpy as np
import cupy as cp

def _int_to_bitarray(val, nbits, lib):
    """
    Convert val (int or numpy/cupy array) -> bit array length nbits in backend lib (np or cp).
    Always returns lib.ndarray(dtype=uint8).
    """

    if isinstance(val, (int, np.integer)):
        # create host NumPy, then convert to lib if needed
        bits_np = np.zeros((nbits,), dtype=np.uint8)
        v = int(val)
        for i in range(nbits):
            bits_np[nbits - 1 - i] = (v >> i) & 1
        return lib.asarray(bits_np) if lib is cp else bits_np

    if isinstance(val, cp.ndarray) and lib is np:
        return cp.asnumpy(val).astype(np.uint8)

    if isinstance(val, np.ndarray) and lib is cp:
        return cp.asarray(val.astype(np.uint8))

    if isinstance(val, (np.ndarray, cp.ndarray)):
        return val.astype(np.uint8)
    arr_np = np.asarray(val, dtype=np.uint8)
    return lib.asarray(arr_np) if lib is cp else arr_np


class NDCMultiPairGenerator(Sequence):
    def __init__(self, encryption_function, plain_bits, key_bits, nr,
                 delta_state=0, delta_key=0, n_samples=10**7, batch_size=10**5,
                 pairs=2, use_gpu=True, to_float32=True):

        self.encryption_function = encryption_function
        self.plain_bits = plain_bits
        self.key_bits = key_bits
        self.nr = nr
        self.delta_state = delta_state
        self.delta_key = delta_key
        self.n = int(n_samples)
        self.batch_size = int(batch_size)
        self.pairs = int(pairs)
        self.use_gpu = bool(use_gpu)
        self.to_float32 = bool(to_float32)

        self.steps = (self.n + self.batch_size - 1) // self.batch_size
        # input: for each pair we store (ΔC || C || C*) each of length plain_bits
        self.input_dim = self.pairs * 3 * self.plain_bits

    def __len__(self):
        return int(self.steps)

    def __getitem__(self, idx):
        curr_n = min(self.batch_size, self.n - idx * self.batch_size)
        lib = cp if self.use_gpu else np

        # prepare delta vectors in correct lib (bit arrays length key_bits / plain_bits)
        delta_key_vec = _int_to_bitarray(self.delta_key, self.key_bits, lib)
        delta_state_vec = _int_to_bitarray(self.delta_state, self.plain_bits, lib)

        # 1) Balanced labels per sample (shape (curr_n,))
        Y = lib.zeros(curr_n, dtype=lib.uint8)
        half = curr_n // 2
        Y[:half] = 1
        # shuffle in-place
        if self.use_gpu:
            lib.random.shuffle(Y)  # cupy supports shuffle
        else:
            np.random.shuffle(Y)

        # 2) Generate per-sample master keys (one key per sample)
        K_sample = lib.random.randint(0, 2, (curr_n, self.key_bits), dtype=lib.uint8)
        # derive related-key per sample
        K_sample_star = K_sample ^ delta_key_vec  # broadcast XOR

        # 3) Repeat keys for all pairs (flatten for encryption)
        # After repeat: shape (curr_n * pairs, key_bits)
        K = lib.repeat(K_sample, self.pairs, axis=0)
        K_star = lib.repeat(K_sample_star, self.pairs, axis=0)

        # 4) Generate plaintexts per-pair (flattened)
        # P_flat shape: (curr_n * pairs, plain_bits)
        P_flat = lib.random.randint(0, 2, (curr_n * self.pairs, self.plain_bits), dtype=lib.uint8)
        P_star_flat = lib.empty_like(P_flat)

        # 5) Build mask mapping per-sample labels to per-pair positions
        # mask_samples shape (curr_n,), mask_pairs shape (curr_n * pairs,)
        mask_samples_true = (Y == 1)
        mask_samples_false = (Y == 0)
        mask_pairs_true = lib.repeat(mask_samples_true, self.pairs, axis=0)
        mask_pairs_false = lib.repeat(mask_samples_false, self.pairs, axis=0)

        # 6) For positive-labeled samples: P* = P ^ delta_state (for all pairs in that sample)
        P_star_flat[mask_pairs_true] = P_flat[mask_pairs_true] ^ delta_state_vec

        # 7) For negative-labeled samples: P* random (noise)
        n_false = int(mask_pairs_false.sum())
        if n_false > 0:
            P_star_flat[mask_pairs_false] = lib.random.randint(0, 2, (n_false, self.plain_bits), dtype=lib.uint8)

        # 8) Encrypt (flattened)
        # encryption_function should accept arrays shaped (N, plain_bits) and (N, key_bits)
        C_flat = self.encryption_function(P_flat, K, self.nr)
        C_star_flat = self.encryption_function(P_star_flat, K_star, self.nr)

        # 9) delta_C and concatenate [ΔC || C || C*] for each pair
        delta_C_flat = C_flat ^ C_star_flat
        # concatenate along bit axis -> shape (curr_n * pairs, 3*plain_bits)
        triple_flat = lib.concatenate([delta_C_flat, C_flat, C_star_flat], axis=1)

        # 10) reshape to samples: (curr_n, pairs * 3 * plain_bits)
        X = triple_flat.reshape(curr_n, -1)

        # optional cast
        if self.to_float32:
            X = X.astype(lib.float32)

        # ensure return numpy arrays for Keras
        if self.use_gpu:
            X = cp.asnumpy(X)
            Y = cp.asnumpy(Y).astype(np.uint8)
        else:
            # ensure correct numpy dtype
            X = X.astype(np.float32) if self.to_float32 else X
            Y = Y.astype(np.uint8)

        return X, Y

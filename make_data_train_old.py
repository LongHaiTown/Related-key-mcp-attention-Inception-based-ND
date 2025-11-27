from tensorflow.keras.utils import Sequence
import numpy as np
import cupy as cp

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
        self.n = n_samples
        self.batch_size = batch_size
        self.pairs = pairs
        self.use_gpu = use_gpu
        self.to_float32 = to_float32

        self.steps = (self.n + self.batch_size - 1) // self.batch_size
        self.input_dim = self.pairs * 3 * self.plain_bits  # ΔC || C || C*

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        curr_n = min(self.batch_size, self.n - idx * self.batch_size)
        lib = cp if self.use_gpu else np

        # Cast delta to correct backend
        delta_key = self.delta_key
        delta_state = self.delta_state
        if self.use_gpu:
            if isinstance(delta_key, np.ndarray): delta_key = cp.asarray(delta_key)
            if isinstance(delta_state, np.ndarray): delta_state = cp.asarray(delta_state)
        else:
            if isinstance(delta_key, cp.ndarray): delta_key = cp.asnumpy(delta_key)
            if isinstance(delta_state, cp.ndarray): delta_state = cp.asnumpy(delta_state)

        # Generate balanced labels
        Y = lib.zeros(curr_n, dtype=lib.uint8)
        Y[:curr_n // 2] = 1
        lib.random.shuffle(Y)

        # Generate random plaintext and key
        P = lib.random.randint(0, 2, (curr_n * self.pairs, self.plain_bits), dtype=lib.uint8)
        K = lib.random.randint(0, 2, (curr_n * self.pairs, self.key_bits), dtype=lib.uint8)

        P_star = lib.empty_like(P)
        K_star = lib.empty_like(K)

        # Create mask by label
        mask_true = lib.repeat(Y == 1, self.pairs)
        mask_false = lib.repeat(Y == 0, self.pairs)

        # If Y = 1 → correct pair (P*, K*)
        P_star[mask_true] = P[mask_true] ^ delta_state
        K_star[mask_true] = K[mask_true] ^ delta_key

        # If Y = 0 → random noise
        P_star[mask_false] = lib.random.randint(0, 2, (int(mask_false.sum()), self.plain_bits), dtype=lib.uint8)
        K_star[mask_false] = lib.random.randint(0, 2, (int(mask_false.sum()), self.key_bits), dtype=lib.uint8)

        # Encrypt
        C = self.encryption_function(P, K, self.nr)
        C_star = self.encryption_function(P_star, K_star, self.nr)
        delta_C = C ^ C_star

        # Format [ΔC || C || C*]
        X = lib.concatenate([delta_C, C, C_star], axis=1)
        X = X.reshape(curr_n, -1)

        if self.to_float32:
            X = X.astype(lib.float32)

        if self.use_gpu:
            X = cp.asnumpy(X)
            Y = cp.asnumpy(Y)

        return X, Y

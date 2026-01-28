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
                 delta_state=0, delta_key=0,
                 n_samples=10**7, batch_size=10**5,
                 pairs=2, use_gpu=True, to_float32=True,
                 start_idx=0):

        self.encryption_function = encryption_function
        self.plain_bits = plain_bits
        self.key_bits = key_bits
        self.nr = nr
        self.delta_state = delta_state
        self.delta_key = delta_key

        self.n = int(n_samples)
        self.batch_size = int(batch_size)
        self.start_idx = int(start_idx)

        self.pairs = int(pairs)
        self.use_gpu = bool(use_gpu)
        self.to_float32 = bool(to_float32)

        self.steps = (self.n + self.batch_size - 1) // self.batch_size
        self.input_dim = self.pairs * 3 * self.plain_bits

    def __len__(self):
        return int(self.steps)

    def __getitem__(self, idx):
        # absolute position in the virtual dataset
        abs_idx = self.start_idx + idx

        curr_n = min(
            self.batch_size,
            self.n - idx * self.batch_size
        )
        if curr_n <= 0:
            raise IndexError

        lib = cp if self.use_gpu else np

        # ---- phần còn lại giữ NGUYÊN ----
        delta_key_vec = _int_to_bitarray(self.delta_key, self.key_bits, lib)
        delta_state_vec = _int_to_bitarray(self.delta_state, self.plain_bits, lib)

        Y = lib.zeros(curr_n, dtype=lib.uint8)
        half = curr_n // 2
        Y[:half] = 1
        lib.random.shuffle(Y)

        K_sample = lib.random.randint(0, 2, (curr_n, self.key_bits), dtype=lib.uint8)
        K_sample_star = K_sample ^ delta_key_vec

        K = lib.repeat(K_sample, self.pairs, axis=0)
        K_star = lib.repeat(K_sample_star, self.pairs, axis=0)

        P_flat = lib.random.randint(
            0, 2, (curr_n * self.pairs, self.plain_bits), dtype=lib.uint8
        )
        P_star_flat = lib.empty_like(P_flat)

        mask_pairs = lib.repeat(Y == 1, self.pairs, axis=0)
        P_star_flat[mask_pairs] = P_flat[mask_pairs] ^ delta_state_vec

        n_false = int((~mask_pairs).sum())
        if n_false > 0:
            P_star_flat[~mask_pairs] = lib.random.randint(
                0, 2, (n_false, self.plain_bits), dtype=lib.uint8
            )

        C_flat = self.encryption_function(P_flat, K, self.nr)
        C_star_flat = self.encryption_function(P_star_flat, K_star, self.nr)

        delta_C_flat = C_flat ^ C_star_flat
        triple_flat = lib.concatenate(
            [delta_C_flat, C_flat, C_star_flat], axis=1
        )

        X = triple_flat.reshape(curr_n, -1)

        if self.to_float32:
            X = X.astype(lib.float32)

        if self.use_gpu:
            X = cp.asnumpy(X)
            Y = cp.asnumpy(Y)
        else:
            X = X.astype(np.float32)
            Y = Y.astype(np.uint8)

        return X, Y


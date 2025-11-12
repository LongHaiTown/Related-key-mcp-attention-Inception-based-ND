# simeck64128_gpu.py -- Simeck64/128 implemented using CuPy for GPU
import cupy as cp

# Sequence generator for key schedule

def get_sequence(num_rounds):
    if num_rounds < 40:
        states = [1] * 5
    else:
        states = [1] * 6
    for i in range(num_rounds - len(states)):
        if num_rounds < 40:
            feedback = states[i + 2] ^ states[i]
        else:
            feedback = states[i + 1] ^ states[i]
        states.append(feedback)
    return cp.array(states, dtype=cp.uint32)

# Parameters for SIMECK64/128
plain_bits = 64
key_bits = 128
word_size = 32
rounds = 44


def WORD_SIZE():
    return word_size


MASK_VAL = 2 ** WORD_SIZE() - 1


def rol(x, k):
    return ((x << k) & MASK_VAL) | (x >> (WORD_SIZE() - k))


def ror(x, k):
    return (x >> k) | ((x << (WORD_SIZE() - k)) & MASK_VAL)


def enc_one_round(p, k):
    x, y = p
    tmp = x & rol(x, 5)
    tmp ^= rol(x, 1)
    y ^= tmp ^ k
    return y, x


def expand_key(k, t):
    sequence = get_sequence(t)

    ks = []
    # k shape: (batch, 4) words; states is reversed order of k words
    states = [k[:, i] for i in range(3, -1, -1)]
    for i in range(t):
        ks.append(states[0])
        tmp = states[1] & rol(states[1], 5)
        tmp ^= rol(states[1], 1)
        # CONSTANT = (2**WORD_SIZE() - 4) == 0xFFFFFFFC for 32-bit
        tmp ^= states[0] ^ (0xFFFFFFFC ^ sequence[i])
        states.append(tmp)
        states.pop(0)

    return ks


def convert_to_binary(arr):
    # arr shape: (batch, num_words)
    nrow, ncol = arr.shape
    X = cp.zeros((nrow, ncol * WORD_SIZE()), dtype=cp.uint8)
    for i in range(WORD_SIZE()):
        for j in range(ncol):
            X[:, j * WORD_SIZE() + i] = ((arr[:, j] >> (WORD_SIZE() - 1 - i)) & 1).astype(cp.uint8)
    return X


def convert_from_binary(arr, _dtype=cp.uint32):
    # arr shape: (batch, num_bits)
    num_words = arr.shape[1] // WORD_SIZE()
    X = cp.zeros((len(arr), num_words), dtype=_dtype)
    for i in range(num_words):
        for j in range(WORD_SIZE()):
            pos = WORD_SIZE() * i + j
            X[:, i] += (1 << (WORD_SIZE() - 1 - j)) * arr[:, pos].astype(_dtype)
    return X


def encrypt(p, k, r):
    P = convert_from_binary(p)
    K = convert_from_binary(k)
    ks = expand_key(K, r)
    x, y = P[:, 0], P[:, 1]
    for i in range(r):
        x, y = enc_one_round((x, y), ks[i])
    C = cp.stack([x, y], axis=1)
    return convert_to_binary(C)


def check_testvectors():
    # From CPU implementation vector
    p = cp.array([[0x656b696c, 0x20646e75]], dtype=cp.uint32)
    k = cp.array([[0x1b1a1918, 0x13121110, 0x0b0a0908, 0x03020100]], dtype=cp.uint32)
    pb = convert_to_binary(p)
    kb = convert_to_binary(k)
    cb = encrypt(pb, kb, rounds)
    c = convert_from_binary(cb, _dtype=cp.uint32)
    print("Ciphertext:", [hex(int(v)) for v in c[0]])
    expected = cp.array([0x45ce6902, 0x5f7ab7ed], dtype=cp.uint32)
    print("Expected:  ", [hex(int(v)) for v in expected])
    print("Pass:", bool(cp.all(c[0] == expected)))


def benchmark_gpu(batch_size=100000):
    p = cp.zeros((batch_size, 2), dtype=cp.uint32)
    k = cp.zeros((batch_size, 4), dtype=cp.uint32)
    pb = convert_to_binary(p)
    kb = convert_to_binary(k)
    cp.cuda.Stream.null.synchronize()
    import time
    st = time.time()
    _ = encrypt(pb, kb, rounds)
    cp.cuda.Stream.null.synchronize()
    et = time.time()
    print(f"{batch_size} blocks in {et-st:.4f}s, speed = {batch_size/(et-st):.2f} blocks/sec.")


if __name__ == "__main__":
    check_testvectors()

# simon64128_gpu.py -- SIMON64/128 implemented using CuPy for GPU
import cupy as cp
import time

# Parameters
plain_bits = 64
key_bits = 128
word_size = 32
rounds = 44
MASK_VAL = 2 ** word_size - 1

def WORD_SIZE(): return word_size

def ALPHA(): return 1

def BETA(): return 8

def GAMMA(): return 2

# Rotation helpers
def rol(x, k):
    return ((x << k) & MASK_VAL) | (x >> (WORD_SIZE() - k))

def ror(x, k):
    return (x >> k) | ((x << (WORD_SIZE() - k)) & MASK_VAL)

# One round of SIMON
def enc_one_round(p, k):
    x, y = p[0], p[1]
    tmp = rol(x, ALPHA()) & rol(x, BETA())
    tmp ^= rol(x, GAMMA())
    y ^= tmp ^ k
    return cp.stack([y, x], axis=0)

# Key schedule (SIMON64/128 uses z-sequence below)
Z = 0b11110000101100111001010001001000000111101001100011010111011011

def expand_key(k, t):
    ks = [0] * t
    # input K is shape (4, batch) in words order (k0, k1, k2, k3)
    # SIMON schedule uses reversed(k[0:4]) as first keys
    ks[0:4] = cp.flip(k[0:4], axis=0)
    m = 4
    rc = MASK_VAL ^ 3
    for i in range(m, t):
        c_z = ((Z >> ((i - m) % 62)) & 1) ^ rc
        tmp = ror(ks[i - 1], 3)
        tmp ^= ks[i - 3]
        tmp ^= ror(tmp, 1)
        ks[i] = ks[i - m] ^ tmp ^ c_z
    return ks

# Convert (word vector → bit vector)
# Input arr shape: (num_words, batch_size)
# Output: (batch_size, num_words * WORD_SIZE())
def convert_to_binary(arr):
    nrows, ncols = arr.shape
    X = cp.zeros((ncols, nrows * WORD_SIZE()), dtype=cp.uint8)
    for i in range(nrows * WORD_SIZE()):
        word_idx = i // WORD_SIZE()
        bit_pos = WORD_SIZE() - 1 - (i % WORD_SIZE())
        X[:, i] = ((arr[word_idx] >> bit_pos) & 1).astype(cp.uint8)
    return X

# Convert (bit vector → word vector)
# Input arr shape: (batch_size, num_words * WORD_SIZE())
# Output: (num_words, batch_size)
def convert_from_binary(arr, _dtype=cp.uint32):
    num_words = arr.shape[1] // WORD_SIZE()
    X = cp.zeros((num_words, arr.shape[0]), dtype=_dtype)
    for i in range(num_words):
        for j in range(WORD_SIZE()):
            pos = WORD_SIZE() * i + j
            X[i] += (1 << (WORD_SIZE() - 1 - j)) * arr[:, pos].astype(_dtype)
    return X

# Main encryption
# p, k are bit-matrices with shape (batch_size, bits)
# returns bit-matrix ciphertext with same shape
def encrypt(p, k, r):
    P = convert_from_binary(p)
    K = convert_from_binary(k)
    ks = expand_key(K, r)
    x, y = P[0], P[1]
    for i in range(r):
        x, y = enc_one_round((x, y), ks[i])
    C_words = cp.stack([x, y], axis=0)
    return convert_to_binary(C_words)

# ==== Test vector check ====
def check_testvectors():
    print("=== [SIMON64/128 - CuPy GPU] Strict Check Testvectors ===")
    # Match NumPy CPU format (rows = words, cols = samples)
    p = cp.array([0x656b696c, 0x20646e75], dtype=cp.uint32).reshape(-1, 1)
    k = cp.array([0x1b1a1918, 0x13121110, 0x0b0a0908, 0x03020100], dtype=cp.uint32).reshape(-1, 1)

    pb = convert_to_binary(p)
    kb = convert_to_binary(k)

    cb = encrypt(pb, kb, rounds)
    c = convert_from_binary(cb, _dtype=cp.uint32)

    got = [hex(int(v)) for v in cp.asnumpy(c[:, 0])]
    exp = [0x44c8fc20, 0xb9dfa07a]
    print("Ciphertext:", got)
    print("Expected:  ", [hex(v) for v in exp])

    expected = cp.array(exp, dtype=cp.uint32)
    ok = cp.all(c[:, 0] == expected)
    print("Pass:", bool(ok))

# ==== Benchmark (optional) ====
def benchmark_gpu(batch_size=100000):
    p = cp.zeros((2, batch_size), dtype=cp.uint32)  # words x batch
    k = cp.zeros((4, batch_size), dtype=cp.uint32)
    pb = convert_to_binary(p)
    kb = convert_to_binary(k)
    cp.cuda.Stream.null.synchronize()
    st = time.time()
    _ = encrypt(pb, kb, rounds)
    cp.cuda.Stream.null.synchronize()
    et = time.time()
    print(f"{batch_size} blocks in {et-st:.4f}s, speed = {batch_size/(et-st):,.2f} blocks/sec.")

if __name__ == "__main__":
    check_testvectors()

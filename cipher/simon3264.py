# simon3264_cupy.py -- SIMON32/64 implemented using CuPy for GPU
import cupy as cp
# Canonical cipher identifier for logging/checkpoint naming
cipher_name = "simon3264"
import time

# Parameters
plain_bits = 32
key_bits = 64
word_size = 16
rounds = 32
MASK_VAL = 2 ** word_size - 1

def WORD_SIZE(): return word_size
def ALPHA(): return 1
def BETA(): return 8
def GAMMA(): return 2

# Rotation helpers
def rol(x, k): return ((x << k) & MASK_VAL) | (x >> (WORD_SIZE() - k))
def ror(x, k): return (x >> k) | ((x << (WORD_SIZE() - k)) & MASK_VAL)

# One round of SIMON
def enc_one_round(p, k):
    tmp, c1 = p[0], p[1]
    tmp = rol(tmp, ALPHA()) & rol(tmp, BETA())
    tmp ^= rol(p[0], GAMMA())
    c1 ^= tmp ^ k
    return cp.stack([c1, p[0]], axis=0)

# Key schedule
Z = 0b01100111000011010100100010111110110011100001101010010001011111
def expand_key(k, t):
    ks = [0] * t
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
def convert_to_binary(arr):
    nrows, ncols = arr.shape  # (num_words, batch_size)
    X = cp.zeros((ncols, nrows * WORD_SIZE()), dtype=cp.uint8)
    for i in range(nrows * WORD_SIZE()):
        word_idx = i // WORD_SIZE()
        bit_pos = WORD_SIZE() - 1 - (i % WORD_SIZE())
        X[:, i] = ((arr[word_idx] >> bit_pos) & 1).astype(cp.uint8)
    return X

# Convert (bit vector → word vector)
def convert_from_binary(arr, _dtype=cp.uint16):
    num_words = arr.shape[1] // WORD_SIZE()
    X = cp.zeros((num_words, arr.shape[0]), dtype=_dtype)
    for i in range(num_words):
        for j in range(WORD_SIZE()):
            pos = WORD_SIZE() * i + j
            X[i] += (1 << (WORD_SIZE() - 1 - j)) * arr[:, pos].astype(_dtype)
    return X

# Main encryption
def encrypt(p, k, r):
    P = convert_from_binary(p)
    K = convert_from_binary(k)
    ks = expand_key(K, r)
    x, y = P[0], P[1]
    for i in range(r):
        x, y = enc_one_round((x, y), ks[i])
    return convert_to_binary(cp.stack([x, y], axis=0))

# ==== Test vector check ====
def check_testvectors():
    print("=== [SIMON32/64 - CuPy GPU] Strict Check Testvectors ===")
    # Match NumPy CPU format (rows = words, cols = samples)
    p = cp.array([0x6565, 0x6877], dtype=cp.uint16).reshape(-1, 1)
    k = cp.array([0x1918, 0x1110, 0x0908, 0x0100], dtype=cp.uint16).reshape(-1, 1)

    pb = convert_to_binary(p)
    kb = convert_to_binary(k)

    cb = encrypt(pb, kb, rounds)
    c = convert_from_binary(cb, _dtype=cp.uint16)

    print("Ciphertext:", [hex(int(v)) for v in cp.asnumpy(c[:, 0])])
    expected = cp.array([[0xc69b, 0xe9bb]], dtype=cp.uint16).T
    print("Expected:  ", [hex(int(v)) for v in cp.asnumpy(expected[:, 0])])

    if cp.all(c[:, 0] == expected[:, 0]):
        print("✅ Test passed.")
    else:
        print("❌ Test failed.")

# ==== Bit-level conversion test ====
def test_conversion():
    x = cp.array([[0x1918, 0x1110, 0x0908, 0x0100]], dtype=cp.uint16).T
    xb = convert_to_binary(x)
    x_recovered = convert_from_binary(xb, _dtype=cp.uint16)
    print("Original: ", [hex(int(v)) for v in x[:, 0]])
    print("Recovered:", [hex(int(v)) for v in x_recovered[:, 0]])
    print("Match:", cp.all(x == x_recovered))

# ==== Benchmark ====
def benchmark_gpu(batch_size=100000):
    p = cp.zeros((2, batch_size), dtype=cp.uint16)
    k = cp.zeros((4, batch_size), dtype=cp.uint16)
    pb = convert_to_binary(p)
    kb = convert_to_binary(k)
    cp.cuda.Stream.null.synchronize()
    st = time.time()
    _ = encrypt(pb, kb, rounds)
    cp.cuda.Stream.null.synchronize()
    et = time.time()
    print(f'{batch_size} blocks in {et-st:.4f}s, speed = {batch_size/(et-st):,.2f} blocks/sec.')

# ==== Entry point ====
if __name__ == "__main__":
    check_testvectors()
    # test_conversion()
    # 
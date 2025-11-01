import cupy as cp
# Canonical cipher identifier for logging/checkpoint naming
cipher_name = "present80"
plain_bits = 64
key_bits = 80
word_size = 4

def WORD_SIZE():
    return 64

Sbox = cp.array([0xc, 0x5, 0x6, 0xb, 0x9, 0x0, 0xa, 0xd,
                 0x3, 0xe, 0xf, 0x8, 0x4, 0x7, 0x1, 0x2], dtype=cp.uint8)
PBox = cp.array([0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51,
                4, 20, 36, 52, 5, 21, 37, 53, 6, 22, 38, 54, 7, 23, 39, 55,
                8, 24, 40, 56, 9, 25, 41, 57, 10, 26, 42, 58, 11, 27, 43, 59,
                12, 28, 44, 60, 13, 29, 45, 61, 14, 30, 46, 62, 15, 31, 47, 63], dtype=cp.uint8)


def SB(arr):
    num_words = arr.shape[1] // 4
    S = arr.copy()
    for i in range(num_words):
        to_sub = cp.zeros(arr.shape[0], dtype=cp.uint8)
        for j in range(4):
            pos = 4 * i + j
            to_sub += 2 ** (3 - j) * arr[:, pos]
        Sbox_sub = Sbox[to_sub]
        bits = cp.unpackbits(Sbox_sub)  
        bits = bits.reshape(-1, 8)[:, -4:]  
        S[:, 4*i:4*(i+1)] = bits
    return S

def P(arr):
    arr = arr.copy()
    arr[:, PBox] = arr[:, cp.arange(64)]
    return arr

def expand_key(k, t):
    ks = [0 for _ in range(t)]
    key = k.copy()
    for r in range(t):
        ks[r] = key[:, :64].copy()
        key = cp.roll(key, 19, axis=1)
        key[:, :4] = SB(key[:, :4])
        round_bin = cp.unpackbits(cp.array([r+1], dtype=cp.uint8))
        key[:, -23:-15] ^= round_bin[-8:]
    return ks

def encrypt(p, k, r):
    ks = expand_key(k, r)
    c = p.copy()
    for i in range(r-1):
        c ^= ks[i]
        c = SB(c)
        c = P(c)
    return c ^ ks[-1]

def convert_to_binary(arr):
    X = cp.zeros((len(arr) * WORD_SIZE(), len(arr[0])), dtype=cp.uint8)
    for i in range(len(arr) * WORD_SIZE()):
        index = i // WORD_SIZE()
        offset = WORD_SIZE() - (i % WORD_SIZE()) - 1
        X[i] = (arr[index] >> offset) & 1
    X = X.transpose()
    return X

def convert_from_binary(arr, _dtype=cp.uint64):
    num_words = arr.shape[1] // WORD_SIZE()
    X = cp.zeros((len(arr), num_words), dtype=_dtype)
    for i in range(num_words):
        for j in range(WORD_SIZE()):
            pos = WORD_SIZE() * i + j
            X[:, i] += (1 << (WORD_SIZE() - 1 - j)) * arr[:, pos].astype(cp.uint64)
    return X

def check_testvector_gpu():
    p = cp.zeros((1, 64), dtype=cp.uint8)
    k = cp.zeros((1, 80), dtype=cp.uint8)
    C = convert_from_binary(encrypt(p, k, 32))
    Chex = hex(int(C[0][0]))
    expected = '0x5579c1387b228445'
    print("Computed:", Chex)
    print("Expected:", expected)
    assert Chex == expected

check_testvector_gpu()

def benchmark_gpu(batch_size=100000):
    p = cp.zeros((batch_size, 64), dtype=cp.uint8)
    k = cp.zeros((batch_size, 80), dtype=cp.uint8)
    cp.cuda.Stream.null.synchronize()
    import time
    st = time.time()
    C = encrypt(p, k, 32)
    cp.cuda.Stream.null.synchronize()
    et = time.time()
    print(f'{batch_size} blocks in {et-st:.4f}s, tốc độ = {batch_size/(et-st):.2f} blocks/sec.')

# benchmark_gpu(10000)
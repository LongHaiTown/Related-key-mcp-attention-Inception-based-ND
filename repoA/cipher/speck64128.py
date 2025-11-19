import cupy as cp
# Canonical cipher identifier for logging/checkpoint naming
cipher_name = "speck64128"

# Cipher SPECK-64-128 (GPU version)
plain_bits = 64
key_bits = 128
word_size = 32
rounds = 27

def WORD_SIZE():
    return word_size

def ALPHA():
    return 8

def BETA():
    return 3

MASK_VAL = 2 ** WORD_SIZE() - 1

def rol(x, k):
    return ((x << k) & MASK_VAL) | (x >> (WORD_SIZE() - k))

def ror(x, k):
    return (x >> k) | ((x << (WORD_SIZE() - k)) & MASK_VAL)

def enc_one_round(p0, p1, k):
    x = ror(p0, ALPHA())
    x = (x + p1) & MASK_VAL
    x ^= k
    y = rol(p1, BETA())
    y ^= x
    return x, y

def expand_key(k, t):
    ks = [k[:, -1]]
    l = [k[:, -2], k[:, -3], k[:, -4]]
    for i in range(t - 1):
        li = l[i % 3]
        x, y = enc_one_round(li, ks[i], cp.full_like(li, i, dtype=cp.uint32))
        l[i % 3] = x
        ks.append(y)
    return ks

def convert_to_binary(arr):
    nrow, ncol = arr.shape
    X = cp.zeros((nrow, ncol * WORD_SIZE()), dtype=cp.uint8)
    for i in range(WORD_SIZE()):
        for j in range(ncol):
            X[:, j * WORD_SIZE() + i] = ((arr[:, j] >> (WORD_SIZE() - 1 - i)) & 1).astype(cp.uint8)
    return X

def convert_from_binary(arr, _dtype=cp.uint32):
    num_words = arr.shape[1] // WORD_SIZE()
    X = cp.zeros((arr.shape[0], num_words), dtype=_dtype)
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
        x, y = enc_one_round(x, y, ks[i])
    C = cp.stack([x, y], axis=1)
    return convert_to_binary(C)

def check_testvector_gpu():
    p = cp.array([[0x3b726574, 0x7475432d]], dtype=cp.uint32)
    k = cp.array([[0x1b1a1918, 0x13121110, 0x0b0a0908, 0x03020100]], dtype=cp.uint32)
    pb = convert_to_binary(p)
    kb = convert_to_binary(k)
    cb = encrypt(pb, kb, 27)
    c = convert_from_binary(cb)
    print("Final ciphertext:", [hex(int(v)) for v in c[0]])
    print("Match expected ciphertext [0x8c6fa548, 0x454e028b]?")
    print(cp.all(c[0] == cp.array([0x8c6fa548, 0x454e028b], dtype=cp.uint32)))

def benchmark_gpu(batch_size=100000):
    p = cp.zeros((batch_size, 2), dtype=cp.uint16)
    k = cp.zeros((batch_size, 4), dtype=cp.uint16)
    pb = convert_to_binary(p)
    kb = convert_to_binary(k)
    cp.cuda.Stream.null.synchronize()
    import time
    st = time.time()
    C = encrypt(pb, kb, rounds)
    cp.cuda.Stream.null.synchronize()
    et = time.time()
    print(f'{batch_size} blocks in {et-st:.4f}s, speed = {batch_size/(et-st):.2f} blocks/sec.')

    
    
check_testvector_gpu()
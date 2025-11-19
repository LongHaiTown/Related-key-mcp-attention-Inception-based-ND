# simeck3264_gpu.py -- Simeck32/64 implemented using CuPy for GPU
import cupy as cp
# Canonical cipher identifier for logging/checkpoint naming
cipher_name = "simmeck3264"

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
    return cp.array(states, dtype=cp.uint16)
    
# Thông số SIMECK32/64
plain_bits = 32
key_bits = 64
word_size = 16
rounds = 32

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
    states = [k[:, i] for i in range(3, -1, -1)]
    for i in range(t):
        ks.append(states[0])
        tmp = states[1] & rol(states[1], 5)
        tmp ^= rol(states[1], 1)
        tmp ^= states[0] ^ (0xfffc ^ sequence[i])
        states.append(tmp)
        states.pop(0)

    return ks

def convert_to_binary(arr):
    nrow, ncol = arr.shape
    X = cp.zeros((nrow, ncol * WORD_SIZE()), dtype=cp.uint8)
    for i in range(WORD_SIZE()):
        for j in range(ncol):
            X[:, j * WORD_SIZE() + i] = ((arr[:, j] >> (WORD_SIZE() - 1 - i)) & 1).astype(cp.uint8)
    return X

def convert_from_binary(arr, _dtype=cp.uint64):
    num_words = arr.shape[1] // WORD_SIZE()
    X = cp.zeros((len(arr), num_words), dtype=_dtype)
    for i in range(num_words):
        for j in range(WORD_SIZE()):
            pos = WORD_SIZE() * i + j
            X[:, i] += (1 << (WORD_SIZE() - 1 - j)) * arr[:, pos].astype(cp.uint64)
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
    p = cp.array([[0x6565, 0x6877]], dtype=cp.uint16)
    k = cp.array([[0x1918, 0x1110, 0x0908, 0x0100]], dtype=cp.uint16)  
    pb = convert_to_binary(p)
    kb = convert_to_binary(k)
    cb = encrypt(pb, kb, 32)
    c = convert_from_binary(cb)
    print("Final ciphertext (bit-level path):", [hex(int(v)) for v in c[0]])
    print("Match expected ciphertext [0x770d, 0x2c76]?")
    print(cp.all(c[0] == cp.array([0x770d, 0x2c76], dtype=cp.uint16)))

check_testvectors()

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
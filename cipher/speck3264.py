import cupy as cp
# Canonical cipher identifier for logging/checkpoint naming
cipher_name = "speck3264"

# Thông số SPECK32/64
plain_bits = 32
key_bits = 64
word_size = 16
rounds = 22

def WORD_SIZE():
    return word_size

def ALPHA():
    return 7

def BETA():
    return 2

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
    l = [k[:, -2], k[:, -3], k[:, -4]]  # reversed part
    for i in range(t - 1):
        li = l[i % 3]
        x, y = enc_one_round(li, ks[i], cp.full_like(li, i, dtype=cp.uint16))
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
        x, y = enc_one_round(x, y, ks[i])
    C = cp.stack([x, y], axis=1)
    return convert_to_binary(C)

def check_testvectors():
    p = cp.array([[0x6574, 0x694c]], dtype=cp.uint16)
    k = cp.array([[0x1918, 0x1110, 0x0908, 0x0100]], dtype=cp.uint16)  
    pb = convert_to_binary(p)
    kb = convert_to_binary(k)
    cb = encrypt(pb, kb, 22)
    c = convert_from_binary(cb)
    print("Final ciphertext (bit-level path):", [hex(int(v)) for v in c[0]])
    print("Match expected ciphertext [0xa868, 0x42f2]?")
    print(cp.all(c[0] == cp.array([0xa868, 0x42f2], dtype=cp.uint16)))

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

    
# others functions
def test_bit_conversion():
    
    print("==== Test bit conversion ====")
    p = cp.array([[0x1234, 0xabcd], [0x0001, 0xffff]], dtype=cp.uint16)
    print("Original plaintext:\n", p)

    pb = convert_to_binary(p)
    print("Bit-level representation shape:", pb.shape)
    print("Bit-level (first row):", pb[0])

    recon = convert_from_binary(pb)
    print("Reconstructed plaintext:\n", recon)

    print("Match?", cp.all(recon == p))

def check_gpu():
    p = cp.array([[0x6574, 0x694c]], dtype=cp.uint16)
    k = cp.array([[0x1918, 0x1110, 0x0908, 0x0100]], dtype=cp.uint16)

    print("Step 1: Round-trip convert_to_binary -> convert_from_binary")
    recon = convert_from_binary(convert_to_binary(p))
    print("Plaintext match:", cp.all(recon == p))

    print("\nStep 2: expand_key full round keys")
    ks = expand_key(k, 22)
    print("First 5 round keys:", [hex(int(x[0])) for x in ks[:5]])

    print("\nStep 3: Direct word-level encrypt")
    x, y = p[:, 0].copy(), p[:, 1].copy()
    for i in range(22):
        x, y = enc_one_round(x, y, ks[i])
    print("Direct ciphertext:", hex(int(x[0])), hex(int(y[0])))

    print("\nStep 4: Full encrypt using binary pipeline")
    pb = convert_to_binary(p)
    kb = convert_to_binary(k)
    cb = encrypt(pb, kb, 22)
    c = convert_from_binary(cb)
    print("Final ciphertext (bit-level path):", [hex(int(v)) for v in c[0]])
    print("Match expected ciphertext [0xa868, 0x42f2]?")
    print(cp.all(c[0] == cp.array([0xa868, 0x42f2], dtype=cp.uint16)))
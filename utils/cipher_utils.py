"""Utility helpers for cipher-based differential data generation.

Provides:
- resolve_cipher_module: dynamic import + validation
- integer_to_binary_array: convert integer to bit array (CuPy/NumPy fallback)
- build_generator_for_diff: construct NDCMultiPairGenerator for a given input difference
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

try:  # Prefer CuPy
    import cupy as cp  # type: ignore
    _USING_CUPY = True
except Exception:  # pragma: no cover
    import numpy as cp  # type: ignore
    _USING_CUPY = False

import numpy as np
from make_data_train import NDCMultiPairGenerator


def resolve_cipher_module(module_path: str):
    mod = importlib.import_module(module_path)
    required = [hasattr(mod, 'encrypt'), hasattr(mod, 'plain_bits'), hasattr(mod, 'key_bits')]
    if not all(required):
        raise ValueError(f"Cipher module '{module_path}' must define encrypt, plain_bits, key_bits")
    return mod


def integer_to_binary_array(int_val: int, num_bits: int):
    """Convert integer to 1xN bit array (unsigned), masking to `num_bits`.

    Ensures output shape is (1, num_bits) even if `int_val` exceeds the bit width.
    Works with CuPy if available, else NumPy fallback via the same alias `cp`.
    """
    if num_bits <= 0:
        raise ValueError(f"num_bits must be positive, got {num_bits}")
    mask = (1 << num_bits) - 1
    v = int(int_val) & mask
    bits = bin(v)[2:].zfill(num_bits)
    return cp.array([int(b) for b in bits], dtype=cp.uint8).reshape(1, num_bits)


def build_generator_for_diff(cipher_mod, input_diff_int: int, *, nr: int, pairs: int, n_samples: int,
                             batch_size: int, seed: int = 42, use_gpu: bool = True):
    delta_plain = integer_to_binary_array(input_diff_int, cipher_mod.plain_bits)
    delta_key = np.zeros(cipher_mod.key_bits, dtype=np.uint8)
    gen = NDCMultiPairGenerator(
        encryption_function=cipher_mod.encrypt,
        plain_bits=cipher_mod.plain_bits,
        key_bits=cipher_mod.key_bits,
        nr=nr,
        delta_state=delta_plain,
        delta_key=delta_key,
        n_samples=n_samples,
        batch_size=batch_size,
        pairs=pairs,
        use_gpu=use_gpu,
        to_float32=True,
    )
    return gen

__all__ = [
    'resolve_cipher_module',
    'integer_to_binary_array',
    'build_generator_for_diff',
    '_USING_CUPY'
]

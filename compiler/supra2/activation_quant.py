#!/usr/bin/env python3
"""Reference block-Hadamard transform and per-token symmetric INT8 quantizer."""

from __future__ import annotations

import math

import numpy as np


def block_hadamard(x: np.ndarray, h_block: int) -> np.ndarray:
    """Apply ``I_(K/h) kron H_h/sqrt(h)`` to row-major activation rows."""
    values = np.asarray(x, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("activation input must have shape [M,K]")
    if h_block <= 0 or h_block & (h_block - 1):
        raise ValueError("Hadamard block size must be a positive power of two")
    if values.shape[1] % h_block:
        raise ValueError("activation width must be divisible by Hadamard block size")

    result = values.copy()
    for offset in range(0, result.shape[1], h_block):
        block = result[:, offset:offset + h_block].copy()
        stride = 1
        while stride < h_block:
            pairs = block.reshape(
                block.shape[0], h_block // (2 * stride), 2, stride).copy()
            block.reshape(
                block.shape[0], h_block // (2 * stride), 2, stride
            )[:, :, 0, :] = pairs[:, :, 0, :] + pairs[:, :, 1, :]
            block.reshape(
                block.shape[0], h_block // (2 * stride), 2, stride
            )[:, :, 1, :] = pairs[:, :, 0, :] - pairs[:, :, 1, :]
            stride *= 2
        result[:, offset:offset + h_block] = block / np.float32(math.sqrt(h_block))
    return result


def quantize_symmetric_int8(x_rot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize each row with ``scale=max(abs(row))/127`` and ties-to-even."""
    values = np.asarray(x_rot, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("rotated activation input must have shape [M,K]")
    absmax = np.max(np.abs(values), axis=1)
    scales = absmax / np.float32(127.0)
    # A zero row has a well-defined all-zero quantized result and scale 1.
    safe_absmax = np.where(absmax == 0, np.float32(1.0), absmax)
    # Evaluate the mathematically equivalent x*127/absmax ratio in float64.
    # Dividing by the rounded float32 scale can create false half-way ties.
    normalized = (values.astype(np.float64) * 127.0) / safe_absmax.astype(np.float64)[:, None]
    quantized = np.clip(np.rint(normalized), -128, 127)
    return quantized.astype(np.int8), scales.astype(np.float32)


def quantize_activations(x: np.ndarray, h_block: int
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return rotated FP32 values, per-row INT8 values, and per-row scales."""
    rotated = block_hadamard(x, h_block)
    quantized, scales = quantize_symmetric_int8(rotated)
    return rotated, quantized, scales

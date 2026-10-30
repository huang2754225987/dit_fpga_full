#!/usr/bin/env python3
"""Verify RTL final-head tokens after unpatchifying to the latent layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compiler.supra2.prepare_final_head import (
    PATCH_VALUES,
    TOKENS,
    read_fp16_hex,
    unpatchify_tokens,
)


ROOT = Path(__file__).resolve().parents[2]


def verify_tensors(actual_tokens: np.ndarray, expected_tokens: np.ndarray,
                   expected_latent: np.ndarray, gate: float = 0.001) -> dict:
    if actual_tokens.shape != (TOKENS, PATCH_VALUES):
        raise ValueError(f"expected actual token shape {(TOKENS, PATCH_VALUES)}")
    if expected_tokens.shape != actual_tokens.shape:
        raise ValueError("actual and expected final-head token shapes differ")
    latent_shape = (4, 32, 32)
    if expected_latent.shape != latent_shape:
        raise ValueError(f"expected latent shape {latent_shape}")

    actual_latent = unpatchify_tokens(actual_tokens)
    reference_latent = unpatchify_tokens(expected_tokens)
    reference_latent_bits = np.ascontiguousarray(reference_latent).view(np.uint16)
    actual_bits = np.ascontiguousarray(actual_tokens).view(np.uint16)
    expected_bits = np.ascontiguousarray(expected_tokens).view(np.uint16)
    latent_bits = np.ascontiguousarray(actual_latent).view(np.uint16)
    expected_latent_bits = np.ascontiguousarray(expected_latent).view(np.uint16)
    if not np.array_equal(reference_latent_bits, expected_latent_bits):
        raise ValueError("reference latent does not match unpatchified reference tokens")
    delta = actual_tokens.astype(np.float32) - expected_tokens.astype(np.float32)
    peak = float(np.max(np.abs(expected_tokens)))
    rel_peak = float(np.max(np.abs(delta)) / max(peak, 1e-30))
    rel_l2 = float(np.linalg.norm(delta) /
                   max(float(np.linalg.norm(expected_tokens.astype(np.float32))), 1e-30))
    latent_delta = actual_latent.astype(np.float32) - expected_latent.astype(np.float32)
    latent_peak = float(np.max(np.abs(expected_latent)))
    latent_rel_peak = float(np.max(np.abs(latent_delta)) / max(latent_peak, 1e-30))
    latent_rel_l2 = float(np.linalg.norm(latent_delta) /
                          max(float(np.linalg.norm(expected_latent.astype(np.float32))), 1e-30))
    metrics = {
        "tokens": TOKENS,
        "head_out": PATCH_VALUES,
        "latent_shape": list(latent_shape),
        "token_bit_mismatches": int(np.count_nonzero(actual_bits != expected_bits)),
        "latent_bit_mismatches": int(np.count_nonzero(latent_bits != expected_latent_bits)),
        "max_abs": float(np.max(np.abs(delta))),
        "relative_to_peak": rel_peak,
        "relative_l2": rel_l2,
        "latent_max_abs": float(np.max(np.abs(latent_delta))),
        "latent_relative_to_peak": latent_rel_peak,
        "latent_relative_l2": latent_rel_l2,
        "gate_relative_to_peak": gate,
        "verification": "PASS" if rel_peak <= gate and latent_rel_peak <= gate else "FAIL",
    }
    if rel_peak > gate or latent_rel_peak > gate:
        raise ValueError(f"final-head token output exceeds {gate:.3%} relative-to-peak gate")
    return metrics


def verify_files(actual_token_file: Path, expected_token_file: Path,
                 expected_latent_file: Path, gate: float = 0.001) -> dict:
    actual = read_fp16_hex(actual_token_file, (TOKENS, PATCH_VALUES))
    expected = read_fp16_hex(expected_token_file, (TOKENS, PATCH_VALUES))
    latent = read_fp16_hex(expected_latent_file, (4, 32, 32))
    return verify_tensors(actual, expected, latent, gate)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual-token-file", type=Path, required=True)
    parser.add_argument("--expected-token-file", type=Path,
                        default=ROOT / "build" / "supra2_final_head" / "reference_tokens_fp16.hex")
    parser.add_argument("--expected-latent-file", type=Path,
                        default=ROOT / "build" / "supra2_final_head" / "reference_latent_fp16.hex")
    parser.add_argument("--gate", type=float, default=0.001)
    args = parser.parse_args()
    print(json.dumps(verify_files(args.actual_token_file, args.expected_token_file,
                                 args.expected_latent_file, args.gate), indent=2))


if __name__ == "__main__":
    main()

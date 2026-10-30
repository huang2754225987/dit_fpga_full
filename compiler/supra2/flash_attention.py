"""Supra2-IMG compiler entry point for tiled FlashAttention microcode."""

from dataclasses import dataclass
from pathlib import Path
import sys

_BACKEND = Path(__file__).resolve().parents[1] / "compiler_backend_py"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import attention
import isa


TOKENS = 256
HEADS = 9
HEAD_DIM = 64
TILE = 32


@dataclass(frozen=True)
class AttentionHead:
    """L1 layout and K/V tile-load instructions for one attention head."""

    q_base: int
    k_base: int
    v_base: int
    score_base: int
    alpha_base: int
    output_base: int
    state_slot_base: int
    key_tile_loaders: tuple


def compile_head(head: AttentionHead, *, query_start=0, query_count=TILE,
                 score_scale_base=None, output_page=1,
                 defer_online_normalization=False):
    """Compile one 32-row query tile for one Supra2 head.

    The caller supplies eight loader instruction lists, one per 32-key K/V
    tile. The emitted stream performs tiled QK GEMMs, one online-softmax
    update per active query and key tile, then PV GEMMs with online rescaling.
    Q/K/V and output data remain in the caller's L1 layout.
    """
    expected_tiles = (TOKENS + TILE - 1) // TILE
    if len(head.key_tile_loaders) != expected_tiles:
        raise ValueError(f"Supra2 self-attention requires {expected_tiles} K/V tile loaders")
    program = attention.emit_flash_attention_query_block(
        q_base=head.q_base + query_start * HEAD_DIM * 2,
        k_base=head.k_base,
        v_base=head.v_base,
        score_base=head.score_base,
        alpha_base=head.alpha_base,
        output_base=head.output_base + query_start * HEAD_DIM * 2,
        query_start=0,
        query_count=query_count,
        sequence_length=TOKENS,
        key_tile_loaders=head.key_tile_loaders,
        head_dim=HEAD_DIM,
        query_row_stride_bytes=HEAD_DIM * 2,
        output_row_stride=HEAD_DIM * 2,
        score_scale_base=score_scale_base,
        output_page=output_page,
        state_slot_base=head.state_slot_base,
        k_token_row_stride_bytes=HEAD_DIM * 2,
        v_token_row_stride_bytes=HEAD_DIM * 2,
        defer_online_normalization=defer_online_normalization,
        causal=False,
    )
    program.append(isa.end())
    return program


def compile_self_attention(heads, *, query_start=0, query_count=TILE,
                           all_queries=True,
                           score_scale_base=None, output_page=1,
                           defer_online_normalization=False):
    """Compile the nine head programs for one Supra2 query tile."""
    if len(heads) != HEADS:
        raise ValueError(f"Supra2 self-attention requires {HEADS} heads")
    programs = []
    block_starts = (range(0, TOKENS, query_count) if all_queries
                    else (query_start,))
    for head in heads:
        for block_start in block_starts:
            programs.extend(compile_head(
                head,
                query_start=block_start,
                query_count=min(query_count, TOKENS - block_start),
                score_scale_base=score_scale_base,
                output_page=output_page,
                defer_online_normalization=defer_online_normalization,
            ))
    return programs


def write_program(program, output_dir):
    """Write little-endian microcode and a readable instruction listing."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    isa.write_binary(output_dir / "flash_attention.bin", program)
    isa.write_text(output_dir / "flash_attention.txt", program)
    return len(program)

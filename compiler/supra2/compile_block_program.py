"""Emit the stage-control program for the connected Supra2 block RTL.

This small ISA controls stage boundaries. Operand movement inside a stage and
the block-13 DMA weight protocol remain in RTL; they are not microcoded here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


STAGES = (
    "LN0", "QKV", "ATTN_SELF", "SPROJ", "RES0", "LN1", "CQ", "CKV",
    "ATTN_CROSS", "CPROJ", "RES1", "LN2", "MLP0", "GELU", "MLP2", "RES2",
)
OP_RUN_STAGE = 0x1
OP_END = 0xF
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "build/supra2_block_program"


def encode_run_stage(stage_id: int) -> int:
    if not 0 <= stage_id < len(STAGES):
        raise ValueError(f"stage ID out of range: {stage_id}")
    return (OP_RUN_STAGE << 28) | (stage_id << 23)


def compile_block_program() -> tuple[int, ...]:
    return tuple(encode_run_stage(i) for i in range(len(STAGES))) + (OP_END << 28,)


def emit(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    words = compile_block_program()
    memory_file = output_dir / "block13_stage_program.hex"
    memory_file.write_text("".join(f"{word:08x}\n" for word in words), encoding="ascii")
    listing = [f"{i:02d}  {word:08x}  RUN_STAGE {STAGES[i]}"
               for i, word in enumerate(words[:-1])]
    listing.append(f"16  {words[-1]:08x}  END")
    (output_dir / "block13_stage_program.txt").write_text("\n".join(listing) + "\n", encoding="ascii")
    manifest = {
        "isa": "supra2-stage-v1",
        "word_bits": 32,
        "encoding": "opcode[31:28], stage_id[27:23], reserved[22:0]=0",
        "instructions": len(words),
        "block": 13,
        "stages": list(STAGES),
        "scope": "stage boundary control; block-13 DMA and operator loops remain RTL",
    }
    (output_dir / "block13_stage_program.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(emit(args.output_dir), indent=2))


if __name__ == "__main__":
    main()

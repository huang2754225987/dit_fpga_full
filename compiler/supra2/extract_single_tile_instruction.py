"""Extract one generated tile command for an isolated operator RTL run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def extract(program_json: Path, output_hex: Path, task_name: str,
            output_n_tiles: int) -> dict:
    manifest = json.loads(program_json.read_text(encoding="utf-8"))
    matches = [item for item in manifest["commands"] if item.get("task") == task_name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one task {task_name!r}, found {len(matches)}")
    command = matches[0]
    word = int(command["word"], 16)
    if command.get("op") != "GEMM_TILE":
        raise ValueError("the isolated regression currently supports GEMM_TILE only")
    if command.get("stage") != "QKV" or command.get("row_tile") != 0:
        raise ValueError("the fixture contract requires the block-13 QKV row tile 0")
    source_n_tiles = int(command["n_tiles_or_head"])
    if output_n_tiles < 1 or output_n_tiles > source_n_tiles:
        raise ValueError("requested output N-tile count must fit within the source command")

    # The isolated test supplies this operator's operands directly; remove the
    # whole-block producer dependencies while preserving compiler tile fields.
    word &= ~(((1 << 16) - 1) << 24)
    word &= ~0xFF
    word = (word & ~(0xFF << 96)) | (output_n_tiles << 96)
    word = (word & ~(((1 << 16) - 1) << 8)) | (1 << 8)
    end = 0x0F << 120
    output_hex.parent.mkdir(parents=True, exist_ok=True)
    output_hex.write_text(f"{word:032x}\n{end:032x}\n", encoding="ascii")
    result = {
        "source_program": str(program_json),
        "source_pc": command["pc"],
        "task": task_name,
        "instruction": f"{word:032x}",
        "opcode": command["op"],
        "stage": command["stage"],
        "row_tile": command["row_tile"],
        "source_n_tiles": source_n_tiles,
        "n_tiles": output_n_tiles,
        "k_tiles": command["k_tiles_or_keys"],
        "fixture_contract": "block13 QKV operator, first 32-row tile, first output N tiles, full K reduction",
        "dependencies": "intentionally satisfied by direct fixture preload for isolated operator testing",
    }
    output_hex.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program-json", type=Path, required=True)
    parser.add_argument("--output-hex", type=Path, required=True)
    parser.add_argument("--task", default="QKV.r0")
    parser.add_argument("--output-n-tiles", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(extract(args.program_json, args.output_hex, args.task,
                             args.output_n_tiles), indent=2))


if __name__ == "__main__":
    main()

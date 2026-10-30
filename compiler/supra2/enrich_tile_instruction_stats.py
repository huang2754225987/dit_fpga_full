"""Join measured RTL retire data with the compiler's tile-program metadata."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    program = json.loads(args.program.read_text(encoding="utf-8"))
    commands = {int(item["pc"]): item for item in program["commands"]}
    rows = []
    with args.stats.open(newline="", encoding="utf-8") as source:
        for measured in csv.DictReader(source):
            pc = int(measured["pc"])
            command = commands[pc]
            actual = int(measured["execution_cycles"])
            estimate = int(command["duration"])
            rows.append({
                "pc": pc,
                "task": command["task"],
                "opcode": command["op"],
                "stage": command["stage"],
                "row_tile": command["row_tile"],
                "n_tiles_or_head": command["n_tiles_or_head"],
                "k_tiles_or_keys": command["k_tiles_or_keys"],
                "scheduled_start_cycle": command["scheduled_start"],
                "estimated_cycles": estimate,
                "actual_cycles": actual,
                "delta_cycles": actual - estimate,
                "actual_retire_cycle": int(measured["retire_cycle"]),
                "dependency_count": command["dependency_count"],
                "wait_for": ";".join(command["wait_for"]),
            })

    fields = list(rows[0]) if rows else [
        "pc", "task", "opcode", "stage", "row_tile", "n_tiles_or_head",
        "k_tiles_or_keys", "scheduled_start_cycle", "estimated_cycles",
        "actual_cycles", "delta_cycles", "actual_retire_cycle",
        "dependency_count", "wait_for",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({"measured_instructions": len(rows),
                      "program_instructions": len(commands),
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()

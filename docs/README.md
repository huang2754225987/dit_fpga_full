# Project directory map

- `README.md`: project overview and current top-level status.
- `HANDOFF.md`: detailed implementation handoff and active engineering notes.
- `AGENTS.md`: persistent scope and workflow instructions.
- `config/`: model and FPGA target metadata.
- `compiler/`: package export, image generation, and software-side preparation.
- `include/`: shared C/C++ headers.
- `rtl/`: synthesizable RTL and simulation testbenches; model-specific RTL is under `rtl/supra2/`.
- `scripts/`: simulation, synthesis, implementation, and utility entry points.
- `tests/`: Python contract and numerical tests.
- `docs/figures/`: generated architecture and algorithm diagrams.
- `docs/innovation_points.md`: evidence-based thesis contribution framing and claim boundaries.
- `papers/`: research material and related-work notes.
- `packages/`: handoff ZIP archives.
- `model/`: local checkpoints; intentionally excluded from source control.
- `build/`: generated fixtures, tool logs, checkpoints, and implementation outputs.

Legacy root-level simulator/Vivado logs, journals, crash dumps, and tool caches
were archived under `build/root_artifacts_20261026/` so the source tree stays
readable; the files were preserved.

The root-level `run_reference.py` remains in place because `tests/test_reference.py`
imports it as a top-level module.

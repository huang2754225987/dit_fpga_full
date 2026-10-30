#!/usr/bin/env python3
"""Export the conditional text context used by the archived block-0 golden."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT=Path(__file__).resolve().parents[2]
DEFAULT_EXPORT=ROOT/"build"/"supra2_hw_export"/"export"


def _write_hex(path:Path,values:np.ndarray)->None:
    path.write_text("".join(f"{int(v)&0xffff:04x}\n" for v in values.flat),
                    encoding="ascii")


def prepare(export_dir:Path,output_dir:Path)->dict:
    with np.load(export_dir/"golden_steps.npz",allow_pickle=False) as steps, \
            np.load(export_dir/"golden_block.npz",allow_pickle=False) as blocks:
        context=np.ascontiguousarray(steps["ctx_all"][0],dtype=np.float16)
        expected=np.ascontiguousarray(blocks["b0.ctx"],dtype=np.float16)
    if context.shape!=(128,768) or expected.shape!=(128,576):
        raise ValueError("Archive conditional context has unexpected dimensions")
    output_dir.mkdir(parents=True,exist_ok=True)
    _write_hex(output_dir/"raw_input_fp16.hex",context.view(np.uint16))
    _write_hex(output_dir/"expected_block_context_fp16.hex",expected.view(np.uint16))
    manifest={"rows":128,"input_features":768,"output_features":576,
              "conditioning":"conditional row 0 from golden_steps.ctx_all",
              "reference":"golden_block.b0.ctx",
              "projection":"quantized ctx_proj, RTL test is INT8 frontend + systolic + epilogue",
              "model_forward":False}
    (output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",
                                             encoding="utf-8")
    return manifest


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir",type=Path,default=DEFAULT_EXPORT)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"build"/"supra2_ctx_proj")
    args=parser.parse_args()
    print(json.dumps(prepare(args.export_dir,args.output_dir),indent=2))


if __name__=="__main__": main()

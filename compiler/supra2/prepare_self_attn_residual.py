#!/usr/bin/env python3
"""Prepare a block self-attention residual from RTL projection output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


ROOT=Path(__file__).resolve().parents[2]
DEFAULT_EXPORT=ROOT/"build"/"supra2_hw_export"/"export"


def _read(path: Path, count: int) -> np.ndarray:
    bits=np.fromiter((int(v,16) for v in path.read_text(encoding="ascii").splitlines()),
                     dtype=np.uint16)
    if bits.size!=count: raise ValueError(f"{path} has {bits.size} values; expected {count}")
    return bits.view(np.float16)


def _write(path: Path, values: np.ndarray) -> None:
    path.write_text("".join(f"{int(v)&0xffff:04x}\n" for v in values.flat),encoding="ascii")


def prepare(export_dir: Path, output_dir: Path, update_file: Path, gate_file: Path,
            residual_file: Path | None = None, block: int = 0) -> dict:
    if residual_file is None:
        with np.load(export_dir/"golden_block.npz",allow_pickle=False) as blocks:
            x=np.asarray(blocks[f"b{block}.x"],dtype=np.float16)
        residual_source=f"archive block input b{block}.x"
    else:
        x=_read(residual_file,256*576).reshape(256,576)
        residual_source=str(residual_file)
    rows,features=x.shape
    update=_read(update_file,rows*features).reshape(rows,features)
    gate=_read(gate_file,features)
    expected=(torch.from_numpy(x.copy())+
              torch.from_numpy(gate.copy()).unsqueeze(0)*torch.from_numpy(update.copy())).numpy()
    output_dir.mkdir(parents=True,exist_ok=True)
    _write(output_dir/"residual_fp16.hex",x.view(np.uint16))
    _write(output_dir/"update_fp16.hex",update.view(np.uint16))
    _write(output_dir/"gate_fp16.hex",gate.view(np.uint16))
    _write(output_dir/"expected_fp16.hex",expected.view(np.uint16))
    manifest={"rows":rows,"features":features,"residual_source":residual_source,
              "update_source":"RTL projection output",
              "block":block,"gate_source":f"CPU-precomputed block-{block} AdaLN gate vector",
              "operation":"FP16 x + FP16(gate * update)","model_forward":False}
    (output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    return manifest


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir",type=Path,default=DEFAULT_EXPORT)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"build"/"supra2_self_attn_residual")
    parser.add_argument("--update-file",type=Path,required=True)
    parser.add_argument("--gate-file",type=Path,required=True)
    parser.add_argument("--residual-file",type=Path,
                        help="optional residual input; defaults to archived block input")
    parser.add_argument("--block",type=int,default=0,
                        help="archive block index used when --residual-file is omitted")
    args=parser.parse_args()
    print(json.dumps(prepare(args.export_dir,args.output_dir,args.update_file,args.gate_file,
                             args.residual_file,args.block),indent=2))


if __name__=="__main__": main()

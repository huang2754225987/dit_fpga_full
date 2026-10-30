#!/usr/bin/env python3
"""Prepare one full-key online-PV row from RTL QKV/QK outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT=Path(__file__).resolve().parents[2]
TOKENS,HEADS,HEAD_DIM=256,9,64


def _read(path:Path,count:int)->np.ndarray:
    bits=np.fromiter((int(v,16) for v in path.read_text(encoding="ascii").splitlines()),dtype=np.uint16)
    if bits.size!=count: raise ValueError(f"{path} contains {bits.size}; expected {count}")
    return bits.view(np.float16)


def _write(path:Path,values:np.ndarray)->None:
    path.write_text("".join(f"{int(v)&0xffff:04x}\n" for v in values.flat),encoding="ascii")


def prepare(output_dir:Path,qkv_file:Path,score_file:Path,query:int|None=None)->dict:
    if query is not None and not 0<=query<TOKENS:
        raise ValueError("query index must be in [0,255]")
    qkv=_read(qkv_file,TOKENS*3*HEADS*HEAD_DIM).reshape(TOKENS,3,HEADS,HEAD_DIM)
    scores=_read(score_file,TOKENS*HEADS*TOKENS).reshape(TOKENS,HEADS,TOKENS)
    values=qkv[:,2].astype(np.float32)
    expected=np.empty((HEADS,TOKENS,HEAD_DIM),dtype=np.float16)
    for head in range(HEADS):
        rows=scores[:,head].astype(np.float32)
        ex=np.exp(rows-np.max(rows,axis=1,keepdims=True),dtype=np.float32)
        probs=ex/np.sum(ex,axis=1,keepdims=True,dtype=np.float32)
        expected[head]=(probs@values[:,head].astype(np.float32)).astype(np.float16)
    if query is not None:
        selected_scores=scores[query]
        selected_expected=expected[:,query]
    else:
        selected_scores=scores.transpose(1,0,2)
        selected_expected=expected
    v_transposed=np.ascontiguousarray(values.transpose(1,2,0),dtype=np.float16)
    output_dir.mkdir(parents=True,exist_ok=True)
    _write(output_dir/"scores_fp16.hex",selected_scores.view(np.uint16))
    _write(output_dir/"values_head_dim_key_fp16.hex",v_transposed.view(np.uint16))
    _write(output_dir/"expected_attention_fp16.hex",selected_expected.view(np.uint16))
    manifest={"query":query,"queries":TOKENS if query is None else 1,
              "query_indices":list(range(TOKENS)) if query is None else [query],
              "heads":HEADS,"keys":TOKENS,"head_dim":HEAD_DIM,
              "qkv_source":"RTL LayerNorm/AdaLN + QKV projection",
              "score_source":"RTL FP16 QK systolic array",
              "reference":"FP32 softmax(QK RTL scores) @ V, rounded to FP16",
              "online_pv":"no materialized probability matrix; tile max/sum rescaling",
              "model_forward":False}
    (output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    return manifest


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qkv-file",type=Path,required=True)
    parser.add_argument("--score-file",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"build"/"supra2_attention_online_pv")
    parser.add_argument("--query",type=int,default=None,
                        help="prepare one query only; default prepares all 256 queries")
    args=parser.parse_args()
    print(json.dumps(prepare(args.output_dir,args.qkv_file,args.score_file,args.query),indent=2))


if __name__=="__main__": main()

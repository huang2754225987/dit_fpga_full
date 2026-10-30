#!/usr/bin/env python3
"""Compare RTL ctx_proj output with the archived block-0 context tensor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def verify(reference_file:Path,actual_file:Path,max_rel_peak:float=0.006)->dict:
    def read(path:Path)->np.ndarray:
        bits=np.fromiter((int(v,16) for v in path.read_text(encoding="ascii").splitlines()),
                         dtype=np.uint16)
        if bits.size!=128*576:
            raise ValueError(f"{path} has {bits.size} values; expected {128*576}")
        return bits.view(np.float16).astype(np.float32)
    expected,actual=read(reference_file),read(actual_file)
    diff=np.abs(actual-expected)
    peak=float(np.max(np.abs(expected)))
    metrics={"elements":int(expected.size),"max_abs":float(diff.max()),
             "rel_to_peak":float(diff.max()/peak),
             "rmse":float(np.sqrt(np.mean(diff*diff))),
             "gate_rel_to_peak":max_rel_peak}
    if metrics["rel_to_peak"]>max_rel_peak:
        raise ValueError(f"ctx_proj relative-to-peak error {metrics['rel_to_peak']:.8g} "
                         f"exceeds {max_rel_peak:.8g}")
    return metrics


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-file",type=Path,required=True)
    parser.add_argument("--actual-file",type=Path,required=True)
    parser.add_argument("--max-rel-peak",type=float,default=0.006)
    args=parser.parse_args()
    print(json.dumps(verify(args.reference_file,args.actual_file,args.max_rel_peak),indent=2))


if __name__=="__main__": main()

import tempfile
import unittest
from pathlib import Path

import numpy as np

from compiler.supra2.prepare_context_projection import DEFAULT_EXPORT, prepare
from compiler.supra2.verify_context_projection import verify


class Supra2ContextProjectionTests(unittest.TestCase):
    def test_context_fixture_and_package_shape_contract(self):
        if not (DEFAULT_EXPORT/"golden_steps.npz").is_file():
            self.skipTest("Supra2 step export has not been extracted")
        with tempfile.TemporaryDirectory(
                prefix="supra2_ctx_proj_",dir=DEFAULT_EXPORT.parents[2]/"build") as temp:
            output=Path(temp)
            manifest=prepare(DEFAULT_EXPORT,output)
            self.assertEqual((manifest["rows"],manifest["input_features"],
                              manifest["output_features"]),(128,768,576))
            raw=np.fromiter(
                (int(v,16) for v in (output/"raw_input_fp16.hex").read_text().splitlines()),
                dtype=np.uint16).view(np.float16).reshape(128,768)
            expected=np.fromiter(
                (int(v,16) for v in
                 (output/"expected_block_context_fp16.hex").read_text().splitlines()),
                dtype=np.uint16).view(np.float16).reshape(128,576)
            with np.load(DEFAULT_EXPORT/"golden_steps.npz",allow_pickle=False) as steps:
                np.testing.assert_array_equal(raw,steps["ctx_all"][0])
            with np.load(DEFAULT_EXPORT/"golden_block.npz",allow_pickle=False) as blocks:
                np.testing.assert_array_equal(expected,blocks["b0.ctx"])
            metrics=verify(output/"expected_block_context_fp16.hex",
                           output/"expected_block_context_fp16.hex")
            self.assertEqual(metrics["rel_to_peak"],0.0)


if __name__=="__main__":
    unittest.main()

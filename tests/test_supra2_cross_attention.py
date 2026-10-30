import tempfile
import unittest
from pathlib import Path

import numpy as np

from compiler.supra2.prepare_cross_attention_qk import prepare as prepare_qk
from compiler.supra2.prepare_cross_attention_online_pv import prepare as prepare_pv
from compiler.supra2.prepare_cross_attention_context import prepare as prepare_context


def write_fp16(path: Path, values: np.ndarray) -> None:
    bits = values.astype(np.float16).view(np.uint16)
    path.write_text("".join(f"{int(v):04x}\n" for v in bits.flat), encoding="ascii")


class Supra2CrossAttentionFixtureTests(unittest.TestCase):
    def test_qk_and_online_pv_fixture_contracts(self):
        rng = np.random.default_rng(20261018)
        q = rng.normal(0, 0.1, (256, 9, 64)).astype(np.float16)
        kv = rng.normal(0, 0.1, (32, 2, 9, 64)).astype(np.float16)
        build_dir = Path(__file__).resolve().parents[1] / "build"
        with tempfile.TemporaryDirectory(prefix="supra2_cross_attn_", dir=build_dir) as temp:
            root = Path(temp)
            q_file, kv_file = root / "q.hex", root / "kv.hex"
            write_fp16(q_file, q)
            write_fp16(kv_file, kv)
            qk_dir, pv_dir = root / "qk", root / "pv"
            manifest = prepare_qk(q_file, kv_file, qk_dir)
            self.assertEqual((manifest["queries"], manifest["keys"],
                              manifest["valid_keys"], manifest["heads"]),
                             (256, 32, 15, 9))
            score_file = root / "scores.hex"
            write_fp16(score_file, np.zeros((256, 9, 32), dtype=np.float16))
            pv_manifest = prepare_pv(kv_file, score_file, pv_dir)
            self.assertEqual((pv_manifest["queries"], pv_manifest["keys"],
                              pv_manifest["valid_keys"]), (256, 32, 15))
            actual = np.fromiter(
                (int(v, 16) for v in (pv_dir / "expected_attention_fp16.hex").read_text().splitlines()),
                dtype=np.uint16).view(np.float16).reshape(9, 256, 64)
            expected = np.broadcast_to(kv[:15, 1].astype(np.float32).mean(axis=0)[:, None, :], actual.shape)
            np.testing.assert_allclose(actual, expected, rtol=0.01, atol=0.001)
            scores = np.fromiter(
                (int(v, 16) for v in (qk_dir / "expected_scores_fp16.hex").read_text().splitlines()),
                dtype=np.uint16).view(np.float16)
            self.assertEqual(scores.size, 256 * 9 * 32)

    def test_masked_context_is_cropped_to_one_tile(self):
        rng = np.random.default_rng(7)
        source = rng.normal(size=(128, 576)).astype(np.float16)
        with tempfile.TemporaryDirectory(
                prefix="supra2_cross_context_",
                dir=Path(__file__).resolve().parents[1] / "build") as temp:
            root = Path(temp)
            source_file = root / "context.hex"
            output = root / "packed"
            write_fp16(source_file, source)
            manifest = prepare_context(source_file,
                                       Path(__file__).resolve().parents[1] / "build" / "supra2_hw_export" / "export",
                                       output)
            self.assertEqual((manifest["source_rows"], manifest["physical_rows"],
                              manifest["valid_keys"]), (128, 32, 15))
            packed = np.fromiter(
                (int(v, 16) for v in (output / "valid_context_fp16.hex").read_text().splitlines()),
                dtype=np.uint16).view(np.float16).reshape(32, 576)
            np.testing.assert_array_equal(packed[:15], source[:15])
            self.assertEqual(np.count_nonzero(packed[15:]), 0)


if __name__ == "__main__":
    unittest.main()

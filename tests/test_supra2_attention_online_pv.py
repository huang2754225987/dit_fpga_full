import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from compiler.supra2 import prepare_attention_online_pv as fixture


class Supra2AttentionOnlinePVFixtureTests(unittest.TestCase):
    def setUp(self):
        build_dir = Path(__file__).resolve().parents[1] / "build"
        self.temp = tempfile.TemporaryDirectory(prefix="supra2_online_pv_",
                                                 dir=build_dir)
        self.root = Path(self.temp.name)
        self.qkv_file = self.root / "qkv.hex"
        self.score_file = self.root / "scores.hex"
        self.tok, self.heads, self.dim = 4, 2, 4
        rng = np.random.default_rng(17)
        qkv = rng.normal(size=(self.tok, 3, self.heads, self.dim)).astype(np.float16)
        scores = rng.normal(size=(self.tok, self.heads, self.tok)).astype(np.float16)
        fixture._write(self.qkv_file, qkv.view(np.uint16))
        fixture._write(self.score_file, scores.view(np.uint16))
        self.qkv = qkv
        self.scores = scores
        self.shape_patch = patch.multiple(
            fixture, TOKENS=self.tok, HEADS=self.heads, HEAD_DIM=self.dim)
        self.shape_patch.start()

    def tearDown(self):
        self.shape_patch.stop()
        self.temp.cleanup()

    def _expected(self):
        result = np.empty((self.heads, self.tok, self.dim), dtype=np.float16)
        values = self.qkv[:, 2].astype(np.float32)
        for head in range(self.heads):
            rows = self.scores[:, head].astype(np.float32)
            ex = np.exp(rows - rows.max(axis=1, keepdims=True))
            probs = ex / ex.sum(axis=1, keepdims=True)
            result[head] = (probs @ values[:, head]).astype(np.float16)
        return result

    def test_default_fixture_covers_all_queries_in_head_major_order(self):
        output = self.root / "all"
        manifest = fixture.prepare(output, self.qkv_file, self.score_file)
        self.assertEqual((manifest["queries"], manifest["heads"]),
                         (self.tok, self.heads))
        scores = np.fromiter(
            (int(v, 16) for v in (output / "scores_fp16.hex").read_text().splitlines()),
            dtype=np.uint16,
        ).view(np.float16).reshape(self.heads, self.tok, self.tok)
        expected = np.fromiter(
            (int(v, 16) for v in (output / "expected_attention_fp16.hex").read_text().splitlines()),
            dtype=np.uint16,
        ).view(np.float16).reshape(self.heads, self.tok, self.dim)
        np.testing.assert_array_equal(scores, self.scores.transpose(1, 0, 2))
        np.testing.assert_array_equal(expected, self._expected())
        self.assertEqual(json.loads((output / "manifest.json").read_text())["queries"],
                         self.tok)

    def test_single_query_fixture_preserves_selected_query_order(self):
        query = 2
        output = self.root / "single"
        manifest = fixture.prepare(output, self.qkv_file, self.score_file, query=query)
        self.assertEqual((manifest["queries"], manifest["query_indices"]), (1, [query]))
        scores = np.fromiter(
            (int(v, 16) for v in (output / "scores_fp16.hex").read_text().splitlines()),
            dtype=np.uint16,
        ).view(np.float16).reshape(self.heads, self.tok)
        expected = np.fromiter(
            (int(v, 16) for v in (output / "expected_attention_fp16.hex").read_text().splitlines()),
            dtype=np.uint16,
        ).view(np.float16).reshape(self.heads, self.dim)
        np.testing.assert_array_equal(scores, self.scores[query])
        np.testing.assert_array_equal(expected, self._expected()[:, query])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import numpy as np

from compiler.supra2.prepare_final_head import unpatchify_tokens
from compiler.supra2.verify_final_head import verify_tensors


class Supra2FinalHeadTests(unittest.TestCase):
    def test_unpatchify_preserves_channel_and_patch_order(self):
        tokens = np.arange(256 * 16, dtype=np.float32).reshape(256, 16)
        latent = unpatchify_tokens(tokens)
        self.assertEqual(latent.shape, (4, 32, 32))
        for token in (0, 1, 15, 16, 255):
            row, col = divmod(token, 16)
            patch = tokens[token].reshape(4, 2, 2)
            np.testing.assert_array_equal(
                latent[:, row * 2:row * 2 + 2, col * 2:col * 2 + 2], patch
            )

    def test_unpatchify_rejects_wrong_geometry(self):
        with self.assertRaises(ValueError):
            unpatchify_tokens(np.zeros((256, 15), dtype=np.float16))

    def test_final_head_verifier_checks_tokens_and_unpatchified_latent(self):
        expected = np.linspace(-1.0, 1.0, 256 * 16, dtype=np.float32)
        expected = expected.astype(np.float16).reshape(256, 16)
        actual = expected.copy()
        actual[0, 0] = np.nextafter(actual[0, 0], np.float16(np.inf))
        latent = unpatchify_tokens(expected)

        metrics = verify_tensors(actual, expected, latent, gate=0.01)

        self.assertEqual(metrics["verification"], "PASS")
        self.assertEqual(metrics["token_bit_mismatches"], 1)
        self.assertEqual(metrics["latent_bit_mismatches"], 1)

    def test_final_head_verifier_rejects_bad_unpatchify_reference(self):
        tokens = np.zeros((256, 16), dtype=np.float16)
        bad_latent = np.ones((4, 32, 32), dtype=np.float16)
        with self.assertRaisesRegex(ValueError, "reference latent"):
            verify_tensors(tokens, tokens, bad_latent)

    def test_final_head_verifier_enforces_relative_error_gate(self):
        expected = np.zeros((256, 16), dtype=np.float16)
        actual = expected.copy()
        expected[0, 0] = 1.0
        actual[0, 0] = 1.25
        with self.assertRaisesRegex(ValueError, "relative-to-peak"):
            verify_tensors(actual, expected, unpatchify_tokens(expected), gate=0.01)


if __name__ == "__main__":
    unittest.main()

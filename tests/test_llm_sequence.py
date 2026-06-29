from __future__ import annotations

import unittest

import torch

from model.llm import GPTPhase1Config, GPTPhase1Model


class GPTPhase1SequenceTest(unittest.TestCase):
    def test_audio_codes_are_offset_into_shared_vocab(self) -> None:
        model = GPTPhase1Model(
            GPTPhase1Config(
                d_model=16,
                n_head=2,
                n_layer=2,
                mlp_ratio=2,
                max_seq_len=128,
                postnet_hidden_layers=[2],
            )
        )

        token_ids = model.audio_codes_to_token_ids(torch.tensor([0, 10, 1023]))

        self.assertEqual(token_ids.tolist(), [2048, 2058, 3071])
        self.assertTrue(model.is_audio_token(token_ids).all())
        self.assertEqual(model.token_ids_to_audio_codes(token_ids).tolist(), [0, 10, 1023])

    def test_training_sequence_interleaves_text_and_audio_segments(self) -> None:
        model = GPTPhase1Model(
            GPTPhase1Config(
                d_model=16,
                n_head=2,
                n_layer=2,
                mlp_ratio=2,
                max_seq_len=128,
                postnet_hidden_layers=[2],
            )
        )

        sequence = model.build_training_sequence(
            [
                ("xin chao", torch.tensor([1, 2])),
                ("tam biet", torch.tensor([3])),
            ]
        )

        self.assertEqual(sequence[0].item(), model.bos_id)
        self.assertIn(model.audio_start_id, sequence.tolist())
        self.assertIn(model.audio_end_id, sequence.tolist())
        self.assertEqual(sequence[-1].item(), model.eos_id)
        self.assertIn(2049, sequence.tolist())
        self.assertIn(2050, sequence.tolist())
        self.assertIn(2051, sequence.tolist())


if __name__ == "__main__":
    unittest.main()

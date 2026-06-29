from __future__ import annotations

import unittest

import torch

from model.llm import GPTPhase1Config, GPTPhase1Model
from model.schemas import Phase1Prompt, VoiceCloneReference


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

    def test_code_loss_targets_only_audio_codes_and_audio_end(self) -> None:
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

        labels = sequence[1:]
        mask = model.is_code_loss_target(labels)
        supervised = labels[mask].tolist()
        unsupervised = labels[~mask].tolist()

        self.assertEqual(mask.sum().item(), 5)
        self.assertEqual(supervised.count(model.audio_end_id), 2)
        self.assertIn(2049, supervised)
        self.assertIn(2050, supervised)
        self.assertIn(2051, supervised)
        self.assertNotIn(model.audio_start_id, supervised)
        self.assertNotIn(model.eos_id, supervised)
        self.assertIn(model.audio_start_id, unsupervised)
        self.assertIn(model.eos_id, unsupervised)

    def test_forward_train_returns_code_loss_mask_for_label_positions(self) -> None:
        model = GPTPhase1Model(
            GPTPhase1Config(
                d_model=16,
                n_head=2,
                n_layer=2,
                mlp_ratio=2,
                max_seq_len=128,
                reference_dim=4,
                num_audio_layers=2,
                max_reference_frames=4,
                postnet_hidden_layers=[2],
            )
        )
        prompt = Phase1Prompt(
            voice_clone_text="xin chao",
            target_text="tam biet",
            voice_clone_reference=VoiceCloneReference(
                text="xin chao",
                timestamps=[],
                continuous_rvq_layers=torch.zeros(2, 2, 4),
                source_sample_id="sample",
            ),
        )
        sequence = model.build_training_sequence([("tam biet", torch.tensor([4, 5]))])

        output = model.forward_train(prompt, sequence)

        self.assertEqual(output.code_loss_mask.shape[0], output.logits.shape[0])
        self.assertEqual(output.code_loss_mask.sum().item(), 3)
        self.assertEqual(output.audio_target_ids.tolist(), [2052, 2053])


if __name__ == "__main__":
    unittest.main()

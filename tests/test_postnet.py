from __future__ import annotations

import unittest

import torch

from model.components.postnet import ResidualPostNet, ResidualPostNetConfig


class ResidualPostNetTest(unittest.TestCase):
    def test_postnet_projects_layer0_and_hidden_states_to_full_acoustic_embedding(self) -> None:
        postnet = ResidualPostNet(
            ResidualPostNetConfig(
                layer0_dim=8,
                llm_hidden_dim=16,
                output_dim=8,
                model_dim=32,
                num_layers=32,
                num_steps=4,
            )
        )
        layer0 = torch.randn(20, 8)
        hidden = torch.randn(20, 16)

        output = postnet(layer0, hidden)

        self.assertEqual(output.shape, (20, 32, 8))

    def test_postnet_aligns_mismatched_sequence_lengths(self) -> None:
        postnet = ResidualPostNet(
            ResidualPostNetConfig(
                layer0_dim=8,
                llm_hidden_dim=16,
                output_dim=8,
                model_dim=32,
            )
        )
        layer0 = torch.randn(12, 8)
        hidden = torch.randn(9, 16)

        output = postnet(layer0, hidden)

        self.assertEqual(output.shape[0], 9)


if __name__ == "__main__":
    unittest.main()

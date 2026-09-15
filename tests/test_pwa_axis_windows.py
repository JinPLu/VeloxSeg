import unittest
from math import prod
from unittest import mock

import torch

from model.VeloxSeg import VeloxSeg
from model.components.PWA import Paired_Windows_Attention, Paired_Windows_TransformerBlock
from nnunetv2.experiment_planning.experiment_planners import veloxseg_rules
from nnunetv2.experiment_planning.experiment_planners.veloxseg_rules import (
    architecture_for_patch,
    attention_window,
    divisors,
    jlc_group_width,
)


def pwa_windows(input_size, big_window, small_window=(1, 1, 1)):
    module = Paired_Windows_Attention(list(input_size), 16, list(big_window), list(small_window),
                                      num_heads=1, min_dim_head=4, dim=len(input_size))
    return module.big_window_size, module.small_window_size


class PlannerWindowTests(unittest.TestCase):
    def test_reference_96_model_windows_and_group_widths(self):
        stages = architecture_for_patch([96, 96, 96], [1.0, 1.0, 1.0], [4], 'B')['arch_kwargs']['stages']
        self.assertEqual([s['big_window'] for s in stages], [[3, 3, 3], [6, 6, 6], [3, 3, 3], [3, 3, 3]])
        self.assertEqual([s['group_width'] for s in stages], [4, 8, 8, 16])

    def test_thin_axis_keeps_in_plane_isotropic_window(self):
        self.assertEqual(attention_window([3, 32, 32], [4.0, 4.0, 4.0], 0, False), [3, 4, 4])

    def test_empty_reference_band_still_prefers_balanced_window(self):
        # No legal window of 40x48x40 is near 27 tokens; the band around the best
        # achievable count keeps the balanced 5x6x5 against the nearer 5x3x5.
        self.assertEqual(attention_window([40, 48, 40], [4.0, 4.0, 4.0], 0, False), [5, 6, 5])

    def test_physical_balance_uses_feature_spacing(self):
        # AutoPET stage 1: 4x5x4 is 48x41x33 mm; 2x5x4 (fewer tokens) is 24x41x33 mm.
        self.assertEqual(attention_window([64, 80, 64], [12.0, 8.15, 8.15], 0, False), [4, 5, 4])

    def test_isotropic_grid_keeps_isotropic_window_within_token_band(self):
        # Hecktor stage 1: 5x2x2 is nearest to 27 tokens but 20x8x8 mm; 5x4x4 stays in the band.
        self.assertEqual(attention_window([40, 64, 64], [4.0, 4.0, 4.0], 0, False), [5, 4, 4])

    def test_odd_anisotropic_grid_windows_tile_with_power_of_two_ratios(self):
        shape = [5, 12, 20]
        window = attention_window(shape, [6.0, 2.0, 2.0], 0, False)
        for n, size in zip(shape, window):
            ratio, remainder = divmod(n, size)
            self.assertEqual(remainder, 0)
            self.assertEqual(ratio & (ratio - 1), 0)
            self.assertGreaterEqual(size, 2)

    def test_final_stage_attends_over_whole_grid(self):
        self.assertEqual(attention_window([3, 5, 7], [1.0, 1.0, 1.0], 3, True), [3, 5, 7])


class PwaAxisExpansionTests(unittest.TestCase):
    def test_equal_ratios_reproduce_uniform_doubling(self):
        big, small = pwa_windows((24, 24, 24), (3, 3, 3))
        self.assertEqual(big, [[3, 3, 3], [6, 6, 6], [12, 12, 12], [24, 24, 24]])
        self.assertEqual(small, [[1, 1, 1], [2, 2, 2], [4, 4, 4], [8, 8, 8]])

    def test_thin_axis_stops_growing_while_others_reach_the_grid(self):
        big, small = pwa_windows((3, 32, 32), (3, 4, 4))
        self.assertEqual(big, [[3, 4, 4], [3, 8, 8], [3, 16, 16], [3, 32, 32]])
        self.assertEqual(small, [[1, 1, 1], [1, 2, 2], [1, 4, 4], [1, 8, 8]])

    def test_token_grid_is_constant_and_last_scale_covers_the_grid(self):
        for shape, window, small in [((5, 12, 20), (5, 3, 5), (1, 1, 1)),
                                     ((40, 64, 64), (5, 2, 2), (1, 1, 1)),
                                     ((8, 32, 16), (4, 4, 2), (2, 2, 1))]:
            big_sizes, small_sizes = pwa_windows(shape, window, small)
            grids = {tuple(b // s for b, s in zip(bw, sw)) for bw, sw in zip(big_sizes, small_sizes)}
            self.assertEqual(grids, {tuple(b // s for b, s in zip(window, small))})
            self.assertEqual(big_sizes[0], list(window))
            self.assertEqual(small_sizes[0], list(small))
            self.assertEqual(big_sizes[-1], list(shape))
            self.assertEqual(len(big_sizes), 1 + max((n // w).bit_length() - 1 for n, w in zip(shape, window)))

    def test_two_dimensional_windows_share_the_axis_rule(self):
        big, small = pwa_windows((6, 16), (3, 2))
        self.assertEqual(big, [[3, 2], [6, 4], [6, 8], [6, 16]])
        self.assertEqual(small, [[1, 1], [2, 2], [2, 4], [2, 8]])

    def test_rejects_non_power_of_two_ratio(self):
        with self.assertRaisesRegex(ValueError, 'power-of-two'):
            pwa_windows((12, 12, 12), (4, 4, 4))


class JlcGroupWidthTests(unittest.TestCase):
    def test_interpolated_widths_are_divisors_rounded_up(self):
        # 2^7 voxels per position: log2 width 2 + 1/3, so 5.04 rounds up to 8 of 32.
        self.assertEqual(jlc_group_width([2, 8, 8], 32), 8)
        # 2^14: 3 + 2/3 -> 12.7 -> 16 of 128.
        self.assertEqual(jlc_group_width([32, 32, 16], 128), 16)
        for channels in (16, 32, 48, 64, 128):
            for stride in ([1, 4, 4], [4, 4, 4], [8, 8, 8], [16, 16, 8], [64, 64, 64]):
                width = jlc_group_width(stride, channels)
                self.assertIn(width, divisors(channels))

    def test_beyond_anchor_policies(self):
        self.assertEqual(jlc_group_width([64, 64, 64], 128), 16)
        self.assertEqual(jlc_group_width([1, 4, 4], 16), 4)
        with mock.patch.dict(veloxseg_rules.JLC_GROUP_WIDTH, {'beyond_anchors': 'extrapolate'}):
            self.assertEqual(jlc_group_width([64, 64, 64], 128), 32)
            self.assertEqual(jlc_group_width([1, 2, 2], 16), 2)


class AnisotropicModelTests(unittest.TestCase):
    def test_forward_backward_reaches_every_pwa_parameter(self):
        stages = [
            {'stride': [4, 4, 4], 'channels': 16, 'kernels': [[1, 1, 1], [3, 3, 3]],
             'conv_depth': 1, 'attn_depth': 1, 'group_width': 4, 'conv_expansion': 2,
             'attn_expansion': 2, 'heads': 1, 'head_dim': 4, 'big_window': [3, 2, 2],
             'small_window': [1, 1, 1]},
            {'stride': [1, 2, 2], 'channels': 32, 'kernels': [[1, 1, 1], [3, 3, 3]],
             'conv_depth': 1, 'attn_depth': 1, 'group_width': 8, 'conv_expansion': 2,
             'attn_expansion': 2, 'heads': 2, 'head_dim': 8, 'big_window': [3, 4, 4],
             'small_window': [1, 1, 1]},
        ]
        torch.manual_seed(0)
        network = VeloxSeg([12, 32, 32], [1, 1], stages, n_classes=2, dropout=0.0)
        attention = [m for m in network.modules() if isinstance(m, Paired_Windows_Attention) and m.num_heads > 0]
        self.assertEqual([m.big_window_size for m in attention],
                         [[[3, 2, 2], [3, 4, 4], [3, 8, 8]], [[3, 4, 4]]])
        outputs = network(torch.randn(1, 2, 12, 32, 32))
        loss = sum(output.float().mean() for output in outputs)
        loss.backward()
        pwa_parameters = [(name, p) for m in attention for name, p in m.named_parameters()]
        self.assertTrue(pwa_parameters)
        for name, parameter in pwa_parameters:
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_block_adds_each_residual_once(self):
        # Pre-norm residual block (Swin): with both branch outputs zeroed the block is the identity.
        torch.manual_seed(0)
        block = Paired_Windows_TransformerBlock([3, 8, 8], [16, 16], [3, 2, 2], [1, 1, 1], num_heads=1,
                                                min_dim_head=4, attn_drop=0.0, proj_drop=0.0)
        for layer in (*block.attn.mix_channels, *(ffn.linear2 for ffn in block.ffns)):
            torch.nn.init.zeros_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
        inputs = [torch.randn(2, 16, 3, 8, 8) for _ in range(2)]
        for output, x in zip(block(inputs), inputs):
            torch.testing.assert_close(output, x)


if __name__ == '__main__':
    unittest.main()

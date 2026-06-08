import unittest

from utils.runtime import (
    a2fseg_deep_output_groups,
    expected_input_channels,
    image_label_modes,
    normalized_deep_loss_weights,
    rotation_range_from_degrees,
    resolve_modal_index,
    select_modal_items,
    validate_file_groups,
    validate_selected_modal,
    veloxseg_output_layout,
)


class RuntimeHelperTests(unittest.TestCase):
    def test_resolve_modal_index_selects_one_valid_modality(self):
        self.assertEqual(resolve_modal_index("1", 3), [0, 1, 0])

    def test_resolve_modal_index_rejects_out_of_range_modality(self):
        with self.assertRaisesRegex(ValueError, "select_modal"):
            resolve_modal_index("3", 3)

    def test_validate_file_groups_rejects_empty_groups(self):
        with self.assertRaisesRegex(ValueError, "No files matched"):
            validate_file_groups("AutoPETII", {"ct": [], "pet": ["pet.nii.gz"]})

    def test_validate_file_groups_rejects_mismatched_counts(self):
        with self.assertRaisesRegex(ValueError, "same number"):
            validate_file_groups(
                "AutoPETII",
                {"ct": ["ct-1.nii.gz"], "pet": ["pet-1.nii.gz", "pet-2.nii.gz"]},
            )

    def test_expected_input_channels_handles_veloxseg_in_ch(self):
        config = {"VeloxSeg": {"in_ch": [1, 1]}}
        self.assertEqual(expected_input_channels("VeloxSeg", config), 2)

    def test_expected_input_channels_handles_singular_input_channel(self):
        config = {"U-RWKV": {"input_channel": 4}}
        self.assertEqual(expected_input_channels("U-RWKV", config), 4)

    def test_validate_selected_modal_rejects_channel_mismatch(self):
        config = {"VeloxSeg": {"in_ch": [1, 1]}}
        with self.assertRaisesRegex(ValueError, "expects 2 input channel"):
            validate_selected_modal(
                model_name="VeloxSeg",
                model_config=config,
                raw_modal_count=2,
                select_modal="0",
            )

    def test_rotation_range_from_degrees_returns_radians(self):
        self.assertAlmostEqual(rotation_range_from_degrees(15), 0.2617993877991494)

    def test_image_label_modes_uses_nearest_for_label(self):
        self.assertEqual(
            image_label_modes(4),
            ("bilinear", "bilinear", "bilinear", "bilinear", "nearest"),
        )

    def test_normalized_deep_loss_weights_resize_equal_weights(self):
        self.assertEqual(
            normalized_deep_loss_weights([1, 1, 1, 1], 5),
            [0.2, 0.2, 0.2, 0.2, 0.2],
        )

    def test_normalized_deep_loss_weights_rejects_custom_mismatch(self):
        with self.assertRaisesRegex(ValueError, "deep_Loss_weight"):
            normalized_deep_loss_weights([4, 2, 1], 5)

    def test_normalized_deep_loss_weights_rejects_zero_sum(self):
        with self.assertRaisesRegex(ValueError, "sum"):
            normalized_deep_loss_weights([0, 0, 0, 0], 5)

    def test_a2fseg_deep_output_groups_covers_all_deep_outputs(self):
        self.assertEqual(
            a2fseg_deep_output_groups(26),
            [(1, 6), (6, 11), (11, 16), (16, 21), (21, 26)],
        )

    def test_a2fseg_deep_output_groups_rejects_partial_group(self):
        with self.assertRaisesRegex(ValueError, "A2FSeg"):
            a2fseg_deep_output_groups(18)

    def test_veloxseg_output_layout_handles_four_deep_outputs(self):
        self.assertEqual(
            veloxseg_output_layout(output_count=8, num_modal=2),
            {
                "seg": (0, 4),
                "reconstruction": 4,
                "decoder_gram": 5,
                "teacher_grams": (6, 7),
            },
        )

    def test_veloxseg_output_layout_handles_single_deep_output(self):
        self.assertEqual(
            veloxseg_output_layout(output_count=5, num_modal=2),
            {
                "seg": (0, 1),
                "reconstruction": 1,
                "decoder_gram": 2,
                "teacher_grams": (3, 4),
            },
        )

    def test_veloxseg_output_layout_rejects_missing_tail(self):
        with self.assertRaisesRegex(ValueError, "VeloxSeg"):
            veloxseg_output_layout(output_count=4, num_modal=2)

    def test_select_modal_items_filters_like_modal_index(self):
        self.assertEqual(
            select_modal_items(["flair", "t1", "t1ce", "t2"], [0, 1, 0, 0]),
            ["t1"],
        )


if __name__ == "__main__":
    unittest.main()

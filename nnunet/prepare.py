"""Prepare BraTS2021 and AutoPET-II for native nnU-Net planning."""
import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np
import SimpleITK as sitk
from nnunetv2.utilities.crossval_split import generate_crossval_split


def copy_BraTS_segmentation_and_convert_labels_to_nnUNet(in_file, out_file):
    """Label conversion from nnUNet's Apache-2.0 Dataset137_BraTS21.py."""
    image = sitk.ReadImage(in_file)
    labels = sitk.GetArrayFromImage(image)
    if not np.isin(labels, [0, 1, 2, 4]).all():
        raise ValueError(f"Expected original BraTS labels 0, 1, 2, 4: {in_file}")
    converted = np.zeros_like(labels)
    converted[labels == 1] = 2
    converted[labels == 2] = 1
    converted[labels == 4] = 3
    output = sitk.GetImageFromArray(converted)
    output.CopyInformation(image)
    sitk.WriteImage(output, out_file)


def prepare(dataset, source):
    config_root = Path(__file__).resolve().parent / "config"
    config = next(config_root.glob(f"Dataset{dataset}_*"))
    raw = Path(os.environ["nnUNet_raw"]) / config.name
    preprocessed = Path(os.environ["nnUNet_preprocessed"]) / config.name
    for destination in (raw, preprocessed):
        if destination.exists():
            raise FileExistsError(f"Use new data/output directories; {destination} already exists")

    if dataset == 137:
        cases = sorted(p.name for p in source.glob("BraTS*") if p.is_dir())
        channels = ("t1", "t1ce", "t2", "flair")
        images = lambda c: [source / c / f"{c}_{m}.nii.gz" for m in channels]
        label = lambda c: source / c / f"{c}_seg.nii.gz"
        expected = 1251
    else:
        cases = sorted(p.name.removesuffix(".nii.gz") for p in (source / "labelsTr").glob("*.nii.gz"))
        images = lambda c: [source / "imagesTr" / f"{c}_{i:04d}.nii.gz" for i in range(2)]
        label = lambda c: source / "labelsTr" / f"{c}.nii.gz"
        expected = 1014

    if len(cases) != expected:
        raise ValueError(f"The dataset expects {expected} cases, found {len(cases)}")
    if dataset == 137:
        training_cases = set(cases)
        splits = generate_crossval_split(cases, seed=12345, n_splits=5)
    else:
        splits = json.loads((config / "splits_final.json").read_text())
        training_cases = set(splits[4]["train"] + splits[4]["val"])
        boundary = int(len(cases) * 0.6) + int(len(cases) * 0.2)
        if set(cases[:boundary]) != training_cases:
            raise ValueError("Input case IDs differ from the published training/validation split")
    for case in cases:
        for path in [*images(case), label(case)]:
            if not path.is_file():
                raise FileNotFoundError(path)

    folders = ("imagesTr", "labelsTr") if dataset == 137 else ("imagesTr", "labelsTr", "imagesTs", "labelsTs")
    for folder in folders:
        (raw / folder).mkdir(parents=True)
    for i, case in enumerate(cases):
        split = "Tr" if case in training_cases else "Ts"
        for channel, path in enumerate(images(case)):
            shutil.copyfile(path, raw / f"images{split}" / f"{case}_{channel:04d}.nii.gz")
        output_label = raw / f"labels{split}" / f"{case}.nii.gz"
        if dataset == 137:
            copy_BraTS_segmentation_and_convert_labels_to_nnUNet(str(label(case)), str(output_label))
        else:
            shutil.copyfile(label(case), output_label)
        print(f"{i + 1}/{len(cases)} {case}", flush=True)

    preprocessed.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config / "dataset.json", raw / "dataset.json")
    shutil.copyfile(config / "dataset.json", preprocessed / "dataset.json")
    (preprocessed / "splits_final.json").write_text(json.dumps(splits, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=int, choices=[137, 221])
    parser.add_argument("source", type=Path, help="Raw BraTS2021 or AutoPETII_spac_norm directory")
    args = parser.parse_args()
    prepare(args.dataset, args.source)

from glob import glob
import os
import numpy as np
import SimpleITK as sitk
import shutil

def zscore(image, seg):
    mask = seg >= 0
    mean = image[mask].mean()
    std = image[mask].std()
    image[mask] = (image[mask] - mean) / (max(std, 1e-8))
    return image

def save_nii(input_, output_, save_path):
    origin = input_.GetOrigin()
    direction = input_.GetDirection()
    space = input_.GetSpacing()
    savedImg = sitk.GetImageFromArray(output_)
    savedImg.SetOrigin(origin)
    savedImg.SetDirection(direction)
    savedImg.SetSpacing(space)
    sitk.WriteImage(savedImg, save_path)

def CTnorm(image, intensityproperties):
    mean_intensity = intensityproperties['mean']
    std_intensity = intensityproperties['std']
    lower_bound = intensityproperties['percentile_00_5']
    upper_bound = intensityproperties['percentile_99_5']
    image = np.clip(image, lower_bound, upper_bound)
    image = (image - mean_intensity) / max(std_intensity, 1e-8)
    return image

root_path = "./datasets/BraTS2021"
save_path = "./datasets/BraTS2021_spac_norm/"

flair_files = sorted(glob(os.path.join(root_path, "*", "*flair.nii.gz")))
t1_files = sorted(glob(os.path.join(root_path, "*", "*t1.nii.gz")))
t1ce_files = sorted(glob(os.path.join(root_path, "*", "*t1ce.nii.gz")))
t2_files = sorted(glob(os.path.join(root_path, "*", "*t2.nii.gz")))
label_files = sorted(glob(os.path.join(root_path, "*", "*seg.nii.gz")))

n = len(label_files)

def preprocess(file, step, n):
    name = file.split("/")[-2]
    save_path_name = os.path.join(save_path, name)
    os.makedirs(save_path_name, exist_ok=True)
    
    Image = sitk.ReadImage(file)
    data = sitk.GetArrayFromImage(Image)

    preprocessed_data = zscore(data, data>data.min())
    save_nii(Image, preprocessed_data, os.path.join(save_path_name, file.split('/')[-1]))
    print(f"{step} / {n}: {os.path.basename(file)}, shape: {data.shape}")
for files in [flair_files, t1_files, t1ce_files, t2_files]:
    n = len(files)
    [preprocess(file, step, n) for step, file in enumerate(files)]

for step, file in enumerate(label_files):
    name = file.split("/")[-2]
    save_path_name = os.path.join(save_path, name)
    
    Image = sitk.ReadImage(file)
    data = sitk.GetArrayFromImage(Image)
    
    data[data == 4] = 3
    save_nii(Image, data, os.path.join(save_path_name, file.split('/')[-1]))
    print(f"{step} / {n}: {os.path.basename(file)}")
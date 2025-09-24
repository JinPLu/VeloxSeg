from glob import glob
import os
import numpy as np
import SimpleITK as sitk

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

ct_root = './datasets/Task221_AutoPETII/imagesTr'
pet_root = './datasets/Task221_AutoPETII/imagesTr'
mask_root = './datasets/Task221_AutoPETII/labelsTr'

ct_save = './datasets/AutoPETII_spac_norm/imagesTr'
pet_save = './datasets/AutoPETII_spac_norm/imagesTr'
mask_save = './datasets/AutoPETII_spac_norm/labelsTr'

images_CT = sorted(glob(os.path.join(ct_root, "*0001.nii.gz")))
images_PET = sorted(glob(os.path.join(pet_root, "*0000.nii.gz")))
labels = sorted(glob(os.path.join(mask_root, "*.nii.gz")))

n = len(images_CT)
[os.makedirs(save_path, exist_ok=True) for save_path in [ct_save, pet_save, mask_save]]

ct_voxels = []
for step, (ct_file, pet_file, mask_file) in enumerate(zip(images_CT, images_PET, labels)):
    ct_name = ct_file.split('/')[-1].replace('_0001', '')
    pet_name = pet_file.split('/')[-1].replace('_0000', '')
    mask_name = mask_file.split('/')[-1]
    assert ct_name == mask_name, f"ct_name is {ct_name}; mask_name is {mask_name}"
    assert pet_name == mask_name, f"ct_name is {pet_name}; mask_name is {mask_name}"
    ct = sitk.ReadImage(ct_file)
    pet = sitk.ReadImage(pet_file)
    mask = sitk.ReadImage(mask_file)
    
    ct_data = sitk.GetArrayFromImage(ct)
    pet_data = sitk.GetArrayFromImage(pet)
    mask_data = sitk.GetArrayFromImage(mask)
    if (np.array(ct_data.shape) != np.array(mask_data.shape)).any():
        shape = ct_data.shape
        mask_data = mask_data[:shape[0], :shape[1], :shape[2]]
        print(f'ct({ct_data.shape}):{ct_file}, old_label{mask_data.shape}:{mask_file}')
    ct_voxels += ct_data[mask_data > 0].reshape(-1).tolist()

    pet_preprocess_data = zscore(pet_data, pet_data>pet_data.min())
    save_nii(pet, pet_preprocess_data, os.path.join(pet_save, pet_file.split('/')[-1]))
    save_nii(mask, mask_data, os.path.join(mask_save, mask_file.split('/')[-1]))
    print(f"Step1: {step}/{n}", pet_preprocess_data.shape)

foreground_intensities = np.array(ct_voxels)
intensity_statistics = {
    'mean': float(np.mean(foreground_intensities)),
    'median': float(np.median(foreground_intensities)),
    'std': float(np.std(foreground_intensities)),
    'min': float(np.min(foreground_intensities)),
    'max': float(np.max(foreground_intensities)),
    'percentile_99_5': float(np.percentile(foreground_intensities, 99.5)),
    'percentile_00_5': float(np.percentile(foreground_intensities, 0.5)),
}

print('\n\n', intensity_statistics, '\n\n')
for step, ct_file in enumerate(images_CT):
    ct = sitk.ReadImage(ct_file)
    ct_data = sitk.GetArrayFromImage(ct)
    ct_preprocessed_data = CTnorm(ct_data, intensity_statistics)
    save_nii(ct, ct_preprocessed_data, os.path.join(ct_save, ct_file.split('/')[-1]))
    print(f"Step2: {step}/{n}")
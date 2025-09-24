import os
import ants
from glob import glob
# from time import sleep
# import multiprocessing as mp
# from tqdm import tqdm
# from datetime import datetime

def get_config():
    return {
        'ct_path': '/datasets/Hecktor2022/imagesTr/*__CT.nii.gz',
        'pet_path': '/datasets/Hecktor2022/imagesTr/*__PT.nii.gz',
        'label_path': '/datasets/Hecktor2022/labelsTr/*.nii.gz',
        'ct_save_path': '/datasets/Hecktor2022_registrations/imagesTr',
        'pet_save_path': '/datasets/Hecktor2022_registrations/imagesTr',
        'label_save_path': '/datasets/Hecktor2022_registrations/labelsTr',
        "spacing": (1, 1, 1), # 如果不需要固定spacing，则设置为None,
        "num_threads": 4,
    }

def registration(ct_file, pet_file, label_file, ct_save_path, pet_save_path, label_save_path, spacing=None):
    os.makedirs(ct_save_path, exist_ok=True)
    os.makedirs(pet_save_path, exist_ok=True)
    os.makedirs(label_save_path, exist_ok=True)
    ct_name = os.path.basename(ct_file)
    pet_name = os.path.basename(pet_file)
    label_name = os.path.basename(label_file)

    # if os.path.exists(os.path.join(ct_save_path, ct_name)):
    #     print(f'{ct_name} already exists, skipping.')
    #     return
    # if os.path.exists(os.path.join(pet_save_path, pet_name)):
    #     print(f'{pet_name} already exists, skipping.')
    #     return
    # if os.path.exists(os.path.join(label_save_path, label_name)):
    #     print(f'{label_name} already exists, skipping.')
    #     return

    ct_img = ants.image_read(ct_file)
    pet_img = ants.image_read(pet_file)
    label_img = ants.image_read(label_file)	

    if spacing is not None:
        ants.set_spacing(ct_img, spacing)

    outs = ants.registration(ct_img, pet_img, type_of_transforme = 'Affine')  

    reg_pet_img = outs['warpedmovout']  
    reg_label_img = ants.apply_transforms(
                                        fixed = ct_img, 
                                        moving = label_img,
                                        transformlist= outs['fwdtransforms'], 
                                        interpolator = 'nearestNeighbor')  

    ants.image_write(ct_img, os.path.join(ct_save_path, ct_name))
    ants.image_write(reg_pet_img, os.path.join(pet_save_path, pet_name))
    ants.image_write(reg_label_img, os.path.join(label_save_path, label_name))


if __name__ == '__main__':
    config = get_config()

    ct_files = sorted(glob(config['ct_path']))
    pet_files = sorted(glob(config['pet_path']))
    label_files = sorted(glob(config['label_path']))

    n = len(ct_files)
    r = []
    for i in range(n):
        print(f"{i} / {n}")
        registration(ct_files[i], pet_files[i], label_files[i], 
                    config['ct_save_path'], config['pet_save_path'], 
                    config['label_save_path'], 
                    config['spacing'])
    # with mp.get_context("spawn").Pool(config['num_threads']) as p:
    #     remaining = list(range(n))
    #     # p is pretty nifti. If we kill workers they just respawn but don't do any work.
    #     # So we need to store the original pool of workers.
    #     workers = [j for j in p._pool]
            
    #     for i in range(n):
    #         r.append(p.starmap_async(registration, 
    #                                     ((ct_files[i], pet_files[i], label_files[i], 
    #                                     config['ct_save_path'], config['pet_save_path'], 
    #                                     config['label_save_path'], 
    #                                     config['spacing']),)))

    #     with tqdm(desc=f'Time: {datetime.now().strftime("%H:%M:%S")}', total=n) as pbar:
    #         while len(remaining) > 0:
    #             all_alive = all([j.is_alive() for j in workers])
    #             if not all_alive:
    #                 raise RuntimeError('Some background worker is 6 feet under. Yuck. \n'
    #                                 'OK jokes aside.\n'
    #                                 'One of your background processes is missing. This could be because of '
    #                                 'an error (look for an error message) or because it was killed '
    #                                 'by your OS due to running out of RAM. If you don\'t see '
    #                                 'an error message, out of RAM is likely the problem. In that case '
    #                                 'reducing the number of workers might help')
    #             done = [i for i in remaining if r[i].ready()]
    #             # get done so that errors can be raised
    #             _ = [r[i].get() for i in done]
    #             for _ in done:
    #                 r[_].get()  # allows triggering errors
    #                 pbar.update()
    #             remaining = [i for i in remaining if i not in done]
    #             sleep(0.1)
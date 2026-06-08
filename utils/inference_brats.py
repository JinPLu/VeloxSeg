import os
import warnings
import torch
warnings.filterwarnings("ignore")

import random
import numpy as np

seed = 12345
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

import pandas as pd
from datetime import datetime
import nibabel as nib
from .inference_runtime import sliding_window_predict
from .load_model import load_model, checkpoint_DDP_to_SingleGPU
from .metric.metrics_brats import cal_dice, cal_hd95
from .get_logger import get_logger
from .runtime import (
    get_torch_device,
    select_modal_items,
    set_cuda_device_if_available,
    validate_file_groups,
    validate_selected_modal,
)
import json

import pytorch_lightning
from glob import glob
import nibabel as nib
from monai.data import list_data_collate, DataLoader, Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    ToTensord,
    EnsureChannelFirstd,
)

class Net(pytorch_lightning.LightningModule):
    def __init__(self, args, model_config, train_config, test_config):
        super().__init__()
        self.args = args
        self.train_config = train_config
        self.test_config = test_config
        self._model = load_model(args.model_name, model_config)
    
    def forward(self, x):
        outputs = self._model(x)
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
        return outputs

    def prepare_data(self):

        split_1 = self.train_config['train_rate'] + self.train_config['val_rate']
        images_flair = sorted(glob(self.train_config['dataset_path'][self.args.dataset_name]["flair_path"]))
        images_t1 = sorted(glob(self.train_config['dataset_path'][self.args.dataset_name]["t1_path"]))
        images_t1ce = sorted(glob(self.train_config['dataset_path'][self.args.dataset_name]["t1ce_path"]))
        images_t2 = sorted(glob(self.train_config['dataset_path'][self.args.dataset_name]["t2_path"]))
        labels = sorted(glob(self.train_config['dataset_path'][self.args.dataset_name]["label_path"]))
        validate_file_groups(
            self.args.dataset_name,
            {
                "flair": images_flair,
                "t1": images_t1,
                "t1ce": images_t1ce,
                "t2": images_t2,
                "label": labels,
            },
        )
        names = [filename.split("/")[-2] for filename in labels]

        files = [{"flair": flair, "t1": t1,"t1ce": t1ce, "t2":t2, "seg":seg, "name":name} \
                        for flair, t1, t1ce, t2, seg, name in zip(images_flair, images_t1, images_t1ce, 
                                                            images_t2, labels, names)]
        length = len(files)
        test_files = files[int(length*split_1):]
        
        transforms = Compose(
            [
                LoadImaged(keys=["flair", "t1", 't1ce', 't2', "seg"]),
                EnsureChannelFirstd(keys=["flair", "t1", 't1ce', 't2', "seg"]),
                ToTensord(keys=["flair", "t1", 't1ce', 't2', ]),
            ]
        )
        if self.args.specific_sample is None:
            self.val_ds = Dataset(data=test_files, transform=transforms)
        else:
            if self.args.specific_sample < 0 or self.args.specific_sample >= len(test_files):
                raise IndexError(
                    f"specific_sample index {self.args.specific_sample} is out "
                    f"of range for {len(test_files)} samples"
                )
            self.val_ds = Dataset(data=test_files[self.args.specific_sample:self.args.specific_sample+1], transform=transforms)

    def get_val_dataloader(self):
        val_loader = DataLoader(
            self.val_ds, batch_size=1, num_workers=self.args.num_workers, collate_fn=list_data_collate)
        self.val_dataloader = val_loader


    def load_from_checkpoint(self, checkpoint_path):
        checkpoint = checkpoint_DDP_to_SingleGPU(torch.load(checkpoint_path, map_location='cpu')['model'])
        self._model.load_state_dict(checkpoint)


def run_Inference(args):

    date = datetime.now().strftime("%m_%d") if args.train_date is None else args.train_date
    pred_path = None
    model_index = f"_{args.model_index}" if args.model_index is not None else ""

    with open(args.train_config, 'r', encoding='utf-8') as f:
        train_config = json.load(f)

    with open(args.test_config, 'r', encoding='utf-8') as f:
        test_config = json.load(f)

    if args.checkpoint_dir is None:
        assert args.train_date is not None, "Please specify the date of the model to be tested, when there is no checkpoint directory specified."
        args.checkpoint_dir = os.path.join(train_config['save_path'], args.dataset_name, args.model_name, args.train_date + model_index)
    
    if args.model_config is None:
        assert args.train_date is not None, "Please specify the date of the model to be tested, when there is no model config specified."
        args.model_config = os.path.join(train_config['config_path'], "model_config_" + args.train_date + model_index + ".json")

    with open(args.model_config, 'r', encoding='utf-8') as f:
        model_config = json.load(f)

    metrics_path = os.path.join(test_config['result_metric_path'], args.dataset_name, args.model_name, date + model_index)
    os.makedirs(metrics_path, exist_ok=True)
    
    if args.specific_sample is None:
        log_path = os.path.join(train_config['log_path'], "Test", args.model_name)
        os.makedirs(log_path, exist_ok=True)
        logger = get_logger(save_dir = log_path,
                            distributed_rank = 0,
                            filename = f"{date}_{model_index}_{args.dataset_name}_{args.checkpoint_index}.log",
                            mode='w')
    else:
        logger = get_logger(save_dir = None,
                            distributed_rank = 0,
                            stdout = False)
    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint_index + '.pth')
    file_path = os.path.join(metrics_path, f"{args.checkpoint_index}.csv")
    if args.specific_sample is not None:
        pred_path = os.path.join(test_config['result_pred_path'], args.dataset_name, str(args.specific_sample))
        os.makedirs(pred_path, exist_ok=True)
    segment_MRI(args, logger, model_config, train_config, test_config, checkpoint_path, pred_path, file_path)
    logger.handlers.clear()
        

def tensor2numpy(x):
    return x.detach().cpu().numpy()

def numpy2tensor(x, device):
    return torch.tensor(x).to(device)

def segment_MRI(args, logger, model_config, train_config, test_config, checkpoint_path, pred_path, file_path):
    modal_index = validate_selected_modal(
        args.model_name,
        model_config,
        raw_modal_count=4,
        select_modal=args.select_modal,
    )
    logger.info(f"model_config: {args.model_config}")
    logger.info(f"pred_path: {pred_path}")
    logger.info(f"metric_file_path: {file_path}")
    
    net = Net(args, model_config, train_config, test_config)
    net.prepare_data()

    net.load_from_checkpoint(checkpoint_path)
    logger.info(f"loaded checkpoint from {checkpoint_path}")

    device = get_torch_device(args.gpu_id)
    set_cuda_device_if_available(device)

    logger.info(f"device: {device}")
    
    net.to(device)
    net.eval()
    net.get_val_dataloader()
    length = len(net.val_dataloader)
    out = None
    result = []
    logger.info(f'length of dataset is {length}')
    with torch.inference_mode():
        for i, data in enumerate(net.val_dataloader):
            flair, t1, t1ce, t2, label, name = data["flair"].type(torch.FloatTensor).to(device),\
                                    data["t1"].type(torch.FloatTensor).to(device),\
                                    data["t1ce"].type(torch.FloatTensor).to(device),\
                                    data["t2"].type(torch.FloatTensor).to(device),\
                                    data["seg"].type(torch.LongTensor).to(device),\
                                    data['name']
            inputs = torch.cat(
                select_modal_items([flair, t1, t1ce, t2], modal_index),
                dim=1,
            )
            
            out = sliding_window_predict(
                inputs,
                net,
                train_config['patch_size'][args.dataset_name],
                train_config['batch_size'],
                test_config,
            )
            out = out.argmax(dim=1, keepdim=True)
            avg_dice, et_dice, tc_dice, wt_dice = cal_dice(out, label)
            if args.use_hd95 == 1:
                spacing = label.meta['pixdim'][0, 1:4].tolist()
                avg_hd95, et_hd95, tc_hd95, wt_hd95 = cal_hd95(out, label, spacing)
            else:
                avg_hd95, et_hd95, tc_hd95, wt_hd95 = -1, -1, -1, -1
            logger.info(f"{name[0]} {i}/{length} [avg_dice:{avg_dice:4f}, et_dice:{et_dice:4f}, tc_dice:{tc_dice:4f}, wt_dice:{wt_dice:4f}, pix:{(out>0).sum()}/{(label>0).sum()}]")
            result += [[avg_dice, et_dice, tc_dice, wt_dice, avg_hd95, et_hd95, tc_hd95, wt_hd95, float((out>0).sum()), float((label>0).sum())]]

            # save prediction map
            if args.specific_sample is not None:
                out = out[0][0]
                label = label[0][0]
                
                Image = nib.Nifti1Image((out == 3).cpu().numpy().astype(np.uint8), data['seg'].meta['affine'][0])
                nib.save(Image, os.path.join(pred_path, f"{args.model_name}_et.nii.gz"))
                Image = nib.Nifti1Image(((out == 1) | (out == 3)).cpu().numpy().astype(np.uint8), data['seg'].meta['affine'][0])
                nib.save(Image, os.path.join(pred_path, f"{args.model_name}_tc.nii.gz"))
                Image = nib.Nifti1Image((out != 0).cpu().numpy().astype(np.uint8), data['seg'].meta['affine'][0])
                nib.save(Image, os.path.join(pred_path, f"{args.model_name}_wt.nii.gz"))
                
                if not os.path.exists(os.path.join(pred_path, f"flair.nii.gz")):
                    
                    Image = nib.Nifti1Image(flair[0][0].cpu().numpy(), data['flair'].meta['affine'][0])
                    nib.save(Image, os.path.join(pred_path, f"flair.nii.gz"))
                    
                    Image = nib.Nifti1Image(t1[0][0].cpu().numpy(), data['t1'].meta['affine'][0])
                    nib.save(Image, os.path.join(pred_path, f"t1.nii.gz"))
                    
                    Image = nib.Nifti1Image(t1ce[0][0].cpu().numpy(), data['t1ce'].meta['affine'][0])
                    nib.save(Image, os.path.join(pred_path, f"t1ce.nii.gz"))
                    
                    Image = nib.Nifti1Image(t2[0][0].cpu().numpy(), data['t2'].meta['affine'][0])
                    nib.save(Image, os.path.join(pred_path, f"t2.nii.gz"))
                    
                    Image = nib.Nifti1Image((label == 3).cpu().numpy().astype(np.uint8), data['seg'].meta['affine'][0])
                    nib.save(Image, os.path.join(pred_path, f"et.nii.gz"))
                    Image = nib.Nifti1Image(((label == 1) | (label == 3)).cpu().numpy().astype(np.uint8), data['seg'].meta['affine'][0])
                    nib.save(Image, os.path.join(pred_path, f"tc.nii.gz"))
                    Image = nib.Nifti1Image((label != 0).cpu().numpy().astype(np.uint8), data['seg'].meta['affine'][0])
                    nib.save(Image, os.path.join(pred_path, f"wt.nii.gz"))

    result = pd.DataFrame(result)
    result.columns = ["Avg_Dice", "ET_Dice", "TC_Dice", "WT_Dice", "Avg_HD95", "ET_HD95", "TC_HD95", "WT_HD95", "Prediction", "Label"]
    result.to_csv(file_path, index=None)

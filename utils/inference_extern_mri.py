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
from monai.inferers import sliding_window_inference
from .load_model import load_model, checkpoint_DDP_to_SingleGPU
from .metric.metrics_brats import cal_dice, cal_hd95
from .get_logger import get_logger
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
    Spacingd,
    NormalizeIntensityd
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
        if type(outputs) == list:
            outputs = outputs[0]
        return outputs

    def prepare_data(self):

        if ('AutoPET' in self.args.test_dataset) or ('Hecktor' in self.args.test_dataset):
            images_CT = sorted(glob(self.train_config['dataset_path'][self.args.test_dataset][self.args.spacing]["ct_path"]))
            images_PET = sorted(glob(self.train_config['dataset_path'][self.args.test_dataset][self.args.spacing]["pet_path"]))
            labels = sorted(glob(self.train_config['dataset_path'][self.args.test_dataset][self.args.spacing]["label_path"]))
            names = [filename.split("_")[-2] for filename in images_CT]
        
            files = [{"img": img, "img_ct": ct, "label":label, "name":name} \
                        for img, ct, label, name in zip(images_PET, images_CT, labels, names)]
            transforms = Compose(
                [
                    LoadImaged(keys=["img", "img_ct", "label"]),
                    EnsureChannelFirstd(keys=["img", "img_ct", "label"]),
                    Spacingd(keys=["img", "img_ct", "label"], 
                            pixdim=self.train_config['spacing'][self.args.train_dataset], 
                            mode=("bilinear", "bilinear", "nearest")),
                    ToTensord(keys=["img", "img_ct", "label"]),
                ]
            )

        elif self.args.test_dataset == 'MSD2019':
            data_files = sorted(glob(self.train_config['dataset_path'][self.args.test_dataset]["data_path"]))
            label_files = sorted(glob(self.train_config['dataset_path'][self.args.test_dataset]["label_path"]))
            names = [filename.split("/")[-1].replace(".nii.gz", '') for filename in label_files]
        
            files = [{"data": data, "label":label, "name":name} \
                        for data, label, name in zip(data_files, label_files, names)]
            transforms = Compose(
                [
                    LoadImaged(keys=["data", "label"]),
                    EnsureChannelFirstd(keys=["data", "label"]),
                    NormalizeIntensityd(keys=['data'], channel_wise=True),
                    ToTensord(keys=["data", "label"]),
                ]
            )
        if self.args.specific_sample is None:
            self.val_ds = Dataset(data=files, transform=transforms)
        else:
            self.val_ds = Dataset(data=files[self.args.specific_sample:self.args.specific_sample+1], transform=transforms)

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
    print(model_index)

    with open(args.train_config, 'r', encoding='utf-8') as f:
        train_config = json.load(f)

    with open(args.test_config, 'r', encoding='utf-8') as f:
        test_config = json.load(f)
        
    args.checkpoint_dir = os.path.join(train_config['save_path'], args.train_dataset, args.model_name, args.train_date + model_index)

    if args.model_config is None:
        assert args.train_date is not None, "Please specify the date of the model to be tested, when there is no model config specified."
        args.model_config = os.path.join(train_config['config_path'], "model_config_" + args.train_date + model_index + ".json")

    with open(args.model_config, 'r', encoding='utf-8') as f:
        model_config = json.load(f)

    metrics_path = os.path.join(test_config['result_metric_path'], f"{args.train_dataset}_to_{args.test_dataset}", args.model_name, date + model_index)
    os.makedirs(metrics_path, exist_ok=True)
    
    for index in args.checkpoint_index.split(","):
        if args.specific_sample is None:
            log_path = os.path.join(train_config['log_path'], "Test", args.model_name)
            os.makedirs(log_path, exist_ok=True)
            logger = get_logger(save_dir = log_path,
                                distributed_rank = 0,
                                filename = f"{date}_{model_index}_{args.train_dataset}_to_{args.test_dataset}_{index}.log",
                                mode='w')
        else:
            logger = get_logger(save_dir = None,
                                distributed_rank = 0,
                                stdout = False)
        checkpoint_path = os.path.join(args.checkpoint_dir, index + '.pth')
        file_path = os.path.join(metrics_path, f"{index}.csv")
        if args.specific_sample is not None:
            pred_path = os.path.join(test_config['result_pred_path'], f"{args.train_dataset}_to_{args.test_dataset}", args.model_name)
            os.makedirs(pred_path, exist_ok=True)
        segment_PETCT_Spacing(args, logger, model_config, train_config, test_config, checkpoint_path, pred_path, file_path)
        logger.handlers.clear()
        

def tensor2numpy(x):
    return x.detach().cpu().numpy()

def numpy2tensor(x, device):
    return torch.tensor(x).to(device)

def segment_PETCT_Spacing(args, logger, model_config, train_config, test_config, checkpoint_path, pred_path, file_path):    
    
    net = Net(args, model_config, train_config, test_config)
    net.prepare_data()

    net.load_from_checkpoint(checkpoint_path)

    if args.gpu_id == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    
    net.to(device)
    net.eval()
    net.get_val_dataloader()
    length = len(net.val_dataloader)
    out = None
    result = []
    if args.specific_sample is None:
        logger.info(f"model_config: {args.model_config}")
        logger.info(f"pred_path: {pred_path}")
        logger.info(f"metric_file_path: {file_path}")
        logger.info(f"loaded checkpoint from {checkpoint_path}")
        logger.info(f"device: {device}")
        logger.info(f'length of dataset is {length}')
    with torch.no_grad():
        for i, data in enumerate(net.val_dataloader):
            data, label, name = data["data"].type(torch.FloatTensor).to(device),\
                                data["label"].type(torch.LongTensor).to(device),\
                                data["name"]
            
            out = sliding_window_inference(data, train_config['patch_size'][args.train_dataset], train_config['batch_size'],
                                                predictor=net, overlap=0.25)

            out = out.argmax(dim=1, keepdim=True)
            if args.specific_sample is None:
                avg_dice, et_dice, tc_dice, wt_dice = cal_dice(out, label)
                if args.use_hd95:
                    avg_hd95, et_hd95, tc_hd95, wt_hd95 = cal_hd95(out, label)
                else:
                    avg_hd95, et_hd95, tc_hd95, wt_hd95 = -1, -1, -1, -1
                logger.info(f"{name[0]} {i}/{length} [avg_dice:{avg_dice:.2f}, et_dice:{et_dice:.2f}, tc_dice:{tc_dice:.2f}, wt_dice:{wt_dice:.2f}, avg_hd:{avg_hd95:.2f}, et_hd:{et_hd95:.2f}, tc_hd:{tc_hd95:.2f}, wt_hd:{wt_hd95:.2f}, pix:{(out>0).sum()}/{(label>0).sum()}]")
                result += [[avg_dice, et_dice, tc_dice, wt_dice, avg_hd95, et_hd95, tc_hd95, wt_hd95, float((out>0).sum()), float((label>0).sum())]]

            # break
        result = pd.DataFrame(result)
        result.columns = ["Avg_Dice", "ET_Dice", "TC_Dice", "WT_Dice", "Avg_HD95", "ET_HD95", "TC_HD95", "WT_HD95", "Prediction", "Label"]
        result.to_csv(file_path, index=None)

    
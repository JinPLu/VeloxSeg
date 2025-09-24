import os
import warnings
import torch
warnings.filterwarnings("ignore")

import random
import numpy as np

from .seed import seed_everything
seed_everything(seed = 12345)

import pandas as pd
from datetime import datetime
import nibabel as nib
from monai.inferers import sliding_window_inference
from .load_model import load_model, checkpoint_DDP_to_SingleGPU
from .metric.metrics import metrics_tensor, get_hausdorff
from .get_logger import get_logger
import json
import time

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
        if type(outputs) == list:
            outputs = outputs[0]
        return outputs

    def prepare_data(self):

        split_1 = self.train_config['train_rate'] + self.train_config['val_rate']
        images_CT = sorted(glob(self.train_config['dataset_path'][self.args.dataset_name]["ct_path"]))
        images_PET = sorted(glob(self.train_config['dataset_path'][self.args.dataset_name]["pet_path"]))
        labels = sorted(glob(self.train_config['dataset_path'][self.args.dataset_name]["label_path"]))
        names = [filename.split("_")[-1].replace('.nii.gz', '') for filename in labels]
       
        files = [{"img": img, "img_ct": ct, "label":label, "name":name} \
                    for img, ct, label, name in zip(images_PET, images_CT, labels, names)]

        transforms = Compose(
            [
                LoadImaged(keys=["img", "img_ct", "label"]),
                EnsureChannelFirstd(keys=["img", "img_ct", "label"]),
                ToTensord(keys=["img", "img_ct", "label"]),
            ]
        )

        length = len(files)
        test_files = files[int(length*split_1):]
        # test_files = files[:int(length*split_1)]
        if self.args.specific_sample is None:
            self.val_ds = Dataset(data=test_files, transform=transforms)
        else:
            self.val_ds = Dataset(data=test_files[self.args.specific_sample:self.args.specific_sample+1], transform=transforms)

    def get_val_dataloader(self):
        val_loader = DataLoader(
            self.val_ds, batch_size=1, num_workers=self.args.num_workers, collate_fn=list_data_collate)
        self.val_dataloader = val_loader


    def load_from_checkpoint(self, checkpoint_path):
        ckt = torch.load(checkpoint_path, map_location='cpu')
        if 'model' in ckt.keys():
            checkpoint = checkpoint_DDP_to_SingleGPU(ckt['model'])
        else:
            checkpoint = checkpoint_DDP_to_SingleGPU(ckt)
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
    segment_PETCT(args, logger, model_config, train_config, test_config, checkpoint_path, pred_path, file_path)
    logger.handlers.clear()
        

def tensor2numpy(x):
    return x.detach().cpu().numpy()

def numpy2tensor(x, device):
    return torch.tensor(x).to(device)

def segment_PETCT(args, logger, model_config, train_config, test_config, checkpoint_path, pred_path, file_path):    
    modal_index = [0, 0]
    if args.select_modal is not None:
        modal_index[int(args.select_modal)] = 1
    else:
        modal_index = [1 for _ in range(len(modal_index))]
    
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
            ct, pet, label, name = data["img_ct"].type(torch.FloatTensor).to(device),\
                                   data["img"].type(torch.FloatTensor).to(device),\
                                   data["label"].type(torch.LongTensor).to(device),\
                                   data["name"]
            # inputs = torch.cat([ct, pet], dim=1)
            start = time.time()
            label[label != 0] = 1
            
            # print('**********', label.max())
            
            if args.specific_sample is not None:
                # if not os.path.exists(os.path.join(pred_path, f"ct.nii.gz")):
                Image = nib.Nifti1Image(ct[0][0].cpu().numpy(), data['img_ct'].meta['affine'][0])
                nib.save(Image, os.path.join(pred_path, f"ct.nii.gz"))
                
                Image = nib.Nifti1Image(pet[0][0].cpu().numpy(), data['img'].meta['affine'][0])
                nib.save(Image, os.path.join(pred_path, f"pet.nii.gz"))
                
                Image = nib.Nifti1Image(label[0][0].cpu().numpy().astype(np.uint8), data['label'].meta['affine'][0])
                nib.save(Image, os.path.join(pred_path, f"label.nii.gz"))
            
            
            _, _, h, w, d = label.size()
            inputs = [ct, pet]
            inputs = torch.cat([inputs[i] for i in range(len(inputs)) if modal_index[i]], dim=1)
            
            if h * w * d > 500*500*1000:
                out = sliding_window_inference(inputs, train_config['patch_size'][args.dataset_name], train_config['batch_size'], 
                                               predictor=net, overlap=0.25,
                                               sw_device=device, device='cpu')
            else:
                out = sliding_window_inference(inputs, train_config['patch_size'][args.dataset_name], train_config['batch_size'],
                                               predictor=net, overlap=0.25)
            end = time.time()
            out = out.argmax(dim=1, keepdim=True)
            if args.specific_sample is None:
                fp, fn, prec, rec, f1, iou, dice = metrics_tensor(label, out)
            
            if args.use_hd95 == 1:
                hd95 = get_hausdorff(label, out, label.meta['pixdim'][0, 1:4].tolist())
            else:
                hd95 = -1
            time_cost = end - start
            

            # save prediction map
            if args.specific_sample is not None:
                mask_export = nib.Nifti1Image(out[0][0].cpu().numpy().astype(np.uint8), data['label'].meta['affine'][0])
                nib.save(mask_export, os.path.join(pred_path, f"{args.model_name}_predition.nii.gz"))

            else:
                logger.info(f"{name[0]} {i}/{length} [time: {time_cost:.4f}, fp:{fp:4f}, fn:{fn:4f}, recall:{rec:4f}, precision:{prec:4f}, f1:{f1:4f}, iou:{iou:4f}, dice:{dice:4f}, hd95:{hd95:4f}, pix:{out.sum()}/{label.sum()}]")
                result += [[time_cost, fp, fn, rec, prec, f1, iou, dice, hd95, float(out.sum()), float(label.sum())]]
            
                del out, inputs, ct, pet, label, time_cost, fp, fn, rec, prec, f1, iou, dice
            # break
    if args.specific_sample is None:
        result = pd.DataFrame(result)
        result.columns = ["Time", "FP", "FN", "Recall", "Precision", "F1", "IoU", "Dice", "HD95", "Prediction", "Label"]
        result.to_csv(file_path, index=None)
    
    
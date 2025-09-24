import os
import warnings
warnings.filterwarnings("ignore")

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from .seed import seed_everything
seed_everything(seed = 12345)

from .optimizers.optimizers import build_optimizer
from .optimizers.schedulers import build_scheduler

import torch
from glob import glob
from utils.metric.metrics_brats import show_deep_metrics
from monai.data import list_data_collate, DataLoader
from datetime import datetime
import monai
from monai.transforms import (
    Compose,
    LoadImaged,
    RandCropByPosNegLabeld,
    ToTensord,
    EnsureChannelFirstd,
    RandRotated,
)
import time
from torch.utils.tensorboard import SummaryWriter
from .load_model import load_model, save_checkpoint, load_checkpoint
from utils.get_logger import get_logger
from .loss import Loss

def run_train(args, train_config, model_config):

    torch.set_num_threads(args.num_workers)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    date = datetime.now().strftime("%m_%d")
    log_path = os.path.join(train_config['log_path'], "Train", args.model_name)
    os.makedirs(log_path, exist_ok=True)
    
    model = load_model(args.model_name, model_config)
    
    modal_index = [0, 0, 0, 0]
    if args.select_modal is not None:
        modal_index[int(args.select_modal)] = 1
    else:
        modal_index = [1 for _ in range(4)]
    num_modal = len(model_config[args.model_name]['in_ch'])

    optimizer = build_optimizer(
        model=model,
        optimizer_type=train_config["optimizer"]["optimizer_type"],
        optimizer_args=train_config["optimizer"]["optimizer_args"],
    )
    warmup_scheduler = build_scheduler(
        optimizer=optimizer, scheduler_type="warmup_scheduler", config=train_config
    )
    training_scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_type="training_scheduler",
        config=train_config,
    )
    warmup_epoch = train_config['warmup_scheduler']['warmup_epochs']
    
    

    start_epoch, best_train_dice = 0, 0
    scheduler = None
    if args.checkpoint_path is not None:
        model, optimizer, scheduler, start_epoch, best_train_dice = load_checkpoint(model, args.checkpoint_path, 
                                                                                    optimizer, training_scheduler,
                                                                                    warmup_scheduler, device, 
                                                                                    warmup_epoch)
        date = "_".join(args.checkpoint_path.split("/")[-2].split("_")[:2])
    else:
        scheduler = warmup_scheduler
    model.to(device)
    
    if args.model_index is not None:
        logger_name = f"{date}_{args.dataset_name}_{args.model_index}.log"
    else:
        logger_name =  f"{date}_{args.dataset_name}.log"
    logger = get_logger(save_dir = log_path,
                        distributed_rank = 0,
                        filename = logger_name,
                        stdout=False)
    
    if args.checkpoint_path is not None:
        logger.info("Load Checkpoint, Continue to Train!!!!")
    logger.info(f"Using GPU: {args.gpu_id}")

    # set file path
    images_flair = sorted(glob(train_config['dataset_path'][args.dataset_name]["flair_path"]))
    images_t1 = sorted(glob(train_config['dataset_path'][args.dataset_name]["t1_path"]))
    images_t1ce = sorted(glob(train_config['dataset_path'][args.dataset_name]["t1ce_path"]))
    images_t2 = sorted(glob(train_config['dataset_path'][args.dataset_name]["t2_path"]))
    labels = sorted(glob(train_config['dataset_path'][args.dataset_name]["label_path"]))

    # set save path
    index = f"_{args.model_index}" if args.model_index is not None else ""
    save_path = os.path.join(train_config['save_path'], args.dataset_name, args.model_name, date + index)
    os.makedirs(save_path, exist_ok=True)
    logger.info(f"Checkpoint Save path: {save_path}")
    logger.info(f"Now Model Config: \n{model_config[args.model_name]}\n")
    
    # loss for seg_rc_style_loss
    seg_rc_style_loss = Loss(args=args, config=train_config, device=device, num_modal=num_modal)

    # data transform
    train_transforms = Compose(
            [
                LoadImaged(keys=["flair", "t1", 't1ce', 't2', "seg"]),
                EnsureChannelFirstd(keys=["flair", "t1", 't1ce', 't2', "seg"]),
                RandCropByPosNegLabeld(
                        keys=["flair", "t1", 't1ce', 't2', "seg"],
                        label_key="seg",
                        spatial_size=train_config['patch_size'][args.dataset_name],
                        pos=1,
                        neg=1,
                        num_samples=2,
                    ),
                RandRotated(keys=["flair", "t1", 't1ce', 't2', "seg"], 
                        range_z=15, 
                        prob=0.5,
                    ),
                
                ToTensord(keys=["flair", "t1", 't1ce', 't2', ]),
            ]
        )

    val_transforms = Compose(
            [
                LoadImaged(keys=["flair", "t1", 't1ce', 't2', "seg"]),
                EnsureChannelFirstd(keys=["flair", "t1", 't1ce', 't2', "seg"]),
                RandCropByPosNegLabeld(
                        keys=["flair", "t1", 't1ce', 't2', "seg"],
                        label_key="seg",
                        spatial_size=train_config['patch_size'][args.dataset_name],
                        pos=1,
                        neg=1,
                        num_samples=2,
                    ),
                ToTensord(keys=["flair", "t1", 't1ce', 't2', ]),
            ]
        )

    length = len(labels)
    logger.info(f"The number of samples: {length}")

    # split dataset
    split = train_config['train_rate']
    split_1 = train_config['train_rate'] + train_config['val_rate']
    files = [{"flair": flair, "t1": t1,"t1ce": t1ce, "t2":t2, "seg":seg} \
                    for flair, t1, t1ce, t2, seg in zip(images_flair, images_t1, images_t1ce, 
                                                        images_t2, labels)]

    train_files = files[: int(split * length)]
    val_files = files[int(split * length):int(split_1*length)]
    logger.info(f"Training set includes: {len(train_files)}")
    logger.info(f"Validation set includes: {len(val_files)}")

    # create dataloader
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(
        train_ds,
        batch_size=train_config['batch_size'],
        num_workers=args.num_workers,
        collate_fn=list_data_collate,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )

    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(
        val_ds, 
        batch_size=train_config['batch_size'], 
        num_workers=args.num_workers,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
        )

    writer = SummaryWriter(os.path.join(save_path, 'logs'))

    # training start
    iteration = 0
    best_val_dice = 0
    step = 0
    
    for epoch in range(start_epoch, train_config['epochs']):
        if epoch == warmup_epoch:
            scheduler = training_scheduler
        start = time.time()
        model.train()
        total_loss, total_avg_dice, total_et_dice, total_tc_dice, total_wt_dice = 0, 0, 0, 0, 0
        logger.info(f"\n**************************************|| Start Epoch {epoch+1} Training ||**************************************\n")
        for step, batch_data in enumerate(train_loader):
            iteration += 1
            flair, t1, t1ce, t2, labels = batch_data["flair"].type(torch.FloatTensor).to(device),\
                                    batch_data["t1"].type(torch.FloatTensor).to(device),\
                                    batch_data["t1ce"].type(torch.FloatTensor).to(device),\
                                    batch_data["t2"].type(torch.FloatTensor).to(device),\
                                    batch_data["seg"].type(torch.LongTensor).to(device)
            optimizer.zero_grad()
            inputs = [flair, t1, t1ce, t2]
            inputs = torch.cat([inputs[i] for i in range(len(inputs)) if modal_index[i]], dim=1)
            output = model(inputs)

            loss = seg_rc_style_loss(output, labels, sr_labels=inputs)
            loss.backward()
            optimizer.step()

            l = loss.item()
            
            string = f"train {epoch+1}/{train_config['epochs']} {step}/{len(train_loader)} Training Loss:{l:.4f}\n"
            
            # only for print metrics
            not_pred = 0
            if args.model_name == "VeloxSeg":
                not_pred += 2 + num_modal
                    
            if not_pred > 0:
                metrics, deep_metric_string = show_deep_metrics(output[:-not_pred], labels, train_config['show_deep_metric'])
            else:
                metrics, deep_metric_string = show_deep_metrics(output, labels, train_config['show_deep_metric'])
            logger.info(string + deep_metric_string)
            
            total_loss += l
            total_avg_dice += metrics[0]
            total_et_dice += metrics[1]
            total_tc_dice += metrics[2]
            total_wt_dice += metrics[3]

        scheduler.step()
        mean_loss = total_loss / len(train_loader)
        mean_avg_dice = total_avg_dice / len(train_loader)
        mean_et_dice = total_et_dice / len(train_loader)
        mean_tc_dice = total_tc_dice / len(train_loader)
        mean_wt_dice = total_wt_dice / len(train_loader)

        if epoch % train_config['save_model_interval'] == 0:
                save_checkpoint(model, optimizer, scheduler, epoch, best_train_dice, os.path.join(save_path, f"{epoch}.pth"))

        if mean_avg_dice >= best_train_dice:
            logger.info(f"get new best dice {best_train_dice} -> {mean_avg_dice}, save new 'train_best.pth'")
            best_train_dice = mean_avg_dice 
            save_checkpoint(model, optimizer, scheduler, epoch, best_train_dice, os.path.join(save_path, f"train_best.pth"))

        logger.info(f"training epoch {epoch + 1}: best training_epoch_dice == {best_train_dice}")
        logger.info(f"training epoch {epoch + 1}: average [Avg:{mean_avg_dice:.4f}, ET:{mean_et_dice:.4f}, TC:{mean_tc_dice:.4f}, WT:{mean_wt_dice:.4f}]")
        logger.info(f"training epoch {epoch + 1}: time cost {time.time() - start} s")

        logger.info(f"\n**************************************|| End Epoch {epoch+1} Training ||**************************************\n")
        # validation
        if (epoch + 1) % train_config['val_interval'] == 0:
            logger.info(f"\n**************************************|| Start Epoch {epoch+1} Validating ||**************************************\n")
            model.eval()
            total_avg_dice, total_et_dice, total_tc_dice, total_wt_dice = 0, 0, 0, 0
            with torch.no_grad():
                for step, batch_data in enumerate(val_loader):
                    flair, t1, t1ce, t2, labels = batch_data["flair"].type(torch.FloatTensor).to(device),\
                                    batch_data["t1"].type(torch.FloatTensor).to(device),\
                                    batch_data["t1ce"].type(torch.FloatTensor).to(device),\
                                    batch_data["t2"].type(torch.FloatTensor).to(device),\
                                    batch_data["seg"].type(torch.LongTensor).to(device)
                    inputs = [flair, t1, t1ce, t2]
                    inputs = torch.cat([inputs[i] for i in range(len(inputs)) if modal_index[i]], dim=1)
                    output = model(inputs)
                    string = f"validating {epoch+1}/{train_config['epochs']} {step}/{len(val_loader)}\n"
                    metrics, deep_metric_string = show_deep_metrics(output, labels, train_config['show_deep_metric'])
                    total_avg_dice += metrics[0]
                    total_et_dice += metrics[1]
                    total_tc_dice += metrics[2]
                    total_wt_dice += metrics[3]
                    logger.info(string + deep_metric_string)

            mean_avg_dice = total_avg_dice / len(val_loader)
            mean_et_dice = total_et_dice / len(val_loader)
            mean_tc_dice = total_tc_dice / len(val_loader)
            mean_wt_dice = total_wt_dice / len(val_loader)

            if mean_avg_dice >= best_val_dice:
                logger.info(f"get new best dice {best_val_dice} -> {mean_avg_dice}, save new 'val_best.pth'")
                best_val_dice = mean_avg_dice
                save_checkpoint(model, optimizer, scheduler, epoch, best_val_dice, os.path.join(save_path, f"val_best.pth"))

            logger.info(f"validating epoch {epoch + 1}: best validating_epoch_dice == {best_val_dice}")
            logger.info(f"validating epoch {epoch + 1}: average [Avg:{mean_avg_dice:.4f}, ET:{mean_et_dice:.4f}, TC:{mean_tc_dice:.4f}, WT:{mean_wt_dice:.4f}]")
            logger.info(f"validating epoch {epoch + 1}: time cost {time.time() - start} s")


            logger.info(f"\n**************************************|| End Epoch {epoch+1} Validating ||**************************************\n")
    writer.close()

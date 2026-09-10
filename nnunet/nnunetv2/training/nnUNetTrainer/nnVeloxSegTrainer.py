"""nnU-Net lifecycle integration for the public VeloxSeg model."""
import random

import numpy as np
import torch
from torch import nn

from model.VeloxSeg import VeloxSeg
from model.loss import VeloxSegLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.experiment_planning.experiment_planners.veloxseg_rules import segmentation_loss
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn


class nnVeloxSegTrainer(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self._train_step_index = 0
        settings = self.configuration_manager.configuration['training']
        random.seed(settings['seed'])
        np.random.seed(settings['seed'])
        torch.manual_seed(settings['seed'])
        torch.cuda.manual_seed_all(settings['seed'])
        for key in ('num_epochs', 'num_iterations_per_epoch', 'num_val_iterations_per_epoch',
                    'oversample_foreground_percent'):
            setattr(self, key, settings[key])

    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs,
                                   arch_init_kwargs_req_import, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        expected = f'{VeloxSeg.__module__}.{VeloxSeg.__name__}'
        if architecture_class_name != expected:
            raise ValueError(f'nnVeloxSegTrainer requires {expected}')
        kwargs = dict(arch_init_kwargs)
        if sum(kwargs['in_ch']) != num_input_channels:
            raise ValueError('Modality groups must match the dataset channel count')
        kwargs['n_classes'] = num_output_channels
        kwargs['deep_supervision'] = enable_deep_supervision
        return VeloxSeg(**kwargs)

    def _get_deep_supervision_scales(self):
        # The shared loss resizes this one target to each native output grid,
        # including region/ignore channels. Standalone training uses the same path.
        return None

    def _build_loss(self):
        loss = segmentation_loss(self.label_manager, self.configuration_manager.batch_dice, self.is_ddp)
        return VeloxSegLoss(loss, self.configuration_manager.network_arch_init_kwargs['in_ch'])

    def configure_optimizers(self):
        settings = self.configuration_manager.configuration['training']
        self.initial_lr = settings['initial_lr']
        self.weight_decay = settings['weight_decay']
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.initial_lr,
                                      weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.num_epochs, eta_min=settings['minimum_lr'])
        return optimizer, scheduler

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self._train_step_index = 0

    def train_step(self, batch):
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target'].to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == 'cuda'):
            output = self.network(data)
            loss = self.loss(output, target, data)
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        result = {'loss': loss.detach().cpu().numpy()}
        self._train_step_index += 1
        if self._train_step_index == 1 or self._train_step_index % 25 == 0:
            self.print_to_log_file(
                f'Train step {self._train_step_index}/{self.num_iterations_per_epoch}, '
                f'loss {float(result["loss"]):.4f}')
        return result

    def validation_step(self, batch):
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target'].to(self.device, non_blocking=True)
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == 'cuda'):
            output = self.network(data)
        loss = self.loss.segmentation_loss(output.float(), target)
        axes = [0, *range(2, output.ndim)]
        if self.label_manager.has_regions:
            predicted = (torch.sigmoid(output) > .5).long()
        else:
            # Full-patch voxel counts exceed FP16's finite range. Match the
            # upstream trainer's FP32 one-hot tensor for online Dice reduction.
            predicted = torch.zeros_like(output, dtype=torch.float32)
            predicted.scatter_(1, output.argmax(1, keepdim=True), 1)
        mask = None
        if self.label_manager.has_ignore_label:
            if self.label_manager.has_regions:
                mask = 1 - target[:, -1:].float()
                target = target[:, :-1]
            else:
                mask = (target != self.label_manager.ignore_label).float()
                target = target.masked_fill(target == self.label_manager.ignore_label, 0)
        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted, target, axes=axes, mask=mask)
        if not self.label_manager.has_regions:
            tp, fp, fn = tp[1:], fp[1:], fn[1:]
        return {'loss': loss.detach().cpu().numpy(), 'tp_hard': tp.detach().cpu().numpy(),
                'fp_hard': fp.detach().cpu().numpy(), 'fn_hard': fn.detach().cpu().numpy()}

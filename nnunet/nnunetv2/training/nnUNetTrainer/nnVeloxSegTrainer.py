"""nnU-Net lifecycle integration for the public VeloxSeg model."""
import random
from math import ceil

import numpy as np
import torch
from torch import nn

from model.VeloxSeg import VeloxSeg
from model.components.attention_utils import LayerNorm, PositionalEmbedding
from model.loss import VeloxSegLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.experiment_planning.experiment_planners.veloxseg_rules import segmentation_loss
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn

# Normalization modules whose affine parameters AdamW leaves undecayed when
# decay_exclusions is set: VeloxSeg's channels-first LayerNorm plus every torch
# norm reachable through model.components.common_function.get_norm.
NORM_MODULES = (LayerNorm, nn.LayerNorm, nn.GroupNorm, nn.modules.batchnorm._NormBase)


class nnVeloxSegTrainer(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        settings = self.configuration_manager.configuration['training']
        random.seed(settings['seed'])
        np.random.seed(settings['seed'])
        torch.manual_seed(settings['seed'])
        torch.cuda.manual_seed_all(settings['seed'])
        for key in ('num_epochs', 'num_iterations_per_epoch', 'num_val_iterations_per_epoch',
                    'oversample_foreground_percent'):
            setattr(self, key, settings[key])

    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager,
                                   num_input_channels, num_output_channels,
                                   enable_deep_supervision=True):
        expected = f'{VeloxSeg.__module__}.{VeloxSeg.__name__}'
        if configuration_manager.network_arch_class_name != expected:
            raise ValueError(f'nnVeloxSegTrainer requires {expected}')
        kwargs = dict(configuration_manager.network_arch_init_kwargs)
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
        minimum_lr = settings['minimum_lr']
        warmup_epochs = ceil(settings['warmup_updates'] / self.num_iterations_per_epoch)
        if not 0 <= warmup_epochs < self.num_epochs:
            raise ValueError('warmup_updates must resolve to warmup epochs in [0, num_epochs)')
        self.warmup_epochs = warmup_epochs
        if settings['decay_exclusions']:
            excluded, decayed = [], []
            for module in self.network.modules():
                for name, parameter in module.named_parameters(recurse=False):
                    if (isinstance(module, NORM_MODULES) or name == 'bias'
                            or (isinstance(module, PositionalEmbedding)
                                and name == 'relative_position_bias_table')):
                        excluded.append(parameter)
                    else:
                        decayed.append(parameter)
            groups = [{'params': decayed, 'weight_decay': self.weight_decay},
                      {'params': excluded, 'weight_decay': 0.0}]
        else:
            groups = [{'params': list(self.network.parameters()), 'weight_decay': self.weight_decay}]
        optimizer = torch.optim.AdamW(groups, lr=self.initial_lr)
        minimum_lr_factor = minimum_lr / self.initial_lr

        def lr_lambda(epoch):
            if warmup_epochs and epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            if warmup_epochs:
                progress = (epoch - warmup_epochs + 1) / (self.num_epochs - warmup_epochs)
            elif self.num_epochs == 1:
                progress = 1
            else:
                progress = epoch / (self.num_epochs - 1)
            return minimum_lr_factor + (1 - minimum_lr_factor) * (1 + np.cos(np.pi * progress)) / 2

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return optimizer, scheduler

    def on_train_start(self):
        super().on_train_start()
        batch = self.configuration_manager.batch_size
        oversample = self.configuration_manager.configuration['training']['oversample_foreground_percent']
        forced = batch - round(batch * (1 - oversample))
        self.print_to_log_file(f'Forced foreground samples: {forced}/{batch} = {forced / batch:.4f} '
                               f'(oversample_foreground_percent={oversample})')
        self.print_to_log_file(f'Warmup epochs: {self.warmup_epochs}')
        for index, group in enumerate(self.optimizer.param_groups):
            self.print_to_log_file(
                f"Param group {index}: weight_decay={group['weight_decay']}, tensors={len(group['params'])}, "
                f"elements={sum(parameter.numel() for parameter in group['params'])}")

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self.applied_optimizer_steps = 0
        self.skipped_optimizer_steps = 0

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
            scale = self.grad_scaler.get_scale()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            # GradScaler lowers the scale exactly when it skipped the step for inf/NaN gradients.
            skipped = self.grad_scaler.get_scale() < scale
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
            skipped = False
        self.skipped_optimizer_steps += skipped
        self.applied_optimizer_steps += not skipped
        return {'loss': loss.detach().cpu().numpy()}

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        self.print_to_log_file(f'Optimizer steps applied: {self.applied_optimizer_steps}, '
                               f'skipped by GradScaler: {self.skipped_optimizer_steps}')

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

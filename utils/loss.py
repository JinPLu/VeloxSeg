import torch
from torch import nn
import monai
from model.loss import VeloxSegLoss
from .runtime import (
    a2fseg_deep_output_groups,
    normalized_deep_loss_weights,
)
    
class Loss(nn.Module):
    def __init__(self, args, config, device, model_config):
        super(Loss, self).__init__()
        self.model_name = args.model_name
        self.device = device
        
        self.seg_loss_ce = nn.CrossEntropyLoss()
        self.seg_loss_dice = monai.losses.DiceLoss(include_background=False, 
                                                    to_onehot_y=True, 
                                                    softmax=True)
        
        deep_loss_weight = torch.tensor(config['deep_Loss_weight'], dtype=torch.float32)
        
        self.register_buffer("deep_loss_weight", deep_loss_weight)
        if self.model_name == 'VeloxSeg':
            self.veloxseg_loss = VeloxSegLoss(
                self.seg_loss, model_config['in_ch'],
                config['RC_Loss_weight'], config['Feature_Loss_weight'])

    def seg_loss(self, output, labels):
        return self.seg_loss_ce(output, labels.squeeze(1)) + self.seg_loss_dice(input=output, target = labels)

    def _deep_weights(self, output_count, device):
        weights = normalized_deep_loss_weights(
            self.deep_loss_weight.detach().cpu().tolist(),
            output_count,
        )
        return torch.as_tensor(weights, dtype=torch.float32, device=device)

    def deep_seg_loss(self, outputs, labels):
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

        weights = self._deep_weights(len(outputs), outputs[0].device)
        loss = outputs[0].new_tensor(0.0)
        for weight, output in zip(weights, outputs):
            loss = loss + weight * self.seg_loss(output, labels)
        return loss
    
    def cal_loss(self, output, labels, sr_labels=None):
        
        if self.model_name in ["VeloxSeg"]:
            return self.veloxseg_loss(output, labels, sr_labels)

        elif self.model_name == 'A2FSeg':
            loss = output[0].new_tensor(0.0)
            for start, end in a2fseg_deep_output_groups(len(output)):
                loss = loss + self.deep_seg_loss(output[start:end], labels)

            primary_weight = self._deep_weights(
                len(self.deep_loss_weight),
                output[0].device,
            )[0]
            return loss + primary_weight * self.seg_loss_ce(output[0], labels.squeeze(1))
        
        elif self.model_name in ['VSmTrans', 'UNETRpp', 'HDense']:
            return self.deep_seg_loss(output, labels)
            
        else:
            return self.seg_loss(output, labels)
    
    def forward(self, output, labels, sr_labels=None):
        return self.cal_loss(output, labels, sr_labels)

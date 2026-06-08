import torch
from torch import nn
import monai
from .runtime import (
    a2fseg_deep_output_groups,
    normalized_deep_loss_weights,
    veloxseg_output_layout,
)
    
class Loss(nn.Module):
    def __init__(self, args, config, device, num_modal=2):
        super(Loss, self).__init__()
        self.model_name = args.model_name
        self.device = device
        self.num_modal = num_modal
        
        self.seg_loss_ce = nn.CrossEntropyLoss()
        self.seg_loss_dice = monai.losses.DiceLoss(include_background=False, 
                                                    to_onehot_y=True, 
                                                    softmax=True)
        self.rc_loss = nn.MSELoss()
        self.gram_loss = nn.MSELoss()
        
        deep_loss_weight = torch.tensor(config['deep_Loss_weight'], dtype=torch.float32)
        
        self.rc_loss_weight = config.get('RC_Loss_weight')
        self.feature_loss_weight = config.get('Feature_Loss_weight')
        self.register_buffer("deep_loss_weight", deep_loss_weight)

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
            layout = veloxseg_output_layout(len(output), self.num_modal)
            seg_start, seg_end = layout["seg"]
            seg_loss = self.deep_seg_loss(output[seg_start:seg_end], labels)
            rc_loss =  self.rc_loss(output[layout["reconstruction"]], sr_labels)
            
            feature_loss = 0
            for teacher_index in layout["teacher_grams"]:
                feature_loss = feature_loss + self.gram_loss(
                    output[layout["decoder_gram"]],
                    output[teacher_index],
                )
            feature_loss = feature_loss / self.num_modal

            return seg_loss + self.rc_loss_weight * rc_loss + self.feature_loss_weight * feature_loss

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

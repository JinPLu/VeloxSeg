import torch
from torch import nn
import monai
    
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
        
        deep_loss_weight = torch.tensor(config['deep_Loss_weight'])
        
        self.rc_loss_weight = config.get('RC_Loss_weight')
        self.feature_loss_weight = config.get('Feature_Loss_weight')
        self.deep_loss_weight = deep_loss_weight / deep_loss_weight.sum()

    def seg_loss(self, output, labels):
        return self.seg_loss_ce(output, labels.squeeze(1)) + self.seg_loss_dice(input=output, target = labels)

    def deep_seg_loss(self, outputs, labels):
        assert len(outputs) == len(self.deep_loss_weight), f'len(outputs) ({len(outputs)})!= len(self.deep_loss_weight) ({len(self.deep_loss_weight)})'
        loss = 0
        for weight, output in zip(self.deep_loss_weight, outputs):
            loss = loss + weight * self.seg_loss(output, labels)
        return loss
    
    def cal_loss(self, output, labels, sr_labels=None):
        
        if self.model_name in ["VeloxSeg"]:
            seg_loss = self.deep_seg_loss(output[:4], labels)
            rc_loss =  self.rc_loss(output[4], sr_labels)
            
            feature_loss = 0
            for i in range(self.num_modal):
                feature_loss = feature_loss + self.gram_loss(output[5], output[6+i])
            feature_loss = feature_loss / self.num_modal

            return seg_loss + self.rc_loss_weight * rc_loss + self.feature_loss_weight * feature_loss

        elif self.model_name == 'A2FSeg':
            return self.deep_seg_loss(output[1:6], labels) + self.deep_seg_loss(output[6:11], labels) + \
                    self.deep_seg_loss(output[11:16], labels) + \
                    self.deep_loss_weight[0] * self.seg_loss_ce(output[0], labels.squeeze(1))
        
        elif self.model_name in ['VSmTrans', 'UNETRpp', 'HDense']:
            return self.deep_seg_loss(output, labels)
            
        else:
            return self.seg_loss(output, labels)
    
    def forward(self, output, labels, sr_labels=None):
        return self.cal_loss(output, labels, sr_labels)

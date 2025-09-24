import torch
import numpy as np
import copy
from medpy.metric.binary import hd95

def show_deep_metrics(outputs, labels, deep=True):
    if not isinstance(outputs, list):
        outputs = [outputs]
    output = outputs[0].argmax(dim=1, keepdim=True)
    avg_dice, et_dice, tc_dice, wt_dice = cal_dice(labels, output)
    string = f"[Avg:{avg_dice:.4f}, ET:{et_dice:.4f}, TC:{tc_dice:.4f}, WT:{wt_dice:.4f} pix:{(output != 0).sum():6}/{(labels != 0).sum():6}]\n"
    res = [avg_dice, et_dice, tc_dice, wt_dice]
    if deep and len(outputs) > 1:
        for output in outputs[1:]:
            output = output.argmax(dim=1, keepdim=True)
            avg_dice, et_dice, tc_dice, wt_dice = cal_dice(labels, output)
            string += f"[Avg:{avg_dice:.4f}, ET:{et_dice:.4f}, TC:{tc_dice:.4f}, WT:{wt_dice:.4f} pix:{(output != 0).sum():6}/{(labels != 0).sum():6}]\n"
    string += '\n'
    return res, string

def Dice(output, target, eps=1e-6):
    inter = torch.sum(output * target, dim=(1,2,3)) + eps
    union = torch.sum(output,dim=(1,2,3)) + torch.sum(target,dim=(1,2,3)) + eps * 2
    x = 2 * inter / union
    dice = torch.mean(x)
    return dice

def HD95(gt, pred):
    if (gt.max() == 0) or (pred.max() == 0):
        return np.NaN
    else:
        hausdorff_distance95 = hd95(pred.detach().cpu().numpy(), gt.detach().cpu().numpy())
        return float(hausdorff_distance95)

def cal_dice(output, target):
    et_dice = Dice((output == 3).float(), (target == 3).float())
    tc_dice = Dice(((output == 1) | (output == 3)).float(), ((target == 1) | (target == 3)).float())
    wt_dice = Dice((output != 0).float(), (target != 0).float())

    return float((et_dice+tc_dice+wt_dice)/3), float(et_dice), float(tc_dice), float(wt_dice)

def cal_hd95(output, target):
    et_hd95 = HD95((output == 3).float(), (target == 3).float())
    tc_hd95 = HD95(((output == 1) | (output == 3)).float(), ((target == 1) | (target == 3)).float())
    wt_hd95 = HD95((output != 0).float(), (target != 0).float())

    return float((et_hd95+tc_hd95+wt_hd95)/3), float(et_hd95), float(tc_hd95), float(wt_hd95)

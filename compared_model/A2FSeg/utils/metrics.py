import torchio as tio
from pathlib import Path
import torch
import numpy as np
import copy
from medpy.metric.binary import hd,hd95

def show_deep_metrics(outputs, labels, deep=True):
    s = ''
    end = 0 if not deep else len(outputs) - 1
    res = []
    for i, output in enumerate(outputs):
        if (not deep) & (i > 0):
            break
        if i == end:
            s = '\n'
        output = output.argmax(dim=1, keepdim=True)
        fp, fn, iou, dice = metrics_tensor(labels, output)
        print("[FP:{:.4f}, FN:{:.4f}, IoU:{:.4f}, Dice:{:.4f} pix:{:6}/{:6}]{}".format(fp, fn, iou, dice, 
                                                                                       output.sum(), labels.sum(), s))
        if i == 0:
            res += [fp, fn, iou, dice]
    return res

def get_hausdorff(gt, pred):
    if (gt.max() == 0) or (pred.max() == 0):
        return np.NaN
    else:
        hausdorff_distance95 = hd95(pred.detach().cpu().numpy(), gt.detach().cpu().numpy())
        return float(hausdorff_distance95)


def metrics(gt, pred):
    gt = gt.detach().numpy()
    pred = pred.detach().numpy()
    
    pred = pred.astype(int)
    gt = gt.astype(int)
    
    fp_array = copy.deepcopy(pred)
    fn_array = copy.deepcopy(gt)
    gt_sum = np.sum(gt)
    pred_sum = np.sum(pred)
    
    intersection = gt & pred
    union = gt | pred
    intersection_sum = np.count_nonzero(intersection)
    union_sum = np.count_nonzero(union)
    
    tp_array = intersection
    
    diff = pred - gt
    fp_array[diff < 1] = 0
    
    diff = gt - pred
    fn_array[diff < 1] = 0
    
    tn_array = np.ones(gt.shape) - union
    
    tp, fp, fn, tn = np.sum(tp_array), np.sum(fp_array), np.sum(fn_array), np.sum(tn_array)
    
    smooth = 0.001
    precision = tp / (pred_sum + smooth)
    recall = tp / (gt_sum + smooth)
    
    false_positive_rate = fp / (fp + tn + smooth)
    false_negtive_rate = fn / (fn + tp + smooth)

    jaccard = intersection_sum / (union_sum + smooth)
    dice = 2 * intersection_sum / (gt_sum + pred_sum + smooth)
    
    return false_positive_rate, false_negtive_rate, jaccard, dice


def metrics_tensor(gt, pred):
    # b = pred.shape[0]
    # res = [metrics_tensor_one_patience(gt[i:i+1], pred[i:i+1]) for i in range(b)]
    # return np.mean(res, axis=0)
    return metrics_tensor_one_patience(gt, pred)

def metrics_tensor_one_patience(gt, pred):
    assert (len(gt.shape) == len(pred.shape)) 
    if pred.shape[1] == 2:
        pred = pred[:, 1:]

    if gt.shape[1] == 2:
        gt = gt[:, 1:]
    
    pred = pred.type(torch.IntTensor)
    gt = gt.type(torch.IntTensor)
    fp_array = copy.deepcopy(pred)
    fn_array = copy.deepcopy(gt)
    gt_sum = torch.sum(gt)
    pred_sum = torch.sum(pred)
    
    intersection = gt & pred
    union = gt | pred
    intersection_sum = intersection.sum()
    union_sum = union.sum()
    
    tp_array = intersection
    
    diff = pred - gt
    fp_array[diff < 1] = 0
    
    diff = gt - pred
    fn_array[diff < 1] = 0
    
    tn_array = torch.ones_like(gt) - union
    
    tp, fp, fn, tn = torch.sum(tp_array), torch.sum(fp_array), torch.sum(fn_array), torch.sum(tn_array)
    
    smooth = 0.001
    precision = tp / (pred_sum + smooth)
    recall = tp / (gt_sum + smooth)
    
    false_positive_rate = fp / (fp + tn + smooth)
    false_negtive_rate = fn / (fn + tp + smooth)

    jaccard = intersection_sum / (union_sum + smooth)
    dice = 2 * intersection_sum / (gt_sum + pred_sum + smooth)
    
    return float(false_positive_rate), float(false_negtive_rate), float(jaccard), float(dice)
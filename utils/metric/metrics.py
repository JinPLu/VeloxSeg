import torch
import numpy as np
import copy
from medpy.metric.binary import hd95

def show_deep_metrics(outputs, labels, deep=True):
    """
    Show the metrics of the predicted and ground truth.
    return:
        res: the metrics of the predicted and ground truth.
        string: the string of the metrics.
    """
    if not isinstance(outputs, list):
        outputs = [outputs]
    output = outputs[0].argmax(dim=1, keepdim=True)
    fp, fn, _, _, _, iou, dice = metrics_tensor(labels, output)
    string = f"[FP:{fp:.4f}, FN:{fn:.4f}, IoU:{iou:.4f}, Dice:{dice:.4f} pix:{output.sum():6}/{labels.sum():6}]\n"
    res = [fp, fn, iou, dice]
    if deep and len(outputs) > 1:
        for output in outputs[1:]:
            output = output.argmax(dim=1, keepdim=True)
            fp, fn, _, _, _, iou, dice = metrics_tensor(labels, output)
            string += f"[FP:{fp:.4f}, FN:{fn:.4f}, IoU:{iou:.4f}, Dice:{dice:.4f} pix:{output.sum():6}/{labels.sum():6}]\n"
    string += '\n'
    return res, string

def get_hausdorff(gt, pred, spacing=(1,1,1)):
    """
    Calculate the hausdorff distance of the predicted and ground truth.
    return:
        hausdorff_distance95: the hausdorff distance of the predicted and ground truth.
    """
    if (gt.max() == 0) or (pred.max() == 0):
        return np.nan
    else:
        pred = pred.squeeze(0).squeeze(0).detach().cpu().numpy()
        gt = gt.squeeze(0).squeeze(0).detach().cpu().numpy()
        hausdorff_distance95 = hd95(pred, gt, voxelspacing=spacing)
        return float(hausdorff_distance95)


def metrics_tensor(gt, pred):
    """
    Calculate the metrics of the predicted and ground truth.
    return:
        fp: false positive rate
        fn: false negative rate
        precision: precision
        recall: recall
        f1_score: f1 score
        iou: intersection over union
        dice: dice coefficient
    """
    assert (len(gt.shape) == len(pred.shape)) 
    if pred.shape[1] == 2:
        pred = pred[:, 1:]

    if gt.shape[1] == 2:
        gt = gt[:, 1:]
    
    pred = pred.type(torch.IntTensor)
    gt = gt.type(torch.IntTensor)
    fp_array = copy.deepcopy(pred)
    fn_array = copy.deepcopy(gt)
    gt_sum = gt.sum((1, 2, 3, 4))
    pred_sum = pred.sum((1, 2, 3, 4))
    
    intersection = gt & pred
    union = gt | pred
    intersection_sum = intersection.sum((1, 2, 3, 4))
    union_sum = union.sum((1, 2, 3, 4))
    
    tp_array = intersection
    
    diff = pred - gt
    fp_array[diff < 1] = 0
    
    diff = gt - pred
    fn_array[diff < 1] = 0
    
    tn_array = torch.ones_like(gt) - union
    
    tp, fp, fn, tn = tp_array.sum((1, 2, 3, 4)), fp_array.sum((1, 2, 3, 4)), fn_array.sum((1, 2, 3, 4)), tn_array.sum((1, 2, 3, 4))
    
    smooth = 1e-5
    precision = tp / (pred_sum + smooth)
    recall = tp / (gt_sum + smooth)
    f1_score = 2 * precision * recall / (precision + recall + smooth)
    
    false_positive_rate = fp / (fp + tn + smooth)
    false_negtive_rate = fn / (fn + tp + smooth)

    jaccard = intersection_sum / (union_sum + smooth)
    dice = 2 * intersection_sum / (gt_sum + pred_sum + smooth)
    
    return [float(metc.mean()) for metc in [false_positive_rate, false_negtive_rate, precision, recall, f1_score, jaccard, dice]]
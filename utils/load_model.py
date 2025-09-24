import torch

def load_model(model_name, config):
    """
    Load the model.
    return:
        model: the model
    """

    if model_name == 'UNet':
        from monai.networks.nets import BasicUNet
        return BasicUNet(**config[model_name])
    
    elif model_name == 'VNet':
        from monai.networks.nets import VNet
        return VNet(**config[model_name])
    
    elif model_name == 'UNETR':
        from monai.networks.nets import UNETR
        return UNETR(**config[model_name])
    
    elif model_name == 'SwinUNETR':
        from monai.networks.nets import SwinUNETR
        return SwinUNETR(**config[model_name])
    
    elif model_name == 'A2FSeg':
        from compared_model.A2FSeg.nnunet.network_architecture.my.generic_MAML3_channel import Generic_MAML_multi3_channel
        return Generic_MAML_multi3_channel(**config[model_name])
    
    elif model_name == 'NestedFormer':
        from compared_model.NestedFormer.medical.model.nested_former import NestedFormer
        return NestedFormer(**config[model_name])
    
    elif model_name == "MedNeXt":
        from compared_model.MedNeXt.create_mednext_v1 import create_mednextv1_small
        return create_mednextv1_small(**config[model_name])
    
    elif model_name == "SlimUNETR":
        from compared_model.SlimUNETR.SlimUNETR import SlimUNETR
        return SlimUNETR(**config[model_name])
    
    elif model_name == "HDense":
        from compared_model.HDense.HDenseFormer import HDenseFormer_16
        return HDenseFormer_16(**config[model_name])
    
    elif model_name == "SegFormer":
        from compared_model.SegFormer.SegFormer import SegFormer3D
        return SegFormer3D(**config[model_name])
    
    elif model_name == "UNETRpp":
        from compared_model.unetr_pp.network_architecture.tumor.unetr_pp_tumor import UNETR_PP
        return UNETR_PP(**config[model_name])
    
    elif model_name == "VSmTrans":
        from compared_model.VSmTrans.VSmTrans import VSmixTUnet
        return VSmixTUnet(**config[model_name])
    
    elif model_name == "HCMA-UNet":
        from compared_model.HCMA.HCMA import HCMA
        return HCMA(**config[model_name])
    
    elif model_name == "U-KAN":
        from compared_model.UKAN.archs import UKAN
        return UKAN(**config[model_name])
        
    elif model_name == "SuperLightNet":
        from compared_model.SuperLightNet.superlightnet import NormalU_Net
        return NormalU_Net(**config[model_name])
    
    elif model_name == "U-RWKV":
        from compared_model.URWKV.URWKV import v_enc_256_fffse_dec_fusion_rwkv_with2x4_3d
        return v_enc_256_fffse_dec_fusion_rwkv_with2x4_3d(**config[model_name])
    
    elif model_name == "Conv_Attn":
        from VeloxSeg.model.Conv_Attn import SlimMSUA
        return SlimMSUA(**config[model_name])
    
    elif model_name == "Conv_Attn_Teacher":
        from VeloxSeg.model.Conv_Attn_Teacher import SlimMSUA_RC
        return SlimMSUA_RC(**config[model_name])

    # our model
    elif model_name == "VeloxSeg":
        from model.VeloxSeg import VeloxSeg
        return VeloxSeg(**config[model_name])

    else:
        raise ValueError("Invalid model name, now {}".format(model_name))
    
def save_checkpoint(model, optimizer, scheduler, epoch, best_iou, filename='checkpoint.pth'):
    state ={
            "model": model.state_dict(),
            "lr_sche": scheduler.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_iou": best_iou,
        }
    torch.save(state, filename)

def checkpoint_DDP_to_SingleGPU(checkpoint):   
    from collections import OrderedDict
    new_checkpoint = OrderedDict()
    for k, v in checkpoint.items():
        name = k
        if name.startswith("module."):
            name = name.replace('module.', '')
        new_checkpoint[name] = v
    return new_checkpoint

def load_checkpoint(model, filename, optimizer = None, scheduler = None, 
                    warmup_scheduler = None, device = torch.device("cpu"),
                    warmup_epoch = 0):
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint_DDP_to_SingleGPU(checkpoint['model']))
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        if (scheduler is not None) or (warmup_scheduler is not None):
            if checkpoint['epoch'] < warmup_epoch:
                scheduler = warmup_scheduler
            else:
                scheduler = scheduler
            scheduler.load_state_dict(checkpoint['lr_sche'])
        return model, optimizer, scheduler, checkpoint['epoch'], checkpoint['best_iou']
    return model

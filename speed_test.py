import os
import torch
import time
import json
from utils.load_model import load_model
from thop import profile
torch.autograd.set_grad_enabled(False)

# Time for warmup
T0 = 10
# Time for testing
T1 = 60
# Input size
INPUT_SIZE = {
    'AutoPETII': (2, 96, 96, 96),
    'Hecktor2022': (2, 128, 128, 64),
    'BraTS2021': (4, 96, 96, 96),
}
# Config path
CONFIG_PATH = {
    'AutoPETII': './config/models_config_autopetii.json',
    'Hecktor2022': './config/models_config_hecktor2022.json',
    'BraTS2021': './config/models_config_brats2021.json',
}

# Batch size
CPU_BS = 1
MAX_GPU_BS = 16

def find_max_batch_size(model, input_shape):
    max_bs = 0 
    now_bs = 1

    model.cuda()
    model.eval()
    while now_bs <= MAX_GPU_BS: 
        try:
            inputs = torch.randn(now_bs, *input_shape).type(torch.FloatTensor).cuda()
            model(inputs)
            max_bs = now_bs
            now_bs *= 2
            
        except torch.cuda.OutOfMemoryError:
            break
            
        finally:
            if 'inputs' in locals():
                del inputs
            torch.cuda.empty_cache()
    return max_bs


def main(args):

    with open(CONFIG_PATH[args.dataset], 'r', encoding='utf-8') as file:
        config = json.load(file)
    
    for device in ['cuda', 'cpu']:

        if 'cuda' in device and not torch.cuda.is_available():
            print("no cuda")
            continue

        if device == 'cpu':
            os.system('echo -n "nb processors "; '
                    'cat /proc/cpuinfo | grep ^processor | wc -l; '
                    'cat /proc/cpuinfo | grep ^"model name" | tail -1')
            print('Using 1 cpu thread')
            torch.set_num_threads(1)
            compute_throughput = compute_throughput_cpu
        else:
            print(torch.cuda.get_device_name(torch.cuda.current_device()))
            compute_throughput = compute_throughput_cuda

        if args.model_list is None:
            keys = config.keys()
        else:
            keys = args.model_list
        for name in keys:
            if name in ['HCMA-UNet', 'U-RWKV']:
                continue
            model = load_model(name, config)
            
            if device == 'cpu':
                if name in ['HCMA-UNet', 'U-RWKV']:
                    continue
                batch_size = CPU_BS
            else:
                batch_size = find_max_batch_size(model, INPUT_SIZE[args.dataset])
                torch.cuda.empty_cache()
                model.cuda()
            model.eval()
            compute_throughput(name, model, batch_size, input_size=INPUT_SIZE[args.dataset])
            
            if device == 'cuda':
                x = torch.rand((1, *INPUT_SIZE[args.dataset])).cuda()
                flops, params = profile(model.cuda(), inputs=(x,))
                print("Params", params / 1e6, "M")
                print("FLOPS：", flops / 1e9, "G")


def compute_throughput_cpu(name, model, batch_size, input_size):
    inputs = torch.randn(batch_size, *input_size).type(torch.FloatTensor)
    # warmup
    start = time.time()
    while time.time() - start < T0:
        model(inputs)

    timing = []
    while sum(timing) < T1:
        start = time.time()
        model(inputs)
        timing.append(time.time() - start)
    timing = torch.as_tensor(timing, dtype=torch.float32)
    print(name, 'cpu', batch_size / timing.mean().item(), 'images/s @ batch size', batch_size)

def compute_throughput_cuda(name, model, batch_size, input_size):
    inputs = torch.randn(batch_size, *input_size).type(torch.FloatTensor).cuda()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start = time.time()
    with torch.amp.autocast('cuda'):
        while time.time() - start < T0:
            model(inputs)
    timing = []
    torch.cuda.synchronize()
    with torch.amp.autocast('cuda'):
        while sum(timing) < T1:
            start = time.time()
            model(inputs)
            torch.cuda.synchronize()
            timing.append(time.time() - start)
    timing = torch.as_tensor(timing, dtype=torch.float32)
    print(name, 'gpu', batch_size / timing.mean().item(), 'images/s @ batch size', batch_size)
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model_list', type=str, required=False, default=None)
    args = parser.parse_args()
    
    assert args.dataset in ['AutoPETII', 'Hecktor2022', 'BraTS2021']
    if args.model_list is not None:
        if ',' in args.model_list:
            args.model_list = args.model_list.replace(' ', '').split(',')
        else:
            args.model_list = [args.model_list]
    main(args)
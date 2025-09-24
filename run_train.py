import json
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True, help='dataset name')
    parser.add_argument('--model_name', type=str, required=True, help='model name')
    parser.add_argument('--train_config', type=str, required=True, help='train_config path')
    parser.add_argument('--model_config', type=str, required=True, help='model_config path')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='checkpoint path')
    parser.add_argument('--gpu_id', type=str, default='0', help='gpu id for inferencing')
    # parser.add_argument("--local-rank", "--local_rank", type=int)
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for data loading')
    parser.add_argument('--model_index', type=str, default=None, help='Markdown index of the model')
    parser.add_argument('--select_modal', type=str, default=None, )
    args = parser.parse_args()
    with open(args.train_config, 'r', encoding='utf-8') as f:
        train_config = json.load(f)
    with open(args.model_config, 'r', encoding='utf-8') as f:
        model_config = json.load(f)
        
    if args.dataset_name == 'AutoPETII':
        from utils.train_autopet import run_train
    elif args.dataset_name == 'Hecktor2022':
        from utils.train_hecktor import run_train
    elif args.dataset_name == 'BraTS2021':
        from utils.train_brats2021 import run_train
    
    run_train(args, train_config, model_config)
    
    

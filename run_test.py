import argparse
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True, help='dataset name for loading data')
    parser.add_argument('--model_name', type=str, required=True, help='model name for loading model')
    parser.add_argument('--train_date', type=str, default=None, help='the date of training model')
    parser.add_argument('--model_index', type=str, default=None, help='the version index of the model')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='checkpoint directory for loading model')
    parser.add_argument('--checkpoint_index', type=str, default='val_best', help='checkpoint index for loading model')
    parser.add_argument('--model_config', type=str, default=None, help='model_config path')
    parser.add_argument('--train_config', type=str, default=None, help='train config file')
    parser.add_argument('--test_config', type=str, default=None, help='test config file')
    parser.add_argument('--gpu_id', type=str, default='0', help='gpu id for inferencing')
    # parser.add_argument("--local-rank", "--local_rank", type=int)
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for data loading')
    parser.add_argument('--specific_sample', type=int, default=None, help='choose a specific sample to predict')
    parser.add_argument('--select_modal', type=str, default=None, )
    parser.add_argument('--use_hd95', type=int, default=None, help='0 for no, 1 for yes')
    args = parser.parse_args()

    if args.dataset_name == 'AutoPETII' or args.dataset_name == 'Hecktor2022':
        from utils.inference_petct import run_Inference
    elif args.dataset_name == 'BraTS2021':
        from utils.inference_brats import run_Inference
    run_Inference(args)
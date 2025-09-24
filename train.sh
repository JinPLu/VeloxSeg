# AutoPET-II
python -u ./run_train.py --dataset_name AutoPETII --model_name VeloxSeg --train_config ./config/train_config_bs4.json --model_config ./config/models_config_autopetii.json --num_workers 4 --gpu_id 0 &

# Hecktor2022
# python -u ./run_train.py --dataset_name Hecktor2022 --model_name VeloxSeg --train_config ./config/train_config_bs4.json --model_config ./config/models_config_hecktor2022.json --num_workers 4 --gpu_id 0 &

# BraTS2021
# python -u ./run_train.py --dataset_name BraTS2021 --model_name VeloxSeg --train_config ./config/train_config_bs4.json --model_config ./config/models_config_brats2021.json --num_workers 4 --gpu_id 0 &

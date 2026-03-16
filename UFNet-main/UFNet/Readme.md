# A unified framework for noisy image super-resolution
## preparation
1. datasets: DIV2K, Urban100, BSD100, Set5 and Set14
2. environment: according to requirements.txt
## Train
options are available in train.py 
noise level can be controlled in `add_noise` function (accurate level and blind level)  
example command:  
```
python train.py --model model --ckpt_name x2_last --ckpt_dir checkpoint/x2_last --num_gpu 1 --scale 2 --patch_size 64 --batch_size 64 --max_steps 600000 --decay 400000 --train_data_path "DIV2K_train.h5"
```
## Test
options are available in test.py
noise level can be controlled in `add_noise` function (accurate level and blind level)  
example command:  
```
python sample.py --model model --test_data_dir dataset/Set5 --scale 2 --ckpt_path ./checkpoint/model_16000_x2.pth --sample_dir Set5_42000_x2
```
sample_nokey is used to load the weights file which don't contain optimizer weights.
# downscaling_Peurto_Rico
repo for downscaling using CGP+GAN


### Training GAN from CGP outputs
```bash
micromamba activate downscaling_pr_new


cd /net/flood/data2/users/x_yan/02_hourly_mpas_PR/gan-1

# Example: historical
nohup python train_gan.py --config config.yml --scenario historical --gpuid 0 \
  > nohup_hist_g0.out 2>&1 &

# Example: future
nohup python train_gan.py --config config.yml --scenario future --gpuid 1 \
  > nohup_fut_g1.out 2>&1 &

nohup python train_direct_gan.py --config config.yml --scenario historical --gpuid 2 --force_resize_lr \
  > nohup_hist_direct.out 2>&1 &
```

```bash

cd /net/flood/data2/users/x_yan/01_hourly_puerto-rico/gan-1
mkdir -p logs

# GPU 0: historical split 2
nohup python train_gan.py --config config.yml --scenario historical --split 2 --gpuid 0 \
    > logs/hist_s2_gpu0.out 2>&1 &

# GPU 1: future split 2
nohup python train_gan.py --config config.yml --scenario future --split 2 --gpuid 1 \
    > logs/fut_s2_gpu1.out 2>&1 &
```

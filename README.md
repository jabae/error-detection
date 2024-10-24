# Error Detection

## Install
```
git clone https://github.com/seung-lab/error-detection.git
cd error-detection

pip install -e .
```

## Training
- Pinky10 training data

Command example:
```
python errordetector/train_pinky10.py --exp_dir /usr/people/jabae/seungmount/research/Alex/error_detection/exp/exp_mip1_fixds_0930 --network UNetMip1 --patch_size 33 320 320 --out_size 33 20 20 --chkpt_num 0 --gpu_ids 0 1 2 3 4 5 6 7 --augment flip rotate90 --lr 0.0001
```

## Inference

Command example:
```
```

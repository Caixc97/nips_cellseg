# Team RedCat solution to Weakly Supervised Cell Segmentation in Multi-modality High-Resolution Microscopy Images

https://neurips22-cellseg.grand-challenge.org/

This repository is a fork of https://github.com/tascj/kaggle-sartorius-cell-instance-segmentation-solution

## Environment setup

Build docker image

```
bash .dev_scripts/build.sh
```

Set env variables

```
export DATA_DIR="/path/to/data"
export CODE_DIR="/path/to/this/repo"
```

Start a docker container
```
bash .dev_scripts/start.sh all
```

## Data preparation

1. Download sartorius competition data from Kaggle https://www.kaggle.com/c/sartorius-cell-instance-segmentation
2. Download LIVECell dataset from https://github.com/sartorius-research/LIVECell
3. Download nips_cellseg competition data.
4. Unzip the files as follows

```
├── LIVECell_dataset_2021
│   ├── images
│   ├── livecell_coco_train.json
│   ├── livecell_coco_val.json
│   └── livecell_coco_test.json
├── sartorius
│   ├── train
│   ├── train_semi_supervised
│   └── train.csv
└── nips
   ├── Train_Labeled
   │   ├── images
   │   └── labels
   ├── Train_Unabeled
   │   └── images
   └── TuningSet
```

Start a docker container and run the following commands

```
mkdir /data/checkpoints/
python tools/prepare_livecell.py
python tools/prepare_kaggle.py
python tools/pre_process_3class.py
python tools/prepare_nips.py
```

The results should look like the 

```
├── LIVECell_dataset_2021
│   ├── images
│   ├── train_8class.json
│   ├── val_8class.json
│   ├── test_8class.json
│   ├── livecell_coco_train.json
│   ├── livecell_coco_val.json
│   └── livecell_coco_test.json
├── sartorius
│   ├── train
│   ├── train_semi_supervised
│   ├── checkpoints
│   ├── train.csv
│   ├── dtrainval.json
│   ├── dtrain_g0.json
│   └── dval_g0.json
└── nips
   ├── Train_Labeled
   │   ├── images
   │   └── labels
   ├── Train_Pre_3class_slide'
   │   ├── images
   │   └── labels
   │   ├── dtrainval.json
   │   ├── dtrain_g0.json
   │   └── dval_g0.json
   ├── Train_Unabeled
   │   └── images
   └── TuningSet
```

## Training

Download COCO pretrained YOLOX-x weights from https://github.com/Megvii-BaseDetection/YOLOX

Convert the weights

```
python tools/convert_official_yolox.py /path/to/yolox_x.pth /path/to/data/checkpoints/yolox_x_coco.pth
```

Start a docker container and run the following commands for training
```
# train detector using the LIVECell dataset
python tools/det/train.py configs/det/yolox_x_livecell.py

# predict bboxes of LIVECell validataion data
python tools/det/test.py configs/det/yolox_x_livecell.py work_dirs/yolox_x_livecell/epoch_30.pth --out work_dirs/yolox_x_livecell/val_preds.pkl --eval bbox

# finetune the detector on sartorius data(train split)
python tools/det/train.py configs/det/yolox_x_kaggle.py --load-from work_dirs/yolox_x_livecell/epoch_15.pth

# predict bboxes of sartorius data(val split)
python tools/det/test.py configs/det/yolox_x_kaggle.py work_dirs/yolox_x_kaggle/epoch_30.pth --out work_dirs/yolox_x_kaggle/val_preds.pkl --eval bbox

# finetune the detector on competition data(train split)
python tools/det/train.py configs/det/yolox_x_nips.py --load-from work_dirs/yolox_x_nips/latest.pth

# predict bboxes of competition data(val split)
python tools/det/test.py configs/det/yolox_x_nips.py work_dirs/yolox_x_nips/latest.pth --out work_dirs/yolox_x_nips/val_preds.pkl --eval bbox

#--------------------

# train segmentor using LIVECell dataset
python tools/seg/train.py configs/seg/unet_livecell.py

# finetune the segmentor on sartorius data(train split)
python tools/seg/train.py configs/seg/unet_kaggle.py --load-from work_dirs/unet_kaggle.py/epoch_1.pth

# finetune the segmentor on competition data(train split)
python tools/seg/train.py configs/seg/unet_nips.py --load-from work_dirs/unet_kaggle.py/latest.pth

# predict instance masks of competition data(val split)
python tools/seg/test.py configs/seg/unet_nips.py work_dirs/unet_nips.py/latest.pth --out work_dirs/uunet_nips.py/val_results.pkl --eval dummy
```


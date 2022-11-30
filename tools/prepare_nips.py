import numpy as np
import random
from skimage import io
import os
from tqdm import tqdm
import mmcv
import pycocotools.mask as mask_utils



def init_coco():
    return {
        'info': {},
        'categories':
            [{
                'id': 0,
                'name': 'cell',
            }]
    }


def gen_coco(img_dir, choose_idx):
    img_infos = []
    ann_infos = []
    img_id = 0
    ann_id = 0
    label_dir = img_dir.replace('images', 'labels')
    pbar = tqdm(total=len(os.listdir(img_dir)))
    for idx, file_name in enumerate(os.listdir(img_dir)):
        pbar.update(1)
        if not idx in choose_idx:
            continue
        img_name = file_name.split('.')[0]
        tif_path = img_name + '.tif'
        label = io.imread(os.path.join(label_dir, tif_path))
        img_info = dict(
            id=img_id,
            width=label.shape[0],
            height=label.shape[1],
            file_name=img_name + '.png',
        )
        idxs = np.unique(label)
        for idx in idxs:
            mask = (label == idx)
            mask = np.asfortranarray(mask)
            rle = mask_utils.encode(mask)
            rle['counts'] = rle['counts'].decode()
            bbox = mask_utils.toBbox(rle).tolist()
            ann_info = dict(
                id=ann_id,
                image_id=img_id,
                category_id=0,
                iscrowd=0,
                segmentation=rle,
                area=bbox[2] * bbox[3],
                bbox=bbox,
            )
            ann_infos.append(ann_info)
            ann_id += 1
        img_infos.append(img_info)
        img_id += 1
    coco = init_coco()
    coco['images'] = img_infos
    coco['annotations'] = ann_infos
    return coco


def main():
    data_root = '../data/nips/Train_Pre_3class_slide'
    all_idx = [x for x in range(len(os.listdir(os.path.join(data_root, 'images'))))]
    dtrainval = gen_coco(os.path.join(data_root, 'images'), all_idx)
    mmcv.dump(dtrainval, os.path.join(data_root, 'dtrainval.json'))
    d = set()
    for file in os.listdir(os.path.join(data_root, 'images')):
        basename = file.split('(')[0]
        d.add(basename)
    d = list(d)
    base_idx = [i for i in range(len(d))]
    train_num = int(len(base_idx) * 0.8)
    for fold in range(1):
        base_samples = random.sample(d, train_num)
        train_samples = []
        for i, file in enumerate(os.listdir(os.path.join(data_root, 'images'))):
            basename = file.split('(')[0]
            if basename in base_samples:
                train_samples.append(i)
        train_coco = gen_coco(os.path.join(data_root, 'images'), train_samples)
        mmcv.dump(train_coco, os.path.join(data_root, f'dtrain_g{fold}.json'))

        val_samples = list(set(all_idx) - set(train_samples))
        val_coco = gen_coco(os.path.join(data_root, 'images'), val_samples)
        mmcv.dump(val_coco, os.path.join(data_root, f'dval_g{fold}.json'))

if __name__ == '__main__':
    main()



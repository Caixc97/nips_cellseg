import tifffile as tif
from skimage import io
import numpy as np
from skimage import exposure
import cv2


def normalize(img_data):
    pre_img_data = np.zeros(img_data.shape, dtype=np.uint8)
    for i in range(3):
        img_channel_i = img_data[:, :, i]
        if len(img_channel_i[np.nonzero(img_channel_i)]) > 0:
            pre_img_data[:, :, i] = normalize_channel(img_channel_i, lower=1, upper=99)
    return pre_img_data


def normalize_channel(img, lower=1, upper=99):
    non_zero_vals = img[np.nonzero(img)]
    percentiles = np.percentile(non_zero_vals, [lower, upper])
    if percentiles[1] - percentiles[0] > 0.001:
        img_norm = exposure.rescale_intensity(img, in_range=(percentiles[0], percentiles[1]), out_range='uint8')
    else:
        img_norm = img
    return img_norm.astype(np.uint8)


def load_from_file(filepath):
    if filepath.endswith('.tif') or filepath.endswith('.tiff'):
        img_data = tif.imread(filepath)
        if len(img_data.shape) == 3:
            img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
    else:
        img_data = io.imread(filepath)
        img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
    if len(img_data.shape) == 2:
        img_data = np.repeat(np.expand_dims(img_data, axis=-1), 3, axis=-1)
    elif len(img_data.shape) == 3 and img_data.shape[-1] > 3:
        img_data = img_data[:, :, :3]
    # img_data = normalize(img_data)
    return img_data


def show_bbox(data, result, model, thr=0.5, show=True):
    from mmcv.image import tensor2imgs
    import matplotlib.pyplot as plt
    import mmcv
    try:
        img_tensor = data['img'][0].data[0]
    except:
        img_tensor = data['img'][0]
    img_metas = data['img_metas'][0].data[0]
    imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
    for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
        h, w, _ = img_meta['img_shape']
        img_show = img[:h, :w, :]
        ori_h, ori_w = img_meta['ori_shape'][:-1]
        img_show = mmcv.imresize(img_show, (ori_w, ori_h))
        model.module.CLASSES = ['cell']
        img_show = model.module.show_result(img_show, result[i],
        bbox_color=(255, 128, 128),
        text_color=(255, 128, 128),
        score_thr=thr)
        if not show:
            return img_show
        else:
            plt.imshow(img_show)
            plt.show()
            plt.close()


def show_result(img,
                seg_preds,
                ind_conf=None,
                bbox_color=(72, 101, 241),
                text_color=(72, 101, 241),
                mask_color=None,
                thickness=2,
                font_size=13,
                win_name='',
                show=False,
                wait_time=0,
                out_file=None):
    import mmcv
    from mmdet.core.visualization import imshow_det_bboxes
    img_bbox = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    color_mask = np.zeros(img_bbox.shape)
    img_bbox = cv2.cvtColor(img_bbox, cv2.COLOR_RGB2BGR)
    color_mask[:, :, 1] = 140
    color_mask[:, :, 2] = 255
    color_mask[seg_preds != 0] = [255, 140, 0]
    img_bbox[seg_preds != 0] = img_bbox[seg_preds != 0] / 2 + color_mask[seg_preds != 0] / 2
    bboxes = []
    for index in np.unique(seg_preds):
        if index == 0:
            continue
        pos = (seg_preds == index).nonzero()
        y_min, x_min = np.min(pos, axis=1)
        y_max, x_max = np.max(pos, axis=1)
        if ind_conf != None:
            bboxes.append([x_min, y_min, x_max, y_max, ind_conf[index]])
        else:
            bboxes.append([x_min, y_min, x_max, y_max, 1])
    bboxes = np.array(bboxes)

    img_bbox = imshow_det_bboxes(
        img_bbox,
        bboxes,
        np.array([0]*len(bboxes)),
        None,
        class_names=['cell'],
        bbox_color=bbox_color,
        text_color=text_color,
        mask_color=mask_color,
        thickness=thickness,
        font_size=font_size,
        win_name=win_name,
        show=show,
        wait_time=wait_time,
        out_file=out_file)

    img_bbox, _ = rescale_img(img_bbox, 1536)
    img, _ = rescale_img(img, 1536)
    img = np.hstack((img, img_bbox))
    return img

def rescale_img(img, max_len=1536):
    h = img.shape[0]
    w = img.shape[1]
    if h > max_len:
        scale = max_len/h
        new_w = int(w * scale)
        img = cv2.resize(img.astype('float32'), (new_w, max_len))
    if w > max_len:
        scale = max_len/w
        new_h = int(h * scale)
        img = cv2.resize(img.astype('float32'), (max_len, new_h))
    return img, img.shape



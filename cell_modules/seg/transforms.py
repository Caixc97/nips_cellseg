import torch
from torchvision.ops import roi_align, roi_pool
from cv2 import resize, INTER_NEAREST

from mmcv.image.geometric import imflip, imresize
import numpy as np
from mmseg.datasets import PIPELINES
from mmseg.datasets.pipelines.formatting import to_tensor, DC


@PIPELINES.register_module()
class BoxJitter(object):
    def __init__(self, jittor_range=(0.8, 1.2), prob=0.5):
        self.jittor_range = jittor_range
        self.prob = prob

    def __call__(self, results):
        if np.random.random() >= self.prob:
            return results

        x1, y1, x2, y2 = results['bbox']
        img_h, img_w, _ = results['ori_shape']
        xc, yc = (x1 + x2) / 2, (y1 + y2) / 2

        t = (yc - y1) * np.random.uniform(*self.jittor_range)
        l = (xc - x1) * np.random.uniform(*self.jittor_range)
        b = (y2 - yc) * np.random.uniform(*self.jittor_range)
        r = (x2 - xc) * np.random.uniform(*self.jittor_range)

        x1 = xc - l
        y1 = yc - t
        x2 = xc + r
        y2 = yc + b

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)
        if y2 > y1 and x2 > x1:
            results['bbox'] = [x1, y1, x2, y2]
            return results
        else:
            print('Invalid box:', x1, y1, x2, y2)
            return results



@PIPELINES.register_module()
class ROIAlign(object):
    def __init__(
        self, output_size, spatial_scale=1.0, sampling_ratio=0, aligned=True
    ):
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned

    def __call__(self, results):
        x1, y1, x2, y2 = results['bbox']  # xyxy
        # crop img
        img = results['img']  # hwc
        img_crop = img[int(y1+0.5):int(y2+0.5), int(x1+0.5):int(x2+0.5), :]
        img_crop = resize(img_crop, self.output_size, interpolation=INTER_NEAREST)

        results['img'] = img_crop
        results['img_shape'] = img_crop.shape
        results['pad_shape'] = img_crop.shape
        results['scale_factor'] = 1.0
        results['keep_ratio'] = False

        # crop mask
        if 'gt_semantic_seg' in results:
            mask = results['gt_semantic_seg']
            input = torch.from_numpy(mask[None, None, ...]).float()
            rois = torch.FloatTensor([[0, x1, y1, x2, y2]])
            mask_crop = roi_align(
                input, rois, self.output_size, self.spatial_scale,
                self.sampling_ratio, self.aligned
            )[0, 0].numpy()
            mask_crop = (mask_crop >= 0.5).astype(int)
            results['gt_semantic_seg'] = mask_crop
        return results



@PIPELINES.register_module()
class CropResize(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, results):
        x1, y1, x2, y2 = [round(_) for _ in results['bbox']]  # xyxy

        img = results['img']  # hwc
        img_crop = img[y1:y2, x1:x2]

        results['img'] = mmcv.imresize(img_crop, self.output_size)
        results['img_shape'] = img_crop.shape
        results['pad_shape'] = img_crop.shape
        results['scale_factor'] = 1.0
        results['keep_ratio'] = False

        # crop mask
        if 'gt_semantic_seg' in results:
            mask = results['gt_semantic_seg']
            mask_crop = mask[y1:y2, x1:x2]
            results['gt_semantic_seg'] = mmcv.imresize(
                mask_crop, self.output_size, interpolation='nearest'
            )
        return results


@PIPELINES.register_module()
class FlipRotate(object):
    def __init__(self):
        pass

    def __call__(self, results):
        img = results['img']
        mask = results['gt_semantic_seg']
        results['flip'] = False
        results['flip_direction'] = None
        for flip_direction in ('horizontal', 'vertical', 'diagonal'):
            if np.random.random() < 0.5:
                img = mmcv.imflip(img, direction='horizontal')
                mask = mmcv.imflip(mask, direction='horizontal')
                results['flip'] = True
                results['flip_direction'] = flip_direction

        results['img'] = img
        results['gt_semantic_seg'] = mask

        return results


@PIPELINES.register_module()
class BBoxFormat(object):
    def __call__(self, results):
        results['bbox'] = DC(to_tensor(results['bbox']))
        return results

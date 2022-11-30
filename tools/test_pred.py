import argparse
import warnings

import torch.nn

warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser('RedCat solution for Microscopy image segmentation', add_help=False)
    parser.add_argument('-i', '--input_path', default='./inputs', type=str, help='training data path; subfolders: images, labels')
    parser.add_argument("-o", '--output_path', default='./outputs', type=str, help='output path')
    parser.add_argument('--det_ckp', default='checkpoints/det.pth', type=str, help='detection checkpoint path')
    parser.add_argument('--seg_ckp', default='checkpoints/seg.pth', type=str, help='segmentation checkpoint path')
    args = parser.parse_args()
    data_path = args.input_path
    output_path = args.output_path
    seg_cfg = 'configs/seg/unet_nips.py'
    det_cfg = 'configs/det/yolox_x_nips.py'

    import sys
    sys.path.append('.')
    import time
    import cell_modules
    import cv2
    cv2.setNumThreads(0)

    import os
    import os.path as osp
    from tqdm import tqdm
    import numpy as np
    import tifffile as tif

    det_ckp_list = [args.det_ckp]
    seg_ckp_list = [args.seg_ckp]
    from mmcv.utils import Config
    det_cfg = Config.fromfile(det_cfg)
    seg_cfg = Config.fromfile(seg_cfg)
    det_cfg.model.train_cfg = None
    seg_cfg.model.train_cfg = None
    from torch import from_numpy
    from torch.autograd import no_grad
    from mmcv.parallel import MMDataParallel, collate
    from mmcv.runner import load_checkpoint
    from mmdet.models import build_detector as build_detector_det
    from mmdet.datasets.pipelines import Compose as Compose_det
    from mmseg.models import build_segmentor as build_segmentor_seg
    from mmseg.datasets.pipelines import Compose as Compose_seg
    from utils import normalize, rescale_img, load_from_file, show_bbox, show_result
    from torchvision.ops import nms
    # We have tried mixed expert models, but the results are not good.
    # I'm too lazy to change the code back :)
    class ModelContainer():
        def __init__(self):
            self.model_list = [None, None, None, None]

        def predict(self, data, cluster, mode='det'):
            if mode == 'det':
                model_det, _ = self.get_model(cluster)
                return model_det(return_loss=False, rescale=True, **data)
            else:
                _, model_seg = self.get_model(cluster)
                return model_seg(return_loss=False, **data)

        def get_model(self, cluster):
            if self.model_list[cluster] == None:
                self.model_list[cluster] = [build_detector_det(det_cfg.model, test_cfg=det_cfg.get('test_cfg')),
                                            build_segmentor_seg(seg_cfg.model, test_cfg=seg_cfg.get('test_cfg'))]
                checkpoint = load_checkpoint(self.model_list[cluster][0], det_ckp_list[cluster], map_location='cpu')
                model_det = MMDataParallel(self.model_list[cluster][0], device_ids=[0])
                model_det.eval()

                checkpoint = load_checkpoint(self.model_list[cluster][1], seg_ckp_list[cluster], map_location='cpu')
                model_seg = MMDataParallel(self.model_list[cluster][1], device_ids=[0])
                model_seg.eval()

                self.model_list[cluster] = [model_det, model_seg]
            return self.model_list[cluster]

        def init_model(self, cluster):
            if self.model_list[cluster] == None:
                self.model_list[cluster] = [build_detector_det(det_cfg.model, test_cfg=det_cfg.get('test_cfg')),
                                            build_segmentor_seg(seg_cfg.model, test_cfg=seg_cfg.get('test_cfg'))]
                checkpoint = load_checkpoint(self.model_list[cluster][0], det_ckp_list[cluster], map_location='cpu')
                model_det = MMDataParallel(self.model_list[cluster][0], device_ids=[0])
                model_det.eval()

                checkpoint = load_checkpoint(self.model_list[cluster][1], seg_ckp_list[cluster], map_location='cpu')
                model_seg = MMDataParallel(self.model_list[cluster][1], device_ids=[0])
                model_seg.eval()

                self.model_list[cluster] = [model_det, model_seg]


    model_container = ModelContainer()

    # data piepline
    for i, p in enumerate(det_cfg.data.test.pipeline):
        if p['type'] == 'LoadImageFromFile':
            del det_cfg.data.test.pipeline[i]
            break
    det_pipeline = Compose_det(det_cfg.data.test.pipeline)
    seg_pipeline = Compose_seg(seg_cfg.data.test.pipeline)


    def load_img(filepath):
        results = {'img_info': {'filename': filepath}, 'img_prefix': None}
        img_data = load_from_file(filepath)
        img = img_data.astype(np.float32)
        results['filename'] = filepath
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results


    def organise_data(results, pipeline, box_info=None):
        if box_info is None:
            results['img'] = results['img'].astype(np.float32)
            data = pipeline(results)
            return data
        else:
            results['bbox'] = box_info['bbox']
            data = pipeline(results)
            return data

    def align_bbox(boxes, flip, rot):
        boxes[boxes<0] = 0
        boxes[boxes>1] = 1
        if flip:
            x_max = 1 - boxes[:, 0]
            x_min = 1 - boxes[:, 2]
            boxes[:, 0] = x_min
            boxes[:, 2] = x_max
        if rot == 1:
            x_min = 1 - boxes[:, 3]
            x_max = 1 - boxes[:, 1]
            y_min = boxes[:, 0]
            y_max = boxes[:, 2]
            boxes[:, 1] = y_min
            boxes[:, 3] = y_max
            boxes[:, 0] = x_min
            boxes[:, 2] = x_max
        elif rot == 2:
            x_min = 1 - boxes[:, 2]
            x_max = 1 - boxes[:, 0]
            y_min = 1 - boxes[:, 3]
            y_max = 1 - boxes[:, 1]
            boxes[:, 1] = y_min
            boxes[:, 3] = y_max
            boxes[:, 0] = x_min
            boxes[:, 2] = x_max
        elif rot == 3:
            x_min = boxes[:, 1]
            x_max = boxes[:, 3]
            y_min = 1 - boxes[:, 2]
            y_max = 1 - boxes[:, 0]
            boxes[:, 0] = x_min
            boxes[:, 2] = x_max
            boxes[:, 1] = y_min
            boxes[:, 3] = y_max


    def det_win_predict(data, scale, flip=False, rot=0, cluster=0):
        win_size = [0, 0]
        win_size[0] = int(data['img_shape'][0]//scale)
        win_size[1] = int(data['img_shape'][1]//scale)
        result = np.empty(shape=(0, 5), dtype='float32')
        ori_data = data
        data = data.copy()
        if rot != 0:
            if rot == 1 or rot == 3:
                data['img_shape'] = (data['img_shape'][1], data['img_shape'][0], 3)
                win_size[0], win_size[1] = win_size[1], win_size[0]
            data['img'] = np.rot90(data['img'], rot)
        if flip:
            data['img'] = np.flip(data['img'], 1)
        for row in range(scale):
            for col in range(scale):
                window = data['img'][row*win_size[0]:row*win_size[0]+int(win_size[0] * 1.2), col*win_size[1]:col*win_size[1]+int(win_size[1]*1.2), :]
                this_data = {
                    'img_info': data['img_info'],
                    'img_prefix': data['img_prefix'],
                    'filename': data['filename'],
                    'ori_filename': data['ori_filename'],
                    'img': window,
                    'img_shape': window.shape,
                    'ori_shape': window.shape,
                    'img_fields': data['img_fields']
                }
                this_data = organise_data(this_data, det_pipeline)
                this_data = collate([this_data])
                this_result = model_container.predict(this_data, cluster, 'det')
                this_result = this_result[0][0]
                this_result += [win_size[1]*col, win_size[0]*row, win_size[1]*col, win_size[0]*row, 0]
                mask1 = (this_result[:, 2] - this_result[:, 0]) > int(win_size[1] * 1.2) * 0.8
                mask2 = (this_result[:, 3] - this_result[:, 1]) > int(win_size[0] * 1.2) * 0.8
                this_result = this_result[~(mask1*mask2)]
                mask3 = this_result[:, 0] == this_result[:, 2]
                mask4 = this_result[:, 1] == this_result[:, 3]
                this_result = this_result[~(mask3+mask4)]
                result = np.append(result, this_result, axis=0)
        result /= [data['img_shape'][1], data['img_shape'][0], data['img_shape'][1], data['img_shape'][0], 1]
        if rot != 0 or flip:
            align_bbox(result, flip, rot)
        return result

    def det_tta_predict(data, cluster=0):
        boxes_list = np.zeros(shape=(0, 4))
        scores_list = np.zeros(shape=(0, ))
        for i in range(4):
            result = det_win_predict(data, 1, flip=False, rot=i, cluster=cluster)
            boxes_list = np.append(boxes_list, result[:, :4], axis=0)
            scores_list = np.append(scores_list, result[:, 4], axis=0)
        for i in range(4):
            result = det_win_predict(data, 1, flip=True, rot=i, cluster=cluster)
            boxes_list = np.append(boxes_list, result[:, :4], axis=0)
            scores_list = np.append(scores_list, result[:, 4], axis=0)
        if len(boxes_list) != 0 and boxes_list[0].shape[0] != 0:
            boxes_list = from_numpy(boxes_list)
            scores_list = from_numpy(scores_list)
            index = nms(boxes_list, scores_list, iou_threshold=0.5)
            boxes = boxes_list[index].numpy()
            scores = scores_list[index].numpy()
            boxes *= [data['img_shape'][1], data['img_shape'][0], data['img_shape'][1], data['img_shape'][0]]
            result = np.insert(boxes, 4, scores, axis=1).astype('float32')
        return [[result.astype("float32")]]



    def inference(data, cluster):
        img = data
        img['img'] = normalize(img['img'])
        if (img['img_shape'][0] > 1536 or img['img_shape'][1] > 1536):
            ori_shape = img['img'].shape
            img['img'], new_shape = rescale_img(img['img'])
            img['ori_shape'] = new_shape
            img['img_shape'] = new_shape
        else:
            ori_shape = None
        with no_grad():
            # det
            result = det_tta_predict(img, cluster)
            # seg
            ins_seg = np.zeros(shape=img['img_shape'][:2], dtype=np.int32)
            num_ins = 0
            batch_size = 96
            batch = []
            batch_count = 0
            img['img'] = np.ascontiguousarray(img['img'])
            index_confidence = {}
            for cat_id, cat_pred in enumerate(result[0]):
                for ins_id, (x1, y1, x2, y2, score) in enumerate(np.flip(cat_pred,axis=0)):
                    if x2 - x1 > img['img_shape'][0] * 0.5 or y2 - y1 > img['img_shape'][1] * 0.5:
                        continue
                    if score < 0.5:
                        continue
                    if x2 - x1 < 3 or y2 - y1 < 3:
                        continue
                    # print(x1, y1, x2, y2)
                    box_info = \
                        dict(
                            filename=filename,
                            bbox=[x1, y1, x2, y2],
                            score=score,
                            category_id=cat_id,
                            height=img['img_shape'][0],
                            width=img['img_shape'][1],
                        )
                    seg_data = dict(
                        ori_filename=filename,
                        filename=filename,
                        img=img['img'],
                        ori_shape=img['img_shape'],
                        shape=img['img_shape'],
                        bbox=box_info['bbox']
                    )
                    seg_data = organise_data(seg_data, seg_pipeline, box_info)
                    if batch_count < batch_size:
                        batch.append(seg_data)
                        batch_count += 1
                    else:
                        seg_data = collate(batch, samples_per_gpu=batch_size)
                        seg_preds = model_container.predict(seg_data, cluster, 'seg')
                        for seg_pred in seg_preds:
                            num_ins += 1
                            ins_seg[seg_pred['seg_pred']] = num_ins
                            index_confidence[num_ins] = score * seg_pred['confidence']
                        batch = []
                        batch_count = 0
                if len(batch) != 0:
                    seg_data = collate(batch, samples_per_gpu=len(batch))
                    seg_preds = model_container.predict(seg_data, cluster, 'seg')
                    for seg_pred in seg_preds:
                        num_ins += 1
                        ins_seg[seg_pred['seg_pred']] = num_ins
                        index_confidence[num_ins] = score * seg_pred['confidence']
            if ori_shape != None:
                ins_seg = cv2.resize(ins_seg.astype(np.float32), (ori_shape[1], ori_shape[0]),
                                       interpolation=cv2.INTER_NEAREST).astype(np.int32)
            return ins_seg, num_ins, index_confidence



    def slide_win_inference(data, win_size=1536, stride=1280):
        cluster = 0
        total_col = max(int((data['img_shape'][1] - win_size + stride - 1)/stride) + 1, 1)
        total_row = max(int((data['img_shape'][0] - win_size + stride - 1)/stride) + 1, 1)
        total_num_ins = 0
        total_seg_preds = np.zeros(shape=data['img_shape'][:2], dtype=np.int32)
        if total_row * total_row > 10:
            pbar = tqdm(total=total_row*total_col)
        count = 0
        index_confidence = {}
        for row in range(total_row):
            for col in range(total_col):
                count += 1
                window = data['img'][row*stride:row*stride+win_size, col*stride:col*stride+win_size, :]
                this_data = {
                    'img_info': data['img_info'],
                    'img_prefix': data['img_prefix'],
                    'filename': data['filename'],
                    'ori_filename': data['ori_filename'],
                    'img': window,
                    'img_shape': window.shape,
                    'ori_shape': window.shape,
                    'img_fields': data['img_fields']
                }
                seg_preds, num_ins, ind_conf = inference(this_data, cluster)
                mask = seg_preds != 0
                seg_preds[mask] += total_num_ins
                new_ind_conf = {}
                for k, v in ind_conf.items():
                    new_ind_conf[k+total_num_ins] = v
                ind_conf = new_ind_conf
                total_num_ins += num_ins
                preds_window = total_seg_preds[row*stride:row*stride+win_size, col*stride:col*stride+win_size]
                overlap = (preds_window != 0) * (mask)
                mapping = dict(zip(seg_preds[overlap], preds_window[overlap]))
                for k, v in mapping.items():
                    seg_preds[seg_preds == k] = v
                    index_confidence[v] = index_confidence[v]/2 + ind_conf[k]/2
                index_confidence.update(ind_conf)
                mask = seg_preds != 0
                total_seg_preds[row * stride:row * stride + win_size, col * stride:col * stride + win_size][mask] = seg_preds[mask]
                if total_row * total_row > 10:
                    pbar.update(1)
        return total_seg_preds, total_num_ins, index_confidence


    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(os.path.join(output_path, 'tif')):
        os.mkdir(os.path.join(output_path, 'tif'))
    if not os.path.exists(os.path.join(output_path, 'visualise')):
        os.mkdir(os.path.join(output_path, 'visualise'))

    data_list = []
    for img in os.listdir(data_path):
        data_list.append(osp.join(data_path, img))

    for img in tqdm(data_list):
        filename = img
        img = os.path.basename(img)
        img_id = img.split('.')[0]
        t0 = time.time()
        img = load_img(filename)
        if img is None:
            continue
        seg_preds, num_ins, ind_conf = slide_win_inference(img)
        tif.imwrite(os.path.join(output_path, 'tif', img_id + '_label.tiff'), seg_preds)
        t1 = time.time()
        img = img['img']
        ori_shape = img.shape[:2]
        num_ins = len(np.unique(seg_preds)-1)
        # num_ins = 0
        img = normalize(img)
        img = show_result(img, seg_preds, ind_conf)
        from skimage import io
        io.imsave(os.path.join(output_path, 'visualise', img_id + '_vis.png'), img.astype('uint8'))
        print(f'Prediction finished: {img_id}; img size = {ori_shape}; costing: {t1 - t0:.2f}s; num_ins: {num_ins}')


if __name__ == '__main__':
    import time
    main()








# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str


parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--dataset', type=str,
        help='datasets')
parser.add_argument('--config', default='', type=str,
        help='config file')
parser.add_argument('--snapshot', default='', type=str,
        help='snapshot of models to eval')
parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--vis', action='store_true',
        help='whether visualzie result')
args = parser.parse_args()

torch.set_num_threads(1)

def compute_velocity(prev_bbox, curr_bbox, time_interval):
    prev_center = np.array([prev_bbox[0] + prev_bbox[2] / 2, prev_bbox[1] + prev_bbox[3] / 2])
    curr_center = np.array([curr_bbox[0] + curr_bbox[2] / 2, curr_bbox[1] + curr_bbox[3] / 2])
    velocity = np.linalg.norm(curr_center - prev_center) / time_interval
    return velocity

def compute_angular_change(prev_bbox, curr_bbox):
    prev_vector = np.array([prev_bbox[2], prev_bbox[3]])
    curr_vector = np.array([curr_bbox[2], curr_bbox[3]])
    angle = np.arccos(
        np.clip(
            np.dot(prev_vector, curr_vector) / (np.linalg.norm(prev_vector) * np.linalg.norm(curr_vector) + 1e-6),
            -1,
            1
        )
    )
    return np.degrees(angle)

def dynamic_lambda(velocity, angular_change):
    if velocity > 10 or angular_change > 20:
        return 0.7
    return 0.5

def optimize_perturbation(img, tracker, target_bbox, max_steps=10, epsilon=0.05):
    perturb = np.zeros_like(img, dtype=np.float32)
    best_score = 1.0

    # 初始扰动，增加多样性
    random_noise = np.random.uniform(-epsilon * 255, epsilon * 255, img.shape).astype(np.float32)
    perturb += random_noise

    for step in range(max_steps):
        perturbed_img = np.clip(img + perturb, 0, 255).astype(np.uint8)
        outputs = tracker.track(perturbed_img)
        pred_bbox = outputs['bbox']
        score = overlap_ratio(target_bbox, pred_bbox)

        if score < best_score:
            best_score = score
        else:
            epsilon *= 0.9  # 降低步长以避免发散

        gradient = (perturbed_img - img).astype(np.float32)
        gradient = gradient / (np.linalg.norm(gradient) + 1e-8)
        perturb -= epsilon * gradient

        if best_score < 0.5:  # IoU 达到目标值，停止优化
            break

    return np.clip(img + perturb, 0, 255).astype(np.uint8)

def main():
    cfg.merge_from_file(args.config)
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(cur_dir, '../testing_dataset', args.dataset)

    model = ModelBuilder()
    model = load_pretrain(model, args.snapshot).cuda().eval()
    tracker = build_tracker(model)
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    model_name = args.snapshot.split('/')[-1].split('.')[0]
    total_lost = 0

    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:
        for v_idx, video in enumerate(dataset):
            if args.video != '' and video.name != args.video:
                continue

            frame_counter = 0
            lost_number = 0
            toc = 0
            pred_bboxes = []
            prev_bbox = None
            time_interval = 1 / 30  # 假设30fps
            num_frames = 0

            for idx, (img, gt_bbox) in enumerate(video):
                num_frames += 1
                if len(gt_bbox) == 4:
                    gt_bbox = [gt_bbox[0], gt_bbox[1],
                               gt_bbox[0], gt_bbox[1] + gt_bbox[3] - 1,
                               gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1] + gt_bbox[3] - 1,
                               gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1]]

                tic = cv2.getTickCount()
                if idx == frame_counter:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                    tracker.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    pred_bboxes.append(1)
                    prev_bbox = gt_bbox_
                elif idx > frame_counter:
                    velocity = compute_velocity(prev_bbox, gt_bbox_, time_interval) if prev_bbox else 0
                    angular_change = compute_angular_change(prev_bbox, gt_bbox_) if prev_bbox else 0
                    lambda_ = dynamic_lambda(velocity, angular_change)
                    
                    img = optimize_perturbation(img, tracker, np.array(gt_bbox_))

                    outputs = tracker.track(img)
                    pred_bbox = outputs['bbox']
                    if cfg.MASK.MASK:
                        pred_bbox = outputs['polygon']
                    overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
                    if overlap > 0:
                        pred_bboxes.append(pred_bbox)
                    else:
                        pred_bboxes.append(2)
                        frame_counter = idx + 5
                        lost_number += 1
                toc += cv2.getTickCount() - tic

                prev_bbox = pred_bbox

                # 可视化
                if args.vis and idx > frame_counter:
                    cv2.polylines(img, [np.array(gt_bbox, np.int).reshape((-1, 1, 2))],
                                True, (0, 255, 0), 3)
                    if cfg.MASK.MASK:
                        cv2.polylines(img, [np.array(pred_bbox, np.int).reshape((-1, 1, 2))],
                                    True, (0, 255, 255), 3)
                    else:
                        bbox = list(map(int, pred_bbox))
                        cv2.rectangle(img, (bbox[0], bbox[1]),
                                    (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, f'Frame: {idx}', (40, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(img, f'Lost: {lost_number}', (40, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)

            toc /= cv2.getTickFrequency()
            fps = num_frames / toc
            total_lost += lost_number  # 累加 lost 次数

            print(f'({v_idx + 1}) Video: {video.name} Time: {toc:.1f}s Speed: {fps:.1f}fps Lost: {lost_number}')

    print(f"{model_name} total lost: {total_lost}")



def overlap_ratio(rect1, rect2):
    '''
    Compute overlap ratio between two rects
    - rect: 1d array of [x,y,w,h] or
            2d array of N x [x,y,w,h]
    '''
    rect1 = np.array(rect1)  # 确保 rect1 是 NumPy 数组
    rect2 = np.array(rect2)  # 确保 rect2 是 NumPy 数组

    if rect1.ndim == 1:
        rect1 = rect1[None, :]
    if rect2.ndim == 1:
        rect2 = rect2[None, :]
        
    left = np.maximum(rect1[:, 0], rect2[:, 0])
    right = np.minimum(rect1[:, 0] + rect1[:, 2], rect2[:, 0] + rect2[:, 2])
    top = np.maximum(rect1[:, 1], rect2[:, 1])
    bottom = np.minimum(rect1[:, 1] + rect1[:, 3], rect2[:, 1] + rect2[:, 3])

    intersect = np.maximum(0, right - left) * np.maximum(0, bottom - top)
    union = rect1[:, 2] * rect1[:, 3] + rect2[:, 2] * rect2[:, 3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou

def orthogonal_perturbation(delta, prev_sample, target_sample):
    size = int(max(prev_sample.shape[0]/4, prev_sample.shape[1]/4, 224))
    prev_sample_temp = np.resize(prev_sample, (size, size, 3))
    target_sample_temp = np.resize(target_sample, (size, size, 3))
    # Generate perturbation
    perturb = np.random.randn(size, size, 3)
    perturb /= get_diff(perturb, np.zeros_like(perturb))
    perturb *= delta * np.mean(get_diff(target_sample_temp, prev_sample_temp))
    # Project perturbation onto sphere around target
    diff = (target_sample_temp - prev_sample_temp).astype(np.float32)
    diff /= get_diff(target_sample_temp, prev_sample_temp)
    diff = diff.reshape(3, size, size)
    perturb = perturb.reshape(3, size, size)
    for i, channel in enumerate(diff):
        perturb[i] -= np.dot(perturb[i], channel) * channel
    perturb = perturb.reshape(size, size, 3)
    perturb_temp = np.resize(perturb, (prev_sample.shape[0], prev_sample.shape[1], 3))
    return perturb_temp

def forward_perturbation(epsilon, prev_sample, target_sample):
    perturb = (target_sample - prev_sample).astype(np.float32)
    perturb /= get_diff(target_sample, prev_sample)
    perturb *= epsilon
    return perturb

def get_diff(sample_1, sample_2):
    sample_1 = sample_1.reshape(3, sample_1.shape[0], sample_1.shape[1])
    sample_2 = sample_2.reshape(3, sample_2.shape[0], sample_2.shape[1])
    sample_1 = np.resize(sample_1, (3, 271, 271))
    sample_2 = np.resize(sample_2, (3, 271, 271))

    diff = []
    for i, channel in enumerate(sample_1):
        diff.append(np.linalg.norm((channel - sample_2[i]).astype(np.float32)))
    return np.array(diff)


if __name__ == '__main__':
    main()

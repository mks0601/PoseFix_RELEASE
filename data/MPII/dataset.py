#!/usr/bin/python3
# coding=utf-8

import os
import os.path as osp
import numpy as np
import cv2
import json
import pickle
import matplotlib.pyplot as plt
from scipy.io import savemat

import sys
cur_dir = os.path.dirname(__file__)
sys.path.insert(0, osp.join(cur_dir, 'PythonAPI'))
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class Dataset(object):
 
    dataset_name = 'MPII'
    num_kps = 16
    kps_names = ["r_ankle", "r_knee","r_hip", 
                    "l_hip", "l_knee", "l_ankle",
                  "pelvis", "throax",
                  "upper_neck", "head_top",
                  "r_wrist", "r_elbow", "r_shoulder",
                  "l_shoulder", "l_elbow", "l_wrist"]
    kps_lines = [(0, 1), (1, 2), (2, 6), (7, 12), (12, 11), (11, 10), (5, 4), (4, 3), (3, 6), (7, 13), (13, 14), (14, 15), (6, 7), (7, 8), (8, 9)]
    kps_symmetry = [(0, 5), (1, 4), (2, 3), (10, 15), (11, 14), (12, 13)]
    kps_sigmas = np.array([
       .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87,
       .87, .89, .89]) / 10.0
    ignore_kps = []

    test_on_trainset_path = osp.join('..', 'data', dataset_name, 'input_pose', 'test_on_trainset', 'result.json')
    input_pose_path = osp.join('..', 'data', dataset_name, 'input_pose', 'person_keypoints_val2017_CPN50-256x192_results.json') # set directory of the input pose

    img_path = osp.join('..', 'data', dataset_name)
    train_annot_path = osp.join('..', 'data', dataset_name, 'annotations', 'train.json')
    test_annot_path = osp.join('..', 'data', dataset_name, 'annotations', 'test.json')
   

    def load_train_data(self):
        coco = COCO(self.train_annot_path)
        train_data = []
        for aid in coco.anns.keys():
            ann = coco.anns[aid]
            imgname = coco.imgs[ann['image_id']]['file_name']
            joints = ann['keypoints']

            if (ann['image_id'] not in coco.imgs) or ann['iscrowd'] or (ann['num_keypoints'] == 0):
                continue

            # sanitize bboxes
            x, y, w, h = ann['bbox']
            img = coco.loadImgs(ann['image_id'])[0]
            width, height = img['width'], img['height']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if ann['area'] > 0 and x2 >= x1 and y2 >= y1:
                bbox = [x1, y1, x2-x1, y2-y1]
            else:
                continue

            data = dict(image_id = ann['image_id'], imgpath = imgname, bbox=bbox, joints=joints)
            train_data.append(data)

        return train_data
    
    def load_annot(self, db_set):
        if db_set == 'train':
            coco = COCO(self.train_annot_path)
        elif db_set == 'test':
            coco = COCO(self.test_annot_path)
        else:
            print('Unknown db_set')
            assert 0

        return coco

    def load_imgid(self, annot):
        return annot.imgs

    def imgid_to_imgname(self, annot, imgid, db_set):
        imgs = annot.loadImgs(imgid)
        imgname = [i['file_name'] for i in imgs]
        return imgname

    def input_pose_load(self, annot, db_set):
        
        gt_img_id = self.load_imgid(annot)

        with open(self.input_pose_path, 'r') as f:
            input_pose = json.load(f)
        for i in range(len(input_pose)):
            input_pose[i]['score'] = np.mean(input_pose[i]['scores'])
        input_pose = [i for i in input_pose if i['image_id'] in gt_img_id]
        input_pose = [i for i in input_pose if i['category_id'] == 1]
        input_pose = [i for i in input_pose if i['score'] > 0]
        input_pose.sort(key=lambda x: (x['image_id'], x['score']), reverse=True)

        img_id = []
        for i in input_pose:
            img_id.append(i['image_id'])
        imgname = self.imgid_to_imgname(annot, img_id, db_set)
        for i in range(len(input_pose)):
            input_pose[i]['imgpath'] = imgname[i]

        # bbox generate
        for i in range(len(input_pose)):
            input_pose[i]['estimated_joints'] = input_pose[i]['keypoints']
            input_pose[i]['estimated_score'] = input_pose[i]['scores']
            del input_pose[i]['keypoints']
            del input_pose[i]['score']
            del input_pose[i]['scores']
            
            coords = np.array(input_pose[i]['estimated_joints']).reshape(self.num_kps,3)
            coords = np.delete(coords, self.ignore_kps, axis=0)

            xmin = np.min(coords[:,0])
            xmax = np.max(coords[:,0])
            width = xmax - xmin if xmax > xmin else 20
            center = (xmin + xmax)/2.
            xmin = center - width/2.*1.2
            xmax = center + width/2.*1.2

            ymin = np.min(coords[:,1])
            ymax = np.max(coords[:,1])
            height = ymax - ymin if ymax > ymin else 20
            center = (ymin + ymax)/2.
            ymin = center - height/2.*1.2
            ymax = center + height/2.*1.2

            input_pose[i]['bbox'] = [xmin,ymin,xmax-xmin,ymax-ymin]

        return input_pose

    def evaluation(self, result, annot, result_dir, db_set):
        result_path = osp.join(result_dir, 'result.mat')
        savemat(result_path, mdict=result)

    def vis_keypoints(self, img, kps, kp_thresh=0.4, alpha=1):

        # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, len(self.kps_lines) + 2)]
        colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

        # Perform the drawing on a copy of the image, to allow for blending.
        kp_mask = np.copy(img)

        # Draw the keypoints.
        for l in range(len(self.kps_lines)):
            i1 = self.kps_lines[l][0]
            i2 = self.kps_lines[l][1]
            p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
            p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
            if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
                cv2.line(
                    kp_mask, p1, p2,
                    color=colors[l], thickness=2, lineType=cv2.LINE_AA)
            if kps[2, i1] > kp_thresh:
                cv2.circle(
                    kp_mask, p1,
                    radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
            if kps[2, i2] > kp_thresh:
                cv2.circle(
                    kp_mask, p2,
                    radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

        # Blend the keypoints.
        return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

dbcfg = Dataset()


import os
import os.path as osp
import numpy as np
import argparse
from config import cfg
import cv2
import sys
import time
import json
from PIL import Image
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import math

import tensorflow as tf

from tfflat.base import Tester
from tfflat.utils import mem_info
from model import Model

from gen_batch import generate_batch
from dataset import Dataset
from nms.nms import oks_nms

def test_net(tester, input_pose, det_range, gpu_id):

    dump_results = []

    start_time = time.time()

    img_start = det_range[0]
    img_id = 0
    img_id2 = 0
    pbar = tqdm(total=det_range[1] - img_start - 1, position=gpu_id)
    pbar.set_description("GPU %s" % str(gpu_id))
    while img_start < det_range[1]:
        img_end = img_start + 1
        im_info = input_pose[img_start]
        while img_end < det_range[1] and input_pose[img_end]['image_id'] == im_info['image_id']:
            img_end += 1
        
        # all human detection results of a certain image
        cropped_data = input_pose[img_start:img_end]
        #pbar.set_description("GPU %s" % str(gpu_id))
        pbar.update(img_end - img_start)

        img_start = img_end

        kps_result = np.zeros((len(cropped_data), cfg.num_kps, 3))
        area_save = np.zeros(len(cropped_data))

        # cluster human detection results with test_batch_size
        for batch_id in range(0, len(cropped_data), cfg.test_batch_size):
            start_id = batch_id
            end_id = min(len(cropped_data), batch_id + cfg.test_batch_size)
             
            imgs = []
            input_pose_coords = []
            input_pose_valids = []
            input_pose_scores = []
            crop_infos = []
            for i in range(start_id, end_id):
                img, input_pose_coord, input_pose_valid, input_pose_score, crop_info = generate_batch(cropped_data[i], stage='test')
                imgs.append(img)
                input_pose_coords.append(input_pose_coord)
                input_pose_valids.append(input_pose_valid)
                input_pose_scores.append(input_pose_score)
                crop_infos.append(crop_info)
            imgs = np.array(imgs)
            input_pose_coords = np.array(input_pose_coords)
            input_pose_valids = np.array(input_pose_valids)
            input_pose_scores = np.array(input_pose_scores)
            crop_infos = np.array(crop_infos)
            
            # forward
            coord = tester.predict_one([imgs, input_pose_coords, input_pose_valids])[0]
            
            if cfg.flip_test:
                flip_imgs = imgs[:, :, ::-1, :]
                flip_input_pose_coords = input_pose_coords.copy()
                flip_input_pose_coords[:,:,0] = cfg.input_shape[1] - 1 - flip_input_pose_coords[:,:,0]
                flip_input_pose_valids = input_pose_valids.copy()
                for (q, w) in cfg.kps_symmetry:
                    flip_input_pose_coords_w, flip_input_pose_coords_q = flip_input_pose_coords[:,w,:].copy(), flip_input_pose_coords[:,q,:].copy()
                    flip_input_pose_coords[:,q,:], flip_input_pose_coords[:,w,:] = flip_input_pose_coords_w, flip_input_pose_coords_q
                    flip_input_pose_valids_w, flip_input_pose_valids_q = flip_input_pose_valids[:,w].copy(), flip_input_pose_valids[:,q].copy()
                    flip_input_pose_valids[:,q], flip_input_pose_valids[:,w] = flip_input_pose_valids_w, flip_input_pose_valids_q

                flip_coord = tester.predict_one([flip_imgs, flip_input_pose_coords, flip_input_pose_valids])[0]

                flip_coord[:,:,0] = cfg.input_shape[1] - 1 - flip_coord[:,:,0]
                for (q, w) in cfg.kps_symmetry:
                    flip_coord_w, flip_coord_q = flip_coord[:,w,:].copy(), flip_coord[:,q,:].copy()
                    flip_coord[:,q,:], flip_coord[:,w,:] = flip_coord_w, flip_coord_q
                coord += flip_coord
                coord /= 2
            
            # for each human detection from clustered batch
            for image_id in range(start_id, end_id):
               
                kps_result[image_id, :, :2] = coord[image_id - start_id]
                kps_result[image_id, :, 2] = input_pose_scores[image_id - start_id]

                vis=False
                crop_info = crop_infos[image_id - start_id,:]
                area = (crop_info[2] - crop_info[0]) * (crop_info[3] - crop_info[1])
                if vis and np.any(kps_result[image_id,:,2]) > 0.9 and area > 96**2:
                    tmpimg = imgs[image_id-start_id].copy()
                    tmpimg = cfg.denormalize_input(tmpimg)
                    tmpimg = tmpimg.astype('uint8')
                    tmpkps = np.zeros((3,cfg.num_kps))
                    tmpkps[:2,:] = kps_result[image_id,:,:2].transpose(1,0)
                    tmpkps[2,:] = kps_result[image_id,:,2]
                    _tmpimg = tmpimg.copy()
                    _tmpimg = cfg.vis_keypoints(_tmpimg, tmpkps)
                    cv2.imwrite(osp.join(cfg.vis_dir, str(img_id) + '_output.jpg'), _tmpimg)
                    img_id += 1

                # map back to original images
                for j in range(cfg.num_kps):
                    kps_result[image_id, j, 0] = kps_result[image_id, j, 0] / cfg.input_shape[1] * (\
                    crop_infos[image_id - start_id][2] - crop_infos[image_id - start_id][0]) + crop_infos[image_id - start_id][0]
                    kps_result[image_id, j, 1] = kps_result[image_id, j, 1] / cfg.input_shape[0] * (\
                    crop_infos[image_id - start_id][3] - crop_infos[image_id - start_id][1]) + crop_infos[image_id - start_id][1]
                
                area_save[image_id] = (crop_infos[image_id - start_id][2] - crop_infos[image_id - start_id][0]) * (crop_infos[image_id - start_id][3] - crop_infos[image_id - start_id][1])
                
        #vis
        vis = False
        if vis and np.any(kps_result[:,:,2] > 0.9):
            tmpimg = cv2.imread(os.path.join(cfg.img_path, cropped_data[0]['imgpath']))
            tmpimg = tmpimg.astype('uint8')
            for i in range(len(kps_result)):
                tmpkps = np.zeros((3,cfg.num_kps))
                tmpkps[:2,:] = kps_result[i, :, :2].transpose(1,0)
                tmpkps[2,:] = kps_result[i, :, 2]
                tmpimg = cfg.vis_keypoints(tmpimg, tmpkps)
            cv2.imwrite(osp.join(cfg.vis_dir, str(img_id2) + '.jpg'), tmpimg)
            img_id2 += 1
        
        # oks nms
        if cfg.dataset in ['COCO', 'PoseTrack']:
            nms_kps = np.delete(kps_result,cfg.ignore_kps,1)
            nms_score = np.mean(nms_kps[:,:,2],axis=1)
            nms_kps[:,:,2] = 1
            nms_kps = nms_kps.reshape(len(kps_result),-1)
            nms_sigmas = np.delete(cfg.kps_sigmas,cfg.ignore_kps)
            keep = oks_nms(nms_kps, nms_score, area_save, cfg.oks_nms_thr, nms_sigmas)
            if len(keep) > 0 :
                kps_result = kps_result[keep,:,:]
                area_save = area_save[keep]
 
        score_result = np.copy(kps_result[:, :, 2])
        kps_result[:, :, 2] = 1
        kps_result = kps_result.reshape(-1,cfg.num_kps*3)
       
        # save result
        for i in range(len(kps_result)):
            if cfg.dataset == 'COCO':
                result = dict(image_id=im_info['image_id'], category_id=1, score=float(round(np.mean(score_result[i]), 4)),
                             keypoints=kps_result[i].round(3).tolist())
            elif cfg.dataset == 'PoseTrack':
                result = dict(image_id=im_info['image_id'], category_id=1, track_id=0, scores=score_result[i].round(4).tolist(),
                              keypoints=kps_result[i].round(3).tolist())
            elif cfg.dataset == 'MPII':
                result = dict(image_id=im_info['image_id'], scores=score_result[i].round(4).tolist(),
                              keypoints=kps_result[i].round(3).tolist())

            dump_results.append(result)

    return dump_results


def test(test_model):
    
    # annotation load
    d = Dataset()
    annot = d.load_annot(cfg.testset)
    
    # input pose load
    input_pose = d.input_pose_load(annot, cfg.testset)

    # job assign (multi-gpu)
    from tfflat.mp_utils import MultiProc
    img_start = 0
    ranges = [0]
    img_num = len(np.unique([i['image_id'] for i in input_pose]))
    images_per_gpu = int(img_num / len(args.gpu_ids.split(','))) + 1
    for run_img in range(img_num):
        img_end = img_start + 1
        while img_end < len(input_pose) and input_pose[img_end]['image_id'] == input_pose[img_start]['image_id']:
            img_end += 1
        if (run_img + 1) % images_per_gpu == 0 or (run_img + 1) == img_num:
            ranges.append(img_end)
        img_start = img_end

    def func(gpu_id):
        cfg.set_args(args.gpu_ids.split(',')[gpu_id])
        tester = Tester(Model(), cfg)
        tester.load_weights(test_model)
        range = [ranges[gpu_id], ranges[gpu_id + 1]]
        return test_net(tester, input_pose, range, gpu_id)

    MultiGPUFunc = MultiProc(len(args.gpu_ids.split(',')), func)
    result = MultiGPUFunc.work()

    # evaluation
    d.evaluation(result, annot, cfg.result_dir, cfg.testset)

if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--gpu', type=str, dest='gpu_ids')
        parser.add_argument('--test_epoch', type=str, dest='test_epoch')
        args = parser.parse_args()

        # test gpus
        if not args.gpu_ids:
            args.gpu_ids = str(np.argmin(mem_info()))

        if '-' in args.gpu_ids:
            gpus = args.gpu_ids.split('-')
            gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
            gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
            args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
        
        assert args.test_epoch, 'Test epoch is required.'
        return args

    global args
    args = parse_args()
    test(int(args.test_epoch))

import os
import os.path as osp
import numpy as np
import cv2
from config import cfg
import random
import time
import math
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def synthesize_pose(joints, estimated_joints, near_joints, area, num_overlap):

    def get_dist_wrt_ks(ks, area):
        vars = (cfg.kps_sigmas * 2) ** 2
        return np.sqrt(-2 * area * vars * np.log(ks))
    ks_10_dist = get_dist_wrt_ks(0.10, area)
    ks_50_dist = get_dist_wrt_ks(0.50, area)
    ks_85_dist = get_dist_wrt_ks(0.85, area)
    
    synth_joints = joints.copy()
    for j in range(cfg.num_kps):
        # in case of not annotated joints, use other models`s result and add noise
        if joints[j,2] == 0:
            synth_joints[j] = estimated_joints[j]
    num_valid_joint = np.sum(joints[:,2] > 0)
    
    N = 500
    for j in range(cfg.num_kps):
        
        # PoseTrack does not have l_ear and r_ear. This will make input pose heatmap zero-gaussian heatmap.
        if j in cfg.ignore_kps:
            synth_joints[j] = 0
            continue

        # source keypoint position candidates to generate error on that (gt, swap, inv, swap+inv)
        coord_list = []
        # on top of gt
        gt_coord = np.expand_dims(synth_joints[j,:2],0)
        coord_list.append(gt_coord)
        # on top of swap gt
        swap_coord = near_joints[near_joints[:,j,2] > 0, j, :2]
        coord_list.append(swap_coord)
        # on top of inv gt, swap inv gt
        pair_exist = False
        for (q,w) in cfg.kps_symmetry:
            if j == q or j == w:
                if j == q:
                    pair_idx = w
                else:
                    pair_idx = q
                pair_exist = True
        if pair_exist and (joints[pair_idx,2] > 0):
            inv_coord = np.expand_dims(synth_joints[pair_idx,:2],0)
            coord_list.append(inv_coord)
        else:
            coord_list.append(np.empty([0,2]))

        if pair_exist:
            swap_inv_coord = near_joints[near_joints[:,pair_idx,2] > 0, pair_idx, :2]
            coord_list.append(swap_inv_coord)
        else:
            coord_list.append(np.empty([0,2]))

        tot_coord_list = np.concatenate(coord_list)
         
        assert len(coord_list) == 4

        # jitter error
        synth_jitter = np.zeros(3)
        if num_valid_joint <= 10:
            if j == 0 or (j >= 13 and j <= 16): # nose, ankle, knee
                jitter_prob = 0.15
            elif (j >= 1 and j <= 10): # ear, eye, upper body
                jitter_prob = 0.20
            else: # hip
                jitter_prob = 0.25
        else:
            if j == 0 or (j >= 13 and j <= 16): # nose, ankle, knee
                jitter_prob = 0.10
            elif (j >= 1 and j <= 10): # ear, eye, upper body
                jitter_prob = 0.15
            else: # hip
                jitter_prob = 0.20
        angle = np.random.uniform(0,2*math.pi,[N])
        r = np.random.uniform(ks_85_dist[j],ks_50_dist[j],[N]) 
        jitter_idx = 0 # gt
        x = tot_coord_list[jitter_idx][0] + r * np.cos(angle)
        y = tot_coord_list[jitter_idx][1] + r * np.sin(angle)
        dist_mask = True
        for i in range(len(tot_coord_list)):
            if i == jitter_idx:
                continue
            dist_mask = np.logical_and(dist_mask,np.sqrt((tot_coord_list[i][0] - x)**2 + (tot_coord_list[i][1] - y)**2) > r)
        x = x[dist_mask].reshape(-1)
        y = y[dist_mask].reshape(-1)
        if len(x) > 0:
            rand_idx = random.randrange(0, len(x))
            synth_jitter[0] = x[rand_idx]
            synth_jitter[1] = y[rand_idx]
            synth_jitter[2] = 1
      
        
        # miss error
        synth_miss = np.zeros(3)
        if num_valid_joint <= 5:
            if j >= 0 and j <= 4: # face
                miss_prob = 0.15
            elif j == 5 or j == 6 or j == 15 or j == 16: # shoulder, ankle
                miss_prob = 0.20
            else: # other parts
                miss_prob = 0.25
        elif num_valid_joint <= 10:
            if j >= 0 and j <= 4: # face
                miss_prob = 0.10
            elif j == 5 or j == 6 or j == 15 or j == 16: # shoulder, ankle
                miss_prob = 0.13
            else: # other parts
                miss_prob = 0.15
        else:
            if j >= 0 and j <= 4: # face
                miss_prob = 0.02
            elif j == 5 or j == 6 or j == 15 or j == 16: # shoulder, ankle
                miss_prob = 0.05
            else: # other parts
                miss_prob = 0.10
        
        miss_pt_list = []
        for miss_idx in range(len(tot_coord_list)):
            angle = np.random.uniform(0,2*math.pi,[4*N])
            r = np.random.uniform(ks_50_dist[j],ks_10_dist[j],[4*N])
            x = tot_coord_list[miss_idx][0] + r * np.cos(angle)
            y = tot_coord_list[miss_idx][1] + r * np.sin(angle)
            dist_mask = True
            for i in range(len(tot_coord_list)):
                if i == miss_idx:
                    continue
                dist_mask = np.logical_and(dist_mask,np.sqrt((tot_coord_list[i][0] - x)**2 + (tot_coord_list[i][1] - y)**2) > ks_50_dist[j])
            x = x[dist_mask].reshape(-1)
            y = y[dist_mask].reshape(-1)
            if len(x) > 0:
                if miss_idx == 0:
                    coord = np.transpose(np.vstack([x,y]),[1,0])
                    miss_pt_list.append(coord)
                else:
                    rand_idx = np.random.choice(range(len(x)), size=len(x)//4)
                    x = np.take(x,rand_idx)
                    y = np.take(y,rand_idx)
                    coord = np.transpose(np.vstack([x,y]),[1,0])
                    miss_pt_list.append(coord)
        if len(miss_pt_list) > 0:
            miss_pt_list = np.concatenate(miss_pt_list,axis=0).reshape(-1,2)
            rand_idx = random.randrange(0, len(miss_pt_list))
            synth_miss[0] = miss_pt_list[rand_idx][0]
            synth_miss[1] = miss_pt_list[rand_idx][1]
            synth_miss[2] = 1 

        
        # inversion prob
        synth_inv = np.zeros(3)
        if j <= 4: # face
            inv_prob = 0.01
        elif j >= 5 and j <= 10: # upper body
            inv_prob = 0.03
        else: # lower body
            inv_prob = 0.06
        if pair_exist and joints[pair_idx,2] > 0:
            angle = np.random.uniform(0,2*math.pi,[N])
            r = np.random.uniform(0,ks_50_dist[j],[N])
            inv_idx = (len(coord_list[0]) + len(coord_list[1]))
            x = tot_coord_list[inv_idx][0] + r * np.cos(angle)
            y = tot_coord_list[inv_idx][1] + r * np.sin(angle)
            dist_mask = True
            for i in range(len(tot_coord_list)):
                if i == inv_idx:
                    continue
                dist_mask = np.logical_and(dist_mask,np.sqrt((tot_coord_list[i][0] - x)**2 + (tot_coord_list[i][1] - y)**2) > r)
            x = x[dist_mask].reshape(-1)
            y = y[dist_mask].reshape(-1)
            if len(x) > 0:
                rand_idx = random.randrange(0, len(x))
                synth_inv[0] = x[rand_idx]
                synth_inv[1] = y[rand_idx]
                synth_inv[2] = 1


        # swap prob
        synth_swap = np.zeros(3)
        swap_exist = (len(coord_list[1]) > 0) or (len(coord_list[3]) > 0)
        if (num_valid_joint <= 10 and num_overlap > 0) or (num_valid_joint <= 15 and num_overlap >= 3):
            if j >= 0 and j <= 4: # face
                swap_prob = 0.02
            elif j >= 5 and j <= 10: # upper body
                swap_prob = 0.15
            else: # lower body
                swap_prob = 0.10
        else:
            if j >= 0 and j <= 4: # face
                swap_prob = 0.01
            elif j >= 5 and j <= 10: # upper body
                swap_prob = 0.06
            else: # lower body
                swap_prob = 0.03
        if swap_exist:

            swap_pt_list = []
            for swap_idx in range(len(tot_coord_list)):
                if swap_idx == 0 or swap_idx == len(coord_list[0]) + len(coord_list[1]):
                    continue
                angle = np.random.uniform(0,2*math.pi,[N])
                r = np.random.uniform(0,ks_50_dist[j],[N])
                x = tot_coord_list[swap_idx][0] + r * np.cos(angle)
                y = tot_coord_list[swap_idx][1] + r * np.sin(angle)
                dist_mask = True
                for i in range(len(tot_coord_list)):
                    if i == 0 or i == len(coord_list[0]) + len(coord_list[1]):
                        dist_mask = np.logical_and(dist_mask,np.sqrt((tot_coord_list[i][0] - x)**2 + (tot_coord_list[i][1] - y)**2) > r)
                x = x[dist_mask].reshape(-1)
                y = y[dist_mask].reshape(-1)
                if len(x) > 0:
                    coord = np.transpose(np.vstack([x,y]),[1,0])
                    swap_pt_list.append(coord)
            if len(swap_pt_list) > 0:
                swap_pt_list = np.concatenate(swap_pt_list,axis=0).reshape(-1,2)
                rand_idx = random.randrange(0, len(swap_pt_list))
                synth_swap[0] = swap_pt_list[rand_idx][0]
                synth_swap[1] = swap_pt_list[rand_idx][1]
                synth_swap[2] = 1 
                   

        # good prob
        synth_good = np.zeros(3)
        good_prob = 1 - (jitter_prob + miss_prob + inv_prob + swap_prob)
        assert good_prob >= 0
        angle = np.random.uniform(0,2*math.pi,[N//4])
        r = np.random.uniform(0,ks_85_dist[j],[N//4])
        good_idx = 0 # gt
        x = tot_coord_list[good_idx][0] + r * np.cos(angle)
        y = tot_coord_list[good_idx][1] + r * np.sin(angle)
        dist_mask = True
        for i in range(len(tot_coord_list)):
            if i == good_idx:
                continue
            dist_mask = np.logical_and(dist_mask,np.sqrt((tot_coord_list[i][0] - x)**2 + (tot_coord_list[i][1] - y)**2) > r)
        x = x[dist_mask].reshape(-1)
        y = y[dist_mask].reshape(-1)
        if len(x) > 0:
            rand_idx = random.randrange(0, len(x))
            synth_good[0] = x[rand_idx]
            synth_good[1] = y[rand_idx]
            synth_good[2] = 1 

        if synth_jitter[2] == 0:
            jitter_prob = 0
        if synth_inv[2] == 0:
            inv_prob = 0
        if synth_swap[2] == 0:
            swap_prob = 0
        if synth_miss[2] == 0:
            miss_prob = 0
        if synth_good[2] == 0:
            good_prob = 0

        normalizer = jitter_prob + miss_prob + inv_prob + swap_prob + good_prob
        if normalizer == 0:
            synth_joints[j] = 0
            continue

        jitter_prob = jitter_prob / normalizer
        miss_prob = miss_prob / normalizer
        inv_prob = inv_prob / normalizer
        swap_prob = swap_prob / normalizer
        good_prob = good_prob / normalizer
        
        prob_list = [jitter_prob, miss_prob, inv_prob, swap_prob, good_prob]
        synth_list = [synth_jitter, synth_miss, synth_inv, synth_swap, synth_good]
        sampled_idx = np.random.choice(5,1,p=prob_list)[0]
        synth_joints[j] = synth_list[sampled_idx]

        assert synth_joints[j,2] != 0

    return synth_joints

def generate_batch(d, stage='train'):
    
    img = cv2.imread(os.path.join(cfg.img_path, d['imgpath']), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        print('cannot read ' + os.path.join(cfg.img_path, d['imgpath']))
        assert 0

    bbox = np.array(d['bbox']).astype(np.float32)
    
    x, y, w, h = bbox
    aspect_ratio = cfg.input_shape[1]/cfg.input_shape[0]
    center = np.array([x + w * 0.5, y + h * 0.5])
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w,h]) * 1.25
    rotation = 0

    if stage == 'train':

        joints = np.array(d['joints']).reshape(-1,cfg.num_kps,3)
        estimated_joints = np.array(d['estimated_joints']).reshape(-1,cfg.num_kps,3)
        near_joints = np.array(d['near_joints']).reshape(-1,cfg.num_kps,3)
        total_joints = np.concatenate([joints, estimated_joints, near_joints], axis=0)

        # data augmentation
        scale = scale * np.clip(np.random.randn()*cfg.scale_factor + 1, 1-cfg.scale_factor, 1+cfg.scale_factor)
        rotation = np.clip(np.random.randn()*cfg.rotation_factor, -cfg.rotation_factor*2, cfg.rotation_factor*2)\
                if random.random() <= 0.6 else 0
        if random.random() <= 0.5:
            img = img[:, ::-1, :]
            center[0] = img.shape[1] - 1 - center[0]
            total_joints[:,:,0] = img.shape[1] - 1 - total_joints[:,:,0]
            for (q, w) in cfg.kps_symmetry:
                total_joints_q, total_joints_w = total_joints[:,q,:].copy(), total_joints[:,w,:].copy()
                total_joints[:,w,:], total_joints[:,q,:] = total_joints_q, total_joints_w

        trans = get_affine_transform(center, scale, rotation, (cfg.input_shape[1], cfg.input_shape[0]))
        cropped_img = cv2.warpAffine(img, trans, (cfg.input_shape[1], cfg.input_shape[0]), flags=cv2.INTER_LINEAR)
        #cropped_img = cropped_img[:,:, ::-1]
        cropped_img = cfg.normalize_input(cropped_img)
        
        for i in range(len(total_joints)):
            for j in range(cfg.num_kps):
                if total_joints[i,j,2] > 0:
                    total_joints[i,j,:2] = affine_transform(total_joints[i,j,:2], trans)
                    total_joints[i,j,2] *= ((total_joints[i,j,0] >= 0) & (total_joints[i,j,0] < cfg.input_shape[1]) & (total_joints[i,j,1] >= 0) & (total_joints[i,j,1] < cfg.input_shape[0]))
        joints = total_joints[0]
        estimated_joints = total_joints[1]
        near_joints = total_joints[2:]

        xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]
        pt1 = affine_transform(np.array([xmin, ymin]), trans)
        pt2 = affine_transform(np.array([xmax, ymin]), trans)
        pt3 = affine_transform(np.array([xmax, ymax]), trans)
        area = math.sqrt(pow(pt2[0] - pt1[0],2) + pow(pt2[1] - pt1[1],2)) * math.sqrt(pow(pt3[0] - pt2[0],2) + pow(pt3[1] - pt2[1],2))

        # input pose synthesize
        synth_joints = synthesize_pose(joints, estimated_joints, near_joints, area, d['overlap'])

        target_coord = joints[:,:2]
        target_valid = joints[:,2]
        input_pose_coord = synth_joints[:,:2]
        input_pose_valid = synth_joints[:,2]
        
        # for debug
        vis = False
        if vis:
            filename = str(random.randrange(1,500))
            tmpimg = cropped_img.astype(np.float32).copy()
            tmpimg = cfg.denormalize_input(tmpimg)
            tmpimg = tmpimg.astype(np.uint8).copy()
            tmpkps = np.zeros((3,cfg.num_kps))
            tmpkps[:2,:] = target_coord.transpose(1,0)
            tmpkps[2,:] = target_valid
            tmpimg = cfg.vis_keypoints(tmpimg, tmpkps)
            cv2.imwrite(osp.join(cfg.vis_dir, filename + '_gt.jpg'), tmpimg)
 
            tmpimg = cropped_img.astype(np.float32).copy()
            tmpimg = cfg.denormalize_input(tmpimg)
            tmpimg = tmpimg.astype(np.uint8).copy()
            tmpkps = np.zeros((3,cfg.num_kps))
            tmpkps[:2,:] = input_pose_coord.transpose(1,0)
            tmpkps[2,:] = input_pose_valid
            tmpimg = cfg.vis_keypoints(tmpimg, tmpkps)
            cv2.imwrite(osp.join(cfg.vis_dir, filename + '_input_pose.jpg'), tmpimg)
       
        return [cropped_img,
                target_coord, 
                input_pose_coord,
                (target_valid > 0),
                (input_pose_valid > 0)]

    else:
        trans = get_affine_transform(center, scale, rotation, (cfg.input_shape[1], cfg.input_shape[0]))
        cropped_img = cv2.warpAffine(img, trans, (cfg.input_shape[1], cfg.input_shape[0]), flags=cv2.INTER_LINEAR)
        #cropped_img = cropped_img[:,:, ::-1]
        cropped_img = cfg.normalize_input(cropped_img)

        estimated_joints = np.array(d['estimated_joints']).reshape(cfg.num_kps,3)
        for i in range(cfg.num_kps):
            if estimated_joints[i,2] > 0:
                estimated_joints[i,:2] = affine_transform(estimated_joints[i,:2], trans)
                estimated_joints[i,2] *= ((estimated_joints[i,0] >= 0) & (estimated_joints[i,0] < cfg.input_shape[1]) & (estimated_joints[i,1] >= 0) & (estimated_joints[i,1] < cfg.input_shape[0]))

        input_pose_coord = estimated_joints[:,:2]
        input_pose_valid = np.array([1 if i not in cfg.ignore_kps else 0 for i in range(cfg.num_kps)])
        input_pose_score = d['estimated_score']
        crop_info = np.asarray([center[0]-scale[0]*0.5, center[1]-scale[1]*0.5, center[0]+scale[0]*0.5, center[1]+scale[1]*0.5])

        return [cropped_img, input_pose_coord, input_pose_valid, input_pose_score, crop_info]



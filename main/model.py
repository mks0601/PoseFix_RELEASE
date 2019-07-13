import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import json
import math
from functools import partial

from config import cfg
from tfflat.base import ModelDesc

from nets.basemodel import resnet50, resnet101, resnet152, resnet_arg_scope, resnet_v1
resnet_arg_scope = partial(resnet_arg_scope, bn_trainable=cfg.bn_train)

class Model(ModelDesc):
    
    def head_net(self, blocks, is_training, trainable=True):
        
        normal_initializer = tf.truncated_normal_initializer(0, 0.01)
        msra_initializer = tf.contrib.layers.variance_scaling_initializer()
        xavier_initializer = tf.contrib.layers.xavier_initializer()
        
        with slim.arg_scope(resnet_arg_scope(bn_is_training=is_training)):
            
            out = slim.conv2d_transpose(blocks[-1], 256, [4, 4], stride=2,
                trainable=trainable, weights_initializer=normal_initializer,
                padding='SAME', activation_fn=tf.nn.relu,
                scope='up1')
            out = slim.conv2d_transpose(out, 256, [4, 4], stride=2,
                trainable=trainable, weights_initializer=normal_initializer,
                padding='SAME', activation_fn=tf.nn.relu,
                scope='up2')
            out = slim.conv2d_transpose(out, 256, [4, 4], stride=2,
                trainable=trainable, weights_initializer=normal_initializer,
                padding='SAME', activation_fn=tf.nn.relu,
                scope='up3')

            out = slim.conv2d(out, cfg.num_kps, [1, 1],
                    trainable=trainable, weights_initializer=msra_initializer,
                    padding='SAME', normalizer_fn=None, activation_fn=None,
                    scope='out')

        return out

    def extract_coordinate(self, heatmap_outs):
        shape = heatmap_outs.get_shape().as_list()
        batch_size = tf.shape(heatmap_outs)[0]
        height = shape[1]
        width = shape[2]
        output_shape = (height, width)
        
        # coordinate extract from output heatmap
        y = [i for i in range(output_shape[0])]
        x = [i for i in range(output_shape[1])]
        xx, yy = tf.meshgrid(x, y)
        xx = tf.to_float(xx) + 1
        yy = tf.to_float(yy) + 1
        
        heatmap_outs = tf.reshape(tf.transpose(heatmap_outs, [0, 3, 1, 2]), [batch_size, cfg.num_kps, -1])
        heatmap_outs = tf.nn.softmax(heatmap_outs)
        heatmap_outs = tf.transpose(tf.reshape(heatmap_outs, [batch_size, cfg.num_kps, output_shape[0], output_shape[1]]), [0, 2, 3, 1])

        x_out = tf.reduce_sum(tf.multiply(heatmap_outs, tf.tile(tf.reshape(xx,[1, output_shape[0], output_shape[1], 1]), [batch_size, 1, 1, cfg.num_kps])), [1,2])
        y_out = tf.reduce_sum(tf.multiply(heatmap_outs, tf.tile(tf.reshape(yy,[1, output_shape[0], output_shape[1], 1]), [batch_size, 1, 1, cfg.num_kps])), [1,2])
        coord_out = tf.concat([tf.reshape(x_out, [batch_size, cfg.num_kps, 1])\
            ,tf.reshape(y_out, [batch_size, cfg.num_kps, 1])]\
                    , axis=2)
        coord_out = coord_out - 1

        coord_out = coord_out / output_shape[0] * cfg.input_shape[0]

        return coord_out
 
    def render_onehot_heatmap(self, coord, output_shape):
        
        batch_size = tf.shape(coord)[0]

        x = tf.reshape(coord[:,:,0] / cfg.input_shape[1] * output_shape[1],[-1])
        y = tf.reshape(coord[:,:,1] / cfg.input_shape[0] * output_shape[0],[-1])
        x_floor = tf.floor(x)
        y_floor = tf.floor(y)

        x_floor = tf.clip_by_value(x_floor, 0, output_shape[1] - 1)  # fix out-of-bounds x
        y_floor = tf.clip_by_value(y_floor, 0, output_shape[0] - 1)  # fix out-of-bounds y

        indices_batch = tf.expand_dims(tf.to_float(\
                tf.reshape(
                tf.transpose(\
                tf.tile(\
                tf.expand_dims(tf.range(batch_size),0)\
                ,[cfg.num_kps,1])\
                ,[1,0])\
                ,[-1])),1)
        indices_batch = tf.concat([indices_batch, indices_batch, indices_batch, indices_batch], axis=0)
        indices_joint = tf.to_float(tf.expand_dims(tf.tile(tf.range(cfg.num_kps),[batch_size]),1))
        indices_joint = tf.concat([indices_joint, indices_joint, indices_joint, indices_joint], axis=0)
        
        indices_lt = tf.concat([tf.expand_dims(y_floor,1), tf.expand_dims(x_floor,1)], axis=1)
        indices_lb = tf.concat([tf.expand_dims(y_floor+1,1), tf.expand_dims(x_floor,1)], axis=1)
        indices_rt = tf.concat([tf.expand_dims(y_floor,1), tf.expand_dims(x_floor+1,1)], axis=1)
        indices_rb = tf.concat([tf.expand_dims(y_floor+1,1), tf.expand_dims(x_floor+1,1)], axis=1)

        indices = tf.concat([indices_lt, indices_lb, indices_rt, indices_rb], axis=0)
        indices = tf.cast(tf.concat([indices_batch, indices, indices_joint], axis=1),tf.int32)

        prob_lt = (1 - (x - x_floor)) * (1 - (y - y_floor))
        prob_lb = (1 - (x - x_floor)) * (y - y_floor)
        prob_rt = (x - x_floor) * (1 - (y - y_floor))
        prob_rb = (x - x_floor) * (y - y_floor)
        probs = tf.concat([prob_lt, prob_lb, prob_rt, prob_rb], axis=0)

        heatmap = tf.scatter_nd(indices, probs, (batch_size, *output_shape, cfg.num_kps))
        normalizer = tf.reshape(tf.reduce_sum(heatmap,axis=[1,2]),[batch_size,1,1,cfg.num_kps])
        normalizer = tf.where(tf.equal(normalizer,0),tf.ones_like(normalizer),normalizer)
        heatmap = heatmap / normalizer
        
        return heatmap 
  
    def render_gaussian_heatmap(self, coord, output_shape, sigma, valid=None):
        
        x = [i for i in range(output_shape[1])]
        y = [i for i in range(output_shape[0])]
        xx,yy = tf.meshgrid(x,y)
        xx = tf.reshape(tf.to_float(xx), (1,*output_shape,1))
        yy = tf.reshape(tf.to_float(yy), (1,*output_shape,1))
              
        x = tf.reshape(coord[:,:,0],[-1,1,1,cfg.num_kps]) / cfg.input_shape[1] * output_shape[1]
        y = tf.reshape(coord[:,:,1],[-1,1,1,cfg.num_kps]) / cfg.input_shape[0] * output_shape[0]

        heatmap = tf.exp(-(((xx-x)/tf.to_float(sigma))**2)/tf.to_float(2) -(((yy-y)/tf.to_float(sigma))**2)/tf.to_float(2))

        if valid is not None:
            valid_mask = tf.reshape(valid, [-1, 1, 1, cfg.num_kps])
            heatmap = heatmap * valid_mask

        return heatmap * 255.
   
    def make_network(self, is_train):
        if is_train:
            image = tf.placeholder(tf.float32, shape=[cfg.batch_size, *cfg.input_shape, 3])
            target_coord = tf.placeholder(tf.float32, shape=[cfg.batch_size, cfg.num_kps, 2])
            input_pose_coord = tf.placeholder(tf.float32, shape=[cfg.batch_size, cfg.num_kps, 2])
            target_valid = tf.placeholder(tf.float32, shape=[cfg.batch_size, cfg.num_kps])
            input_pose_valid = tf.placeholder(tf.float32, shape=[cfg.batch_size, cfg.num_kps])
            self.set_inputs(image, target_coord, input_pose_coord, target_valid, input_pose_valid)
        else:
            image = tf.placeholder(tf.float32, shape=[None, *cfg.input_shape, 3])
            input_pose_coord = tf.placeholder(tf.float32, shape=[None, cfg.num_kps, 2])
            input_pose_valid = tf.placeholder(tf.float32, shape=[None, cfg.num_kps])
            self.set_inputs(image, input_pose_coord, input_pose_valid)

        input_pose_hm = tf.stop_gradient(self.render_gaussian_heatmap(input_pose_coord, cfg.input_shape, cfg.input_sigma, input_pose_valid))
        backbone = eval(cfg.backbone)
        resnet_fms = backbone([image, input_pose_hm], is_train, bn_trainable=True)
        heatmap_outs = self.head_net(resnet_fms, is_train)
        
        if is_train:
            
            gt_heatmap = tf.stop_gradient(tf.reshape(tf.transpose(\
                    self.render_onehot_heatmap(target_coord, cfg.output_shape),\
                    [0, 3, 1, 2]), [cfg.batch_size, cfg.num_kps, -1]))
            gt_coord = target_coord / cfg.input_shape[0] * cfg.output_shape[0]

            # heatmap loss
            out = tf.reshape(tf.transpose(heatmap_outs, [0, 3, 1, 2]), [cfg.batch_size, cfg.num_kps, -1])
            gt = gt_heatmap
            valid_mask = tf.reshape(target_valid, [cfg.batch_size, cfg.num_kps])
            loss_heatmap = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=gt, logits=out) * valid_mask)

            # coordinate loss
            out = self.extract_coordinate(heatmap_outs) / cfg.input_shape[0] * cfg.output_shape[0]
            gt = gt_coord
            valid_mask = tf.reshape(target_valid, [cfg.batch_size, cfg.num_kps, 1])
            loss_coord = tf.reduce_mean(tf.abs(out - gt) * valid_mask)

            loss = loss_heatmap + loss_coord

            self.add_tower_summary('loss_h', loss_heatmap)
            self.add_tower_summary('loss_c', loss_coord)

            self.set_loss(loss)
            
        else:
            out = self.extract_coordinate(heatmap_outs)
            self.set_outputs(out)




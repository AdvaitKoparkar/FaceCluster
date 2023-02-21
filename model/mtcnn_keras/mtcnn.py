# TODO implement MTCNN module using https://github.com/xiangrufan/keras-mtcnn
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras.layers import (Conv2D, Dense, Flatten, Input, MaxPool2D,
                                     Permute, PReLU)

import cv2
import matplotlib.pyplot as plt

def _rect2square(rectangles):
    w = rectangles[:,2] - rectangles[:,0]
    h = rectangles[:,3] - rectangles[:,1]
    l = np.maximum(w,h).T
    rectangles[:,0] = rectangles[:,0] + w*0.5 - l*0.5
    rectangles[:,1] = rectangles[:,1] + h*0.5 - l*0.5 
    rectangles[:,2:4] = rectangles[:,0:2] + np.repeat([l], 2, axis = 0).T 
    return rectangles

class MTCNN(object):
    def __init__(self, config):
        self.config = config
        self._load_PNet()
        self._load_ONet()
        self._load_RNet()

    def _load_ONet(self):
        input = Input(shape=[48,48,3])
        x = Conv2D(32, (3,3), strides=1, padding='valid', name='conv1')(input)
        x = PReLU(shared_axes=[1,2], name='prelu1')(x)
        x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

        x = Conv2D(64, (3,3), strides=1, padding='valid', name='conv2')(x)
        x = PReLU(shared_axes=[1,2], name='prelu2')(x)
        x = MaxPool2D(pool_size=3, strides=2)(x)

        x = Conv2D(64, (3,3), strides=1, padding='valid', name='conv3')(x)
        x = PReLU(shared_axes=[1,2], name='prelu3')(x)
        x = MaxPool2D(pool_size=2)(x)

        x = Conv2D(128, (2, 2), strides=1, padding='valid', name='conv4')(x)
        x = PReLU(shared_axes=[1,2],name='prelu4')(x)
        x = Permute((3,2,1))(x)
        x = Flatten()(x)

        x = Dense(256, name='conv5') (x)
        x = PReLU(name='prelu5')(x)

        classifier = Dense(2, activation='softmax',name='conv6-1')(x)
        bbox_regress = Dense(4,name='conv6-2')(x)
        landmark_regress = Dense(10,name='conv6-3')(x)

        self.ONet = tf.keras.models.Model([input], [classifier, bbox_regress, landmark_regress])
        self.ONet.load_weights(self.config['ONet_weights'])

    def _load_PNet(self):
        input = Input(shape=[None, None, 3])
        x = Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(input)
        x = PReLU(shared_axes=[1,2],name='PReLU1')(x)
        x = MaxPool2D(pool_size=2)(x)

        x = Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x)
        x = PReLU(shared_axes=[1,2],name='PReLU2')(x)

        x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
        x = PReLU(shared_axes=[1,2],name='PReLU3')(x)

        classifier = Conv2D(2, (1, 1), activation='softmax', name='conv4-1')(x)
        bbox_regress = Conv2D(4, (1, 1), name='conv4-2')(x)
        self.PNet = tf.keras.models.Model([input], [classifier, bbox_regress])
        self.PNet.load_weights(self.config['PNet_weights'], by_name=True)

    def _load_RNet(self):
        input = Input(shape=[24, 24, 3])
        x = Conv2D(28, (3, 3), strides=1, padding='valid', name='conv1')(input)
        x = PReLU(shared_axes=[1, 2], name='prelu1')(x)
        x = MaxPool2D(pool_size=3,strides=2, padding='same')(x)

        x = Conv2D(48, (3, 3), strides=1, padding='valid', name='conv2')(x)
        x = PReLU(shared_axes=[1, 2], name='prelu2')(x)
        x = MaxPool2D(pool_size=3, strides=2)(x)

        x = Conv2D(64, (2, 2), strides=1, padding='valid', name='conv3')(x)
        x = PReLU(shared_axes=[1, 2], name='prelu3')(x)
        x = Permute((3, 2, 1))(x)
        x = Flatten()(x)

        x = Dense(128, name='conv4')(x)
        x = PReLU( name='prelu4')(x)

        classifier = Dense(2, activation='softmax', name='conv5-1')(x)
        bbox_regress = Dense(4, name='conv5-2')(x)
        self.RNet = Model([input], [classifier, bbox_regress])
        self.RNet.load_weights(self.config['RNet_weights'], by_name=True)

    def _visualize(self, img, coords_map, scores):
        fig, ax = plt.subplots(1,1)
        
        for prop_idx in range(len(coords_map)):
            r0, c0 = int(coords_map[prop_idx,0]), int(coords_map[prop_idx,1])
            r1, c1 = int(coords_map[prop_idx,2]), int(coords_map[prop_idx,3])
            score  = scores[prop_idx]

            img = cv2.rectangle(img, (c0, r0), (c1, r1), [0,255,0], 2)
            img = cv2.putText(img, f"{score:.02f}", (c0,r0), 1, 2, [0,255,0])

        ax.imshow(img)
        plt.show()

    def detect_faces(self, img):
        scales = self._generate_scales(img)
        import pdb; pdb.set_trace()
        pnet_output = self._generate_proposals_PNET(img, scales)
        self._visualize(img, *pnet_output)
        # rnet_output = self._refine_proposals_RNET(img, pnet_output)

    def _generate_scales(self, img):
        height, width, channels = img.shape
        
        # get base scale
        if min(height, width) > self.config["image_scaling_config"]["max_dim"]:
            scale = self.config["image_scaling_config"]["max_dim"] / min(height, width)
            width, height = scale*width, scale*height
        elif max(height, width) < 500:
            scale = self.config["image_scaling_config"]["max_dim"] / max(height, width)
            width, height = scale*width, scale*height
        else:
            scale = 1.0

        # calculate scales
        scales = []
        factor_count = 0
        while min(height, width) >= self.config["image_scaling_config"]["min_dim"]:
            scale = scale*pow(self.config["image_scaling_config"]["recursive_scaling_factor"], factor_count)
            scales.append(scale)
            width, height = scale*width, scale*height
            factor_count += 1

        return scales

    def _generate_proposals_PNET(self, img, scales):
        coords_map_all_scales = []
        face_probs_all_scales  = []
        origin_h, origin_w, _ = img.shape
        
        for scale in scales:
            hs = int(origin_h * scale)
            ws = int(origin_w * scale)
            scale_img = self.config["image_resizing_fn"](image=img, output_shape=(hs, ws))
            input = scale_img.reshape(1, *scale_img.shape)
            output = self.PNet.predict(input)
            face_prob_map = output[0][0,:,:,1]
            coords_map = output[1][0, :, :, :]

            coords_map, face_probs = self._PNet_postprocess(face_prob_map, coords_map, scale, (origin_h, origin_w))
            if coords_map.shape[0] == 0:
                continue

            coords_map_all_scales.append(coords_map)
            face_probs_all_scales.append(face_probs)
        
        coords_map_all_scales = np.concatenate(coords_map_all_scales)
        face_probs_all_scales = np.concatenate(face_probs_all_scales)
        coords_map_all_scales, face_probs_all_scales = self._NMS(coords_map_all_scales,
                                                                 face_probs_all_scales,
                                                                 threshold=0.7, # replace with config
                                                                 )
        
        return coords_map_all_scales, face_probs_all_scales

    def _PNet_postprocess(self, face_prob_map, coords_map, scale, origin_dim):
        # threshold for PNet class map
        threshold = 0.2 # replace with config

        # 12x12 input == PNet ==> 1x1 output
        scale_PNet = 12 # replace with config
        
        # output dim of PNet
        out_dim = max(face_prob_map.shape)

        # estimate inputdim based on output
        in_dim  = 2*out_dim + scale_PNet - 1

        # move in pixel in output = move "stride" pixels on input
        stride = 0
        if out_dim > 0: stride = float(in_dim-scale_PNet) / (out_dim-1)

        # get bounding box from class_prob_map and coords_map
        rc = np.array(np.where(face_prob_map > threshold)).T
        face_locs = np.tile(rc, reps=(1,2)).astype(np.float32)
        # map face locs into original image
        face_locs[:,0:2] = np.fix(stride * (face_locs[:,0:2])                * 1/scale).astype(np.float32)
        face_locs[:,2:4] = np.fix(stride * (face_locs[:,2:4] + scale_PNet-1) * 1/scale).astype(np.float32)

        # add coords_map info to bounding boxes
        scores = np.zeros((rc.shape[0],))
        for rc_idx in range(rc.shape[0]):
            r, c = rc[rc_idx, :]
            dc1, dr1, dc2, dr2 = coords_map[r, c, :]
            face_locs[rc_idx, 0:2] = face_locs[rc_idx, 0:2] + (np.array([dr1,dc1]) * scale_PNet * 1/scale)
            face_locs[rc_idx, 2:4] = face_locs[rc_idx, 2:4] + (np.array([dr2,dc2]) * scale_PNet * 1/scale)
            scores[rc_idx] = face_prob_map[r,c]

        # convert to squares        
        face_locs = _rect2square(face_locs)
        
        # non-maximum supression
        face_locs, scores = self._NMS(face_locs, scores, **self.config["PNet_prediction_NMS_config"])

        return face_locs, scores

    def _refine_proposals_RNET(self, img, pnet_output):
        coords_map, face_probs = pnet_output
        num_proposals = len(coords_map)

        rnet_inputs = []
        for proposal_idx in range(num_proposals):
            coords_map_idx = coords_map[proposal_idx]
            cropped_img    = img[coords_map_idx[0]:coords_map_idx[1], coords_map_idx[2]:coords_map_idx[3]]
            cropped_img    = self.config["image_resizing_fn"](image=img, output_shape=(24, 24)) # replace with config
            rnet_inputs.append(cropped_img)

    def _NMS(self, boxes, scores, threshold=0.6):
        # if there are no boxes / scores, return an empty list
        if scores.shape[0] == 0:
            return boxes, scores
    
        # initialize the list of picked indexes 
        pick = []
    
        # grab the coordinates of the bounding boxes
        y1 = boxes[:,0]
        x1 = boxes[:,1]
        y2 = boxes[:,2]
        x2 = boxes[:,3]
    
        # compute the area of the bounding boxes
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
    
        # sort the bounding boxes by scores in descending order
        idxs = np.argsort(scores)[::-1]
    
        # keep looping while some indexes still remain in the indexes
        # list
        while len(idxs) > 0:
            # grab the current index
            i = idxs[0]
            pick.append(i)
            suppress = [0]
    
            # loop over all indexes in the indexes list
            for pos in range(1, len(idxs)):
                # grab the current index
                j = idxs[pos]
    
                # find the largest (y, x) coordinates for the start of
                # the bounding box and the smallest (y, x) coordinates
                # for the end of the bounding box
                yy1 = max(y1[i], y1[j])
                xx1 = max(x1[i], x1[j])
                yy2 = min(y2[i], y2[j])
                xx2 = min(x2[i], x2[j])
    
                # compute the width and height of the bounding box
                h = max(0, yy2 - yy1 + 1)
                w = max(0, xx2 - xx1 + 1)
    
                # compute the ratio of overlap between the computed
                # bounding box and the bounding box in the area list
                overlap = float(w * h) / area[j]
    
                # if there is sufficient overlap, suppress the
                # current bounding box
                if overlap > threshold:
                    suppress.append(pos)
    
            # delete all indexes from the index list that are in the
            # suppression list
            idxs = np.delete(idxs, suppress)
    
        # return only the bounding boxes that were picked
        return boxes[pick], scores[pick]




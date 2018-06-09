#!/usr/bin/env python
# Author: Leo Zhang

import rospy
import numpy as np
import tensorflow as tf
from keras.models import model_from_json
from keras.models import load_model
import os
import math
from micro_doppler_pkg.msg import MicroDoppler
from mds_cnn_node_pkg.msg import MDSPred
from mds_dict import mds_dict


class mds_cnn_node:
    def __init__(self):
        self.frame_id = 'mds_pred'
        self.directory = '/home/leo/workspace/mds_cnn_node_ws/src/mds_cnn_node_pkg/model/'
        
        self.loaded_model = load_model(self.directory+'mds_cnn_model.h5')
        self.loaded_model._make_predict_function()
        self.graph = tf.get_default_graph()

        self.nd = 128
        self.time_domain_bins = 25
        print('Loaded model from disk.')

    def main(self):
        rospy.init_node('mds_cnn_node')
        self.sub_ = rospy.Subscriber('/ti_mmwave/micro_doppler', MicroDoppler, self.mds_predictor)
        self.pub_ = rospy.Publisher('/ti_mmwave/mds_pred', MDSPred, queue_size=100)
        rospy.spin()

    def pred_state(self):
        state = 'Transition...'
        arr = self.pred.flatten()
        tmp = np.argmax(arr)
        if arr[tmp] > .8:
            state = str(mds_dict[tmp])
        return state

    def mds_predictor(self, mds):
        tmp = np.array(mds.micro_doppler_array)
        if max(tmp) != 0:
            tmp = tmp / max(tmp)
        test_data = tmp.reshape(-1, self.nd, self.time_domain_bins, 1)

        with self.graph.as_default():
            self.pred = self.loaded_model.predict(test_data)
            
        cur_state = self.pred_state()
        mds_pred_msg = MDSPred()
        mds_pred_msg.header.frame_id = self.frame_id
        mds_pred_msg.header.stamp = rospy.Time.now()
        mds_pred_msg.mds_pred_array = self.pred.flatten().tolist()
        mds_pred_msg.prediction = cur_state
        self.pub_.publish(mds_pred_msg)

if __name__ == '__main__':
    mds_cnn_node().main()
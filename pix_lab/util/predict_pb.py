from __future__ import print_function, division

import time

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from scipy import misc
from pix_lab.util.util import load_graph

class Predict_pb(object):
    """
        Perform inference for an arunet instance

        :param net: the arunet instance to train

        """
    def __init__(self, path_to_pb, image_paths, output_paths, scale=0.33, mode='L'):
        self.graph = load_graph(path_to_pb)
        self.img_paths = image_paths
        self.output_paths = output_paths
        self.scale = scale
        self.mode = mode

    def predict(self, gpu_device="0"):
        val_size = len(self.img_paths)

        session_conf = tf.ConfigProto()
        session_conf.gpu_options.visible_device_list = gpu_device

        with tf.Session(graph=self.graph, config=session_conf) as sess:
            x = self.graph.get_tensor_by_name('inImg:0')
            predictor = self.graph.get_tensor_by_name('output:0')

            for step in range(0, val_size):
                aImgPath = self.img_paths[step]

                batch_x = self.load_img(aImgPath, self.scale, self.mode)
                # h: batch_x.shape[1], w: batch_x.shape[2]

                # Run validation
                aPred = sess.run(predictor,
                                 feed_dict={x: batch_x})

                np.save(self.output_paths[step], aPred)

            return None

    def load_img(self, path, scale, mode):
        aImg = misc.imread(path, mode=mode)
        sImg = misc.imresize(aImg, scale, interp='bicubic')
        fImg = sImg
        if len(sImg.shape) == 2:
            fImg = np.expand_dims(fImg, 2)
        fImg = np.expand_dims(fImg, 0)

        return fImg

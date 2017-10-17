import _init_paths
from utils import to_categorical_4d, to_categorical_4d_reverse
from FCN2 import FCN8
from test import *
import tensorflow as tf
import numpy as np
import ear_pen
import math
import cv2

model_store_path = '../model/FCN-8/FCN-8.ckpt'

if __name__ == '__main__':
    img_ph = tf.placeholder(tf.float32, [None, 104, 78, 3])
    ann_ph = tf.placeholder(tf.int32, [None, 104, 78, 1])
    net = FCN8(img_ph, ann_ph)
    work(img_ph, ann_ph, net, model_store_path)
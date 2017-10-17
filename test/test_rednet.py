import _init_paths
from utils import to_categorical_4d, to_categorical_4d_reverse
from RedNet import RedNet
from test import *
import tensorflow as tf
import numpy as np
import ear_pen
import math

model_store_path = '../model/RedNet/RedNet.ckpt'

if __name__ == '__main__':
    img_ph = tf.placeholder(tf.float32, [None, 104, 78, 3])
    ann_ph = tf.placeholder(tf.int32, [None, 104, 78, 1])
    net = RedNet(img_ph, ann_ph)
    work(img_ph, ann_ph, net, model_store_path)
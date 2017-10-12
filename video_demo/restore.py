import _init_paths
from utils import to_categorical_4d, to_categorical_4d_reverse
from FCN import FCN8
from draw import drawRec
import tensorflow as tf
import numpy as np
import ear_pen
import math

batch_size = 1
model_store_path = '../model/FCN-8/FCN-8.ckpt'
video_name = 'move.mp4'

if __name__ == '__main__':
    img_ph = tf.placeholder(tf.float32, [None, 104, 78, 3])
    ann_ph = tf.placeholder(tf.int32, [None, 104, 78, 1])
    net = FCN8(img_ph, ann_ph)

    # Load data
    (train_img, train_ann), (test_img, test_ann) = ear_pen.load_data()
    train_ann = np.asarray(train_ann) / 255
    train_ann, _map = to_categorical_4d(train_ann)

    # Work
    saver = tf.train.Saver()
    loss_list = []
    with tf.Session() as sess:
        # Recover model
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, model_store_path)
        print('restore done...')
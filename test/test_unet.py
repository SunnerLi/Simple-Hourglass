import _init_paths
from utils import to_categorical_4d, to_categorical_4d_reverse
from UNet import UNet
import tensorflow as tf
import numpy as np
import ear_pen
import math

batch_size = 1
model_store_path = '../model/UNet/UNet.ckpt'

if __name__ == '__main__':
    img_ph = tf.placeholder(tf.float32, [None, 104, 78, 3])
    ann_ph = tf.placeholder(tf.int32, [None, 104, 78, 1])
    net = UNet(img_ph, ann_ph)

    # Load data
    (train_img, train_ann), (test_img, test_ann) = ear_pen.load_data()
    test_ann = np.asarray(test_ann) / 255
    test_ann, _map = to_categorical_4d(test_ann)

    # Test
    saver = tf.train.Saver()
    loss_list = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, model_store_path)
        loss_sum = 0
        for i in range(math.ceil(len(test_img) / batch_size)):
            feed_dict = {
                img_ph: test_img[i*batch_size: i*batch_size+batch_size],
                ann_ph: test_ann[i*batch_size: i*batch_size+batch_size] 
            }
            loss_sum += sess.run([net.loss], feed_dict=feed_dict)[0]
        loss_list.append(loss_sum)
        print('test loss: ', loss_sum)
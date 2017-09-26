from utils import to_categorical_4d
from model import FCN8
import tensorflow as tf
import numpy as np
import ear_pen

if __name__ == '__main__':
    (train_img, train_ann), (test_img, test_ann) = ear_pen.load_data()
    train_ann = np.asarray(train_ann) / 256
    train_ann = to_categorical_4d(train_ann)

    img_ph = tf.placeholder(tf.float32, [2, 1040, 780, 3])
    ann_ph = tf.placeholder(tf.float32, [2, 1040, 780, 3])
    net = FCN8(img_ph, ann_ph)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1):
            feed_dict = {
                net.img_ph: train_img[0:2],
                net.ann_ph: train_ann[0:2]
            }
            _loss, _ = sess.run([net.loss, net.optimize], feed_dict=feed_dict)
            print('iter: ', i, '\tloss: ', _loss)
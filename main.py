from utils import to_categorical_4d, to_categorical_4d_reverse
from matplotlib import pyplot as plt
from skimage import io
from model import FCN8
import tensorflow as tf
import numpy as np
import ear_pen

if __name__ == '__main__':
    (train_img, train_ann), (test_img, test_ann) = ear_pen.load_data()
    train_ann = np.asarray(train_ann) / 255
    print('max: ', np.max(train_ann))
    train_ann, _map = to_categorical_4d(train_ann)
    print(_map)    

    img_ph = tf.placeholder(tf.float32, [None, 1040, 780, 3])
    ann_ph = tf.placeholder(tf.float32, [None, 1040, 780, 3])
    net = FCN8(img_ph, ann_ph)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            feed_dict = {
                img_ph: train_img[0:2],
                ann_ph: train_ann[0:2]
            }
            _loss, _, _img = sess.run([net.loss, net.train_op, net.predict], feed_dict=feed_dict)
            # _img = np.asarray(to_categorical_4d_reverse(_img, _map)[0, :, :, :], dtype=float)
            _img = np.asarray(to_categorical_4d_reverse(_img, _map)[0, :, :, :] * 255, dtype=int)

            
            if i % 10 == 0:
                io.imsave(str(i)+'.png', _img)
                print('iter: ', i, '\tloss: ', _loss)                
            """
            io.imsave(str(i)+'.png', _img)
            print('iter: ', i, '\tloss: ', _loss)
            """

            # print(_img[-5:, -5:, :])
import _init_paths
from utils import to_categorical_4d, to_categorical_4d_reverse
import tensorflow as tf
import numpy as np
import ear_pen
import math
import cv2

batch_size = 2

def showImg(test_imgs, test_annos, _map, prediction=None):
    """
        test_imgs       - origin
        test_annos      - without reverse
        prediction      - without reverse
    """
    test_annos = np.asarray(to_categorical_4d_reverse(test_annos, _map), dtype=np.uint8)
    prediction = np.asarray(to_categorical_4d_reverse(prediction, _map), dtype=np.uint8)
    show_img = None
    for i in range(len(test_imgs)):
        if i == 0:
            show_img = np.concatenate((test_imgs[i], test_annos[i], prediction[i]), axis=1)
        else:
            _ = np.concatenate((test_imgs[i], test_annos[i], prediction[i]), axis=1)
            show_img = np.concatenate((show_img, _), axis=0)
    show_img = cv2.cvtColor(show_img, cv2.COLOR_BGR2RGB)
    cv2.imshow('predict result', show_img)
    cv2.waitKey(0)
    cv2.imwrite('test.png', show_img)

def work(img_ph, ann_ph, net, model_path):
    # Load data
    (train_img, train_ann), (test_img, test_ann) = ear_pen.load_data()
    test_ann, _map = to_categorical_4d(test_ann)

    # Test
    saver = tf.train.Saver()
    loss_list = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        loss_sum = 0
        for i in range(math.ceil(len(test_img) / batch_size)):
            feed_dict = {
                img_ph: test_img[i*batch_size: i*batch_size+batch_size],
                ann_ph: test_ann[i*batch_size: i*batch_size+batch_size] 
            }
            _loss, _pred= sess.run([net.loss, net.prediction], feed_dict=feed_dict)
            loss_sum += _loss
        loss_list.append(loss_sum)
        print('test loss: ', loss_sum)
    showImg(test_img, test_ann, _map, _pred)
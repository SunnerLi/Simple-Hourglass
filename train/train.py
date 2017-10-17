import _init_paths
from utils import to_categorical_4d, to_categorical_4d_reverse
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import imageio
import ear_pen
import math

epoch = 500
save_period = 20
batch_size = 32
model_store_path = '../model/'

def statistic(x, y, title=''):
    plt.plot(x, y, linestyle='-')
    plt.plot(x, y, 'o')
    plt.title(title)
    plt.savefig('result.png')
    # plt.show()

def train(net, img_ph, ann_ph, title):
    # Load data
    (train_img, train_ann), (test_img, test_ann) = ear_pen.load_data()
    train_ann, _map = to_categorical_4d(train_ann)
    print('map: ', _map)

    # Train
    saver = tf.train.Saver()
    loss_list = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            loss_sum = 0
            for j in range(math.ceil(len(train_img) / batch_size)):
                feed_dict = {
                    img_ph: train_img[j*batch_size: j*batch_size+batch_size],
                    ann_ph: train_ann[j*batch_size: j*batch_size+batch_size] 
                }
                _loss, _, _img = sess.run([net.loss, net.train_op, net.prediction], feed_dict=feed_dict)
                loss_sum += _loss
            _logits = np.asarray(to_categorical_4d_reverse(_img, _map)[0, :, :, :], dtype=np.uint8) 
            _labels = np.asarray(to_categorical_4d_reverse(train_ann, _map)[0, :, :, :], dtype=np.uint8) 
            _img = np.concatenate((_logits, _labels), axis=1)
            print(np.shape(_img))
            if i % save_period == 0:
                imageio.imsave(str(i)+'.png', _img)
                loss_list.append(loss_sum)
            print('iter: ', i, '\tloss: ', loss_sum)
        
        # Store train result
        model_name = title[8:]
        saver.save(sess, model_store_path + model_name + '/' + model_name + '.ckpt')
    loss_list[0] = loss_list[-1]
    statistic(range(int(epoch / save_period)), np.log(loss_list), title)

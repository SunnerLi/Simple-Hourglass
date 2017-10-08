import tensorflow as tf
import layers

class FCN(object):
    def __init__(self):
        self.network = {}

    def vgg_part(self, image_ph):
        with tf.variable_scope('vgg'):
            # Conv1
            conv1_1 = layers.conv2d(image_ph, W=[3, 3, 3, 32], b=[32], strides=[1, 2, 2, 1])
            relu1_1 = tf.nn.relu(conv1_1)
            conv1_2 = layers.conv2d(relu1_1, W=[3, 3, 32, 32], b=[32], strides=[1, 2, 2, 1])
            relu1_2 = tf.nn.relu(conv1_2)
            pool1 = layers.max_pool(relu1_2)

            # Conv2
            conv2_1 = layers.conv2d(pool1, W=[3, 3, 32, 64], b=[64], strides=[1, 2, 2, 1])
            relu2_1 = tf.nn.relu(conv2_1)
            conv2_2 = layers.conv2d(relu2_1, W=[3, 3, 64, 64], b=[64], strides=[1, 2, 2, 1])
            relu2_2 = tf.nn.relu(conv2_2)
            pool2 = layers.max_pool(relu2_2)

image_ph = tf.placeholder(tf.float32, shape=[None, 104, 78, 3])
net = FCN()
net.vgg_part(image_ph)

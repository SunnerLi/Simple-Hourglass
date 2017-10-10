import _init_paths
from train import train
from UNet import UNet
import tensorflow as tf

if __name__ == '__main__':
    img_ph = tf.placeholder(tf.float32, [None, 104, 78, 3])
    ann_ph = tf.placeholder(tf.int32, [None, 104, 78, 1])
    net = UNet(img_ph, ann_ph)
    train(net, img_ph, ann_ph, 'loss of UNet')
#from keras.layers import Conv2DTranspose
import tensorflow as tf

def deconv(_tensor, kernel_size = [3, 3, 256, 1024], output_shape = [32, 65, 49, 256]):
    """
    w = tf.Variable(tf.random_normal(kernel_size))
    deconv = tf.nn.conv2d_transpose(_tensor, w, output_shape, strides=[1, 2, 2, 1])
    """
    with tf.name_scope('deconv'):
        w = tf.Variable(tf.random_normal(kernel_size))
        output_size = tf.stack([tf.shape(_tensor)[0], output_shape[1], output_shape[2], output_shape[3]])
        deconv = tf.nn.conv2d_transpose(_tensor, filter=w, output_shape=output_size, strides=[1, 2, 2, 1], padding='SAME', name=None)
        deconv = tf.reshape(deconv, output_size)
        return deconv

img_ph = tf.placeholder(tf.float32, [None, 33, 25, 1024])
net = deconv(img_ph, kernel_size = [3, 3, 256, 1024], output_shape = [32, 65, 49, 256])
print(net.shape)

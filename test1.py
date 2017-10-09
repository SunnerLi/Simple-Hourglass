import tensorflow as tf
import layers

a = tf.placeholder(tf.float32, [32, 7, 5, 32])
b = layers.conv2d_transpose(a, W=[3, 3, 16, 32], b=[16])
c = tf.reshape(b, [32, 4, 3, 32])
print(c.get_shape().as_list())
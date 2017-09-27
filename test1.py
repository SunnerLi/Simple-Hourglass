#from keras.layers import Conv2DTranspose
import tensorflow as tf

img_ph = tf.placeholder(tf.float32, [None, 1040, 780, 3])
net = tf.layers.conv2d_transpose(img_ph, filters=32, kernel_size=[3, 3])
print(net.shape)
import tensorflow as tf
import layers

class RedNet(object):
    def __init__(self, img_ph, ann_ph):
        # Form the network
        self.network = {}
        self.prediction, logits = self.formNet(img_ph, ann_ph)
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(ann_ph, squeeze_dims=[3]), logits=logits))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)

        # Crop gradient
        grads_and_vars = optimizer.compute_gradients(self.loss)
        crop_grads_and_vars = [(tf.clip_by_value(grad, -0.001, 0.001), var) for grad, var in grads_and_vars]
        self.train_op = optimizer.apply_gradients(crop_grads_and_vars)

    def formNet(self, img_ph, ann_ph, base_filter_num=32, layer_num=8):
        conv_layer_list = []
        deconv_layer_list = []
        curr_layer = img_ph

        for i in range(layer_num):
            conv1 = layers.simplified_conv2d_and_relu(curr_layer)
            curr_layer = layers.simplified_conv2d_and_relu(conv1)
            conv_layer_list.append(curr_layer)
        for i in range(layer_num-1, -1, -1):
            deconv1 = layers.simplified_conv2d_transpose_and_relu(curr_layer)
            if i == 0:
                img_channel = img_ph.get_shape().as_list()[-1]
                deconv2 = layers.simplified_conv2d_transpose_and_relu(deconv1, num_kernel=img_channel)
                curr_layer = tf.add(deconv2, img_ph)               
            else:
                deconv2 = layers.simplified_conv2d_transpose_and_relu(deconv1)
                curr_layer = tf.add(deconv2, conv_layer_list[i])
            deconv_layer_list.append(curr_layer)
        final_layer = layers.conv2d(curr_layer, W=[5, 5, img_channel, img_channel], b=[img_channel])
        return tf.expand_dims(tf.argmax(final_layer, axis=-1), axis=-1), final_layer

if __name__ == '__main__':
    img_ph = tf.placeholder(tf.float32, shape=[None, 1040, 780, 3])
    ann_ph = tf.placeholder(tf.int32, shape=[None, 1040, 780, 1])
    net = RedNet(img_ph, ann_ph)
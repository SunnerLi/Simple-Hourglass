import tensorflow as tf
import layers

class UNet(object):
    def __init__(self, img_ph, ann_ph):
        # Form the network
        self.network = {}
        self.prediction, logits = self.formNet(img_ph, ann_ph)
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(ann_ph, squeeze_dims=[3]), logits=logits))
        optimizer = tf.train.AdamOptimizer()

        # Crop gradient
        grads_and_vars = optimizer.compute_gradients(self.loss)
        crop_grads_and_vars = [(tf.clip_by_value(grad, -0.001, 0.001), var) for grad, var in grads_and_vars]
        self.train_op = optimizer.apply_gradients(crop_grads_and_vars)

    def formNet(self, img_ph, ann_ph, base_filter_num=8):
        down_layer_list = {}
        curr_layer = img_ph

        # Down sampling
        for i in range(5):
            num_filter = base_filter_num * 2 ** i
            if i == 0:
                conv1 = layers.conv2d(curr_layer, W=[3, 3, img_ph.get_shape().as_list()[-1], num_filter], b=[num_filter])
            else:
                conv1 = layers.conv2d(curr_layer, W=[3, 3, num_filter // 2, num_filter], b=[num_filter])
            relu1 = tf.nn.relu(conv1)
            conv2 = layers.conv2d(relu1, W=[3, 3, num_filter, num_filter], b=[num_filter])            
            down_layer_list[i] = tf.nn.relu(conv2)
            print('layer: ', i, '\tsize: ', down_layer_list[i].get_shape().as_list())
            if i < 4:
                curr_layer = layers.max_pool(down_layer_list[i])
        curr_layer = down_layer_list[4]

        # Up sampling
        for i in range(3, -1, -1):
            num_filter = base_filter_num * 2 ** (i+1)
            deconv_output_shape = tf.shape(down_layer_list[i])
            deconv1 = layers.conv2d_transpose(curr_layer, W=[3, 3, num_filter // 2, num_filter], b=[num_filter // 2], stride=2)
            concat1 = layers.crop_and_concat(tf.nn.relu(deconv1), down_layer_list[i])
            conv1 = layers.conv2d(concat1, W=[3, 3, num_filter, num_filter // 2], b=[num_filter // 2], strides=[1, 1, 1, 1])
            relu1 = tf.nn.relu(conv1)
            conv2 = layers.conv2d(relu1, W=[3, 3, num_filter // 2, num_filter // 2], b=[num_filter // 2], strides=[1, 1, 1, 1])            
            relu2 = tf.nn.relu(conv2)
            curr_layer = relu2

        # Output
        conv = layers.conv2d(curr_layer, W=[1, 1, base_filter_num, 3], b=[3])
        relu = tf.nn.relu(conv)
        print('final relu: ', relu.get_shape().as_list())
        return tf.expand_dims(tf.argmax(relu, axis=-1), axis=-1), relu

if __name__ == '__main__':
    img_ph = tf.placeholder(tf.float32, shape=[None, 1040, 780, 3])
    ann_ph = tf.placeholder(tf.int32, shape=[None, 1040, 780, 1])
    net = UNet(img_ph, ann_ph)
    
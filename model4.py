import tensorflow as tf
import layers

class RedNet(object):
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

    def formNet(self, img_ph, ann_ph, base_filter_num=1):
        conv_layer_list = {}
        deconv_layer_list = {}
        curr_layer = img_ph

        # Conv
        for i in range(5):
            num_filter = base_filter_num * 2 ** i
            if i == 0:
                conv1 = layers.conv2d(curr_layer, W=[3, 3, img_ph.get_shape().as_list()[-1], num_filter], b=[num_filter])
            else:
                conv1 = layers.conv2d(curr_layer, W=[3, 3, num_filter // 2, num_filter], b=[num_filter])
            relu1 = tf.nn.relu(conv1)
            conv2 = layers.conv2d(relu1, W=[3, 3, num_filter, num_filter], b=[num_filter])
            curr_layer = tf.nn.relu(conv2)
            conv_layer_list[i] = conv1
            print('conv', i, '\tsize: ', curr_layer.get_shape().as_list())

        # Deconv
        for i in range(3, -1, -1):
            num_filter = base_filter_num * 2 ** (i+1)
            print('curr: ', curr_layer.get_shape().as_list(), '\t', [3, 3, num_filter // 2, num_filter])
            deconv1 = layers.conv2d_transpose(curr_layer, W=[3, 3, num_filter // 2, num_filter], b=[num_filter // 2], output_shape=tf.shape(conv_layer_list[i]), stride=1)
            relu1 = tf.nn.relu(deconv1)
            deconv2 = layers.conv2d_transpose(relu1, W=[3, 3, num_filter // 2, num_filter // 2], b=[num_filter // 2], output_shape=tf.shape(conv_layer_list[i]), stride=1)
            relu2 = tf.nn.relu(deconv2)
            curr_layer = tf.add(relu2, conv_layer_list[i])
            deconv_layer_list[i] = curr_layer
        
        # Output
        deconv1 = layers.conv2d_transpose(deconv_layer_list[0], W=[3, 3, base_filter_num, base_filter_num], b=[base_filter_num], output_shape=tf.shape(conv_layer_list[i]), stride=1)
        relu1 = tf.nn.relu(deconv1)
        final_shape = tf.stack([tf.shape(img_ph)[0], tf.shape(img_ph)[1], tf.shape(img_ph)[2], 3])
        deconv2 = layers.conv2d_transpose(relu1, W=[3, 3, 3, base_filter_num], b=[3], output_shape=final_shape, stride=1)
        relu2 = tf.nn.relu(deconv2)
        print('final relu: ', relu2.get_shape().as_list())

        print('trainable: ', tf.trainable_variables())
        return tf.expand_dims(tf.argmax(relu2, axis=-1), axis=-1), relu2 

if __name__ == '__main__':
    img_ph = tf.placeholder(tf.float32, shape=[None, 1040, 780, 3])
    ann_ph = tf.placeholder(tf.int32, shape=[None, 1040, 780, 1])
    net = RedNet(img_ph, ann_ph)
import tensorflow as tf
import layers

class FCN8(object):
    layers_def = [
        ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1'],
        ['conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2'],
        ['conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3'],
        ['conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4'],
        ['conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'pool5'],
        ['conv6'],
        ['conv7'], 
        ['conv8']
    ]
    num_filter_times = [1, 2, 4, 8, 16, 128, 128, 3]
    filter_sizes = [3, 3, 3, 3, 3, 7, 1, 1]

    def __init__(self, img_ph, ann_ph):
        # Form the network
        self.network = {}
        self.prediction, logits = self.formNet(img_ph)
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(ann_ph, squeeze_dims=[3]), logits=logits))
        optimizer = tf.train.AdamOptimizer()

        # Crop gradient
        grads_and_vars = optimizer.compute_gradients(self.loss)
        crop_grads_and_vars = [(tf.clip_by_value(grad, -0.001, 0.001), var) for grad, var in grads_and_vars]
        self.train_op = optimizer.apply_gradients(crop_grads_and_vars)

    def vgg_part(self, image_ph, base_filter_num=2):
        current_layer = image_ph

        for i in range(len(self.num_filter_times)-1):
            self.num_filter_times[i] *= base_filter_num

        with tf.variable_scope('vgg'):
            for i, block in enumerate(self.layers_def):
                for layer in block:
                    if type(layer) == str:
                        if layer[:4] == 'conv':
                            prev_channel = current_layer.get_shape().as_list()[-1]
                            curr_channel = self.num_filter_times[i]
                            filter_size = self.filter_sizes[i]
                            current_layer = layers.conv2d(current_layer, W=[filter_size, filter_size, prev_channel, curr_channel], b=[curr_channel])
                        elif layer[:4] == 'relu':
                            current_layer = tf.nn.relu(current_layer, name=layer)
                        elif layer[:4] == 'pool':
                            current_layer = layers.max_pool(current_layer)
                        else:
                            print('invalid layer...')
                            exit()
                        self.network[layer] = current_layer
                        print(layer, '\toutput: ', current_layer.get_shape().as_list())
        print('out')
    
    def formNet(self, image_ph):
        self.vgg_part(image_ph)
        with tf.variable_scope('deconv'):
            deconv1_shape = self.network['pool4'].get_shape().as_list()
            prev_channel = deconv1_shape[-1]
            curr_channel = self.network['conv8'].get_shape().as_list()[-1]
            deconv1 = layers.conv2d_transpose(self.network['conv8'], W=[5, 5, prev_channel, curr_channel], b=[prev_channel], output_shape=tf.shape(self.network['pool4']))
            add1 = tf.add(deconv1, self.network['pool4'])

            deconv2_shape = self.network['pool3'].get_shape().as_list()
            prev_channel = deconv2_shape[-1]
            curr_channel = add1.get_shape().as_list()[-1]
            deconv2 = layers.conv2d_transpose(add1, W=[5, 5, prev_channel, curr_channel], b=[prev_channel], output_shape=tf.shape(self.network['pool3']))
            add2 = tf.add(deconv2, self.network['pool3'])

            deconv3_shape = image_ph.get_shape().as_list()
            output_shape = tf.stack([tf.shape(add2)[0], deconv3_shape[1], deconv3_shape[2], 3])
            prev_channel = deconv3_shape[-1]
            curr_channel = add2.get_shape().as_list()[-1]
            deconv3 = layers.conv2d_transpose(add2, W=[16, 16, prev_channel, curr_channel], b=[prev_channel], output_shape=output_shape, stride=8)
            self.predict = tf.argmax(deconv3, axis=-1)
        return tf.expand_dims(self.predict, axis=-1), deconv3

if __name__ == '__main__':
    img_ph = tf.placeholder(tf.float32, shape=[None, 104, 78, 3])
    ann_ph = tf.placeholder(tf.int32, shape=[None, 104, 78, 1])
    net = FCN(img_ph, ann_ph)
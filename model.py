from model_def import FCN8_Def
import tensorflow as tf
import layers

class FCN8(object):
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
        """
            Build the network toward VGG previous part
            If your computer didn't have enough RAM or GPU,
            you should set the size as smaller one
            (The base_filter_num in usual VGG-16 is 32)

            Arg:    image_ph        - The placeholder of images
                    base_filter_num - The base number of filter which can control the computation
        """
        # Import the definition of each layers
        current_layer = image_ph
        _def = FCN8_Def(base_filter_num)

        # Build graph topology
        with tf.variable_scope('vgg'):
            for i, layer_name in enumerate(_def.layers):
                if layer_name[:4] == 'conv':
                    prev_channel = current_layer.get_shape().as_list()[-1]
                    curr_channel = _def.num_filter_times[layer_name]
                    filter_size = _def.filter_sizes[layer_name]
                    current_layer = layers.conv2d(current_layer, W=[filter_size, filter_size, prev_channel, curr_channel], b=[curr_channel])
                elif layer_name[:4] == 'relu':
                    current_layer = tf.nn.relu(current_layer, name=layer_name)
                elif layer_name[:4] == 'pool':
                    current_layer = layers.max_pool(current_layer)
                else:
                    print('invalid layer...')
                    exit()
                self.network[layer_name] = current_layer
                print(layer_name, '\toutput: ', current_layer.get_shape().as_list())
    
    def formNet(self, image_ph):
        """
            Form the whole network of FCN-8
            The previous part is VGG-16 whose last three layer are conv layer
            The back part is deconv process

            Arg:    image_ph        - The placeholder of images
            Ret:    The prediction result and the logits
        """
        self.vgg_part(image_ph)
        with tf.variable_scope('deconv'):
            deconv1_shape = self.network['pool4'].get_shape().as_list()
            prev_channel = deconv1_shape[-1]
            curr_channel = self.network['conv6_3'].get_shape().as_list()[-1]
            deconv1 = layers.conv2d_transpose(self.network['conv6_3'], W=[5, 5, prev_channel, curr_channel], b=[prev_channel], output_shape=tf.shape(self.network['pool4']))
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
    net = FCN8(img_ph, ann_ph)
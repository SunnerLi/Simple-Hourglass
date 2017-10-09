from model_def import UNet_Def
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
        # crop_grads_and_vars = [(tf.clip_by_value(grad, -0.001, 0.001), var) for grad, var in grads_and_vars]
        crop_grads_and_vars = grads_and_vars
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
        _def = UNet_Def(base_filter_num)

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
    
    def formNet(self, image_ph, ann_ph):
        """
            Form the whole network of UNet
            The previous part is VGG-16 whose last three layer are conv layer
            The back part is upsampling + concatenate process

            Arg:    image_ph        - The placeholder of images
            Ret:    The prediction result and the logits
        """
        self.vgg_part(image_ph)
        with tf.variable_scope('deconv'):
            # Upsampling1
            prev_channel = self.network['pool5'].get_shape().as_list()[-1]
            curr_channel = self.network['pool4'].get_shape().as_list()[-1]
            deconv1 = layers.conv2d_transpose(self.network['pool5'], W=[2, 2, curr_channel, prev_channel], b=[curr_channel], output_shape=tf.shape(self.network['pool4']))
            relu6_1 = tf.nn.relu(deconv1)
            crop1 = layers.cropping(relu6_1, self.network['pool4'].get_shape().as_list())
            concat1 = tf.concat([crop1, self.network['pool4']], axis=-1)
            prev_channel = concat1.get_shape().as_list()[-1]
            conv6_1 = layers.conv2d(concat1, W=[3, 3, prev_channel, curr_channel], b=[curr_channel])
            relu6_2 = tf.nn.relu(conv6_1)
            conv6_2 = layers.conv2d(relu6_2, W=[3, 3, curr_channel, curr_channel], b=[curr_channel])
            relu6_3 = tf.nn.relu(conv6_2)

            # Upsampling2
            prev_channel = relu6_3.get_shape().as_list()[-1]
            curr_channel = self.network['pool3'].get_shape().as_list()[-1]
            deconv2 = layers.conv2d_transpose(relu6_3, W=[2, 2, curr_channel, prev_channel], b=[curr_channel], output_shape=tf.shape(self.network['pool3']))
            relu7_1 = tf.nn.relu(deconv2)
            crop2 = layers.cropping(relu7_1, self.network['pool3'].get_shape().as_list())
            concat2 = tf.concat([crop2, self.network['pool3']], axis=-1)
            prev_channel = concat2.get_shape().as_list()[-1]
            conv7_1 = layers.conv2d(concat2, W=[3, 3, prev_channel, curr_channel], b=[curr_channel])
            relu7_2 = tf.nn.relu(conv7_1)
            conv7_2 = layers.conv2d(relu7_2, W=[3, 3, curr_channel, curr_channel], b=[curr_channel])
            relu7_3 = tf.nn.relu(conv7_2)

            # Upsampling3
            prev_channel = relu7_3.get_shape().as_list()[-1]
            curr_channel = self.network['pool2'].get_shape().as_list()[-1]
            deconv3 = layers.conv2d_transpose(relu7_3, W=[2, 2, curr_channel, prev_channel], b=[curr_channel], output_shape=tf.shape(self.network['pool2']))
            relu8_1 = tf.nn.relu(deconv3)
            crop3 = layers.cropping(relu8_1, self.network['pool2'].get_shape().as_list())
            concat3 = tf.concat([crop3, self.network['pool2']], axis=-1)
            prev_channel = concat3.get_shape().as_list()[-1]
            conv8_1 = layers.conv2d(concat3, W=[3, 3, prev_channel, curr_channel], b=[curr_channel])
            relu8_2 = tf.nn.relu(conv8_1)
            conv8_2 = layers.conv2d(relu8_2, W=[3, 3, curr_channel, curr_channel], b=[curr_channel])
            relu8_3 = tf.nn.relu(conv8_2)
            
            # Upsampling4
            prev_channel = relu8_3.get_shape().as_list()[-1]
            curr_channel = self.network['pool1'].get_shape().as_list()[-1]
            deconv4 = layers.conv2d_transpose(relu8_3, W=[2, 2, curr_channel, prev_channel], b=[curr_channel], output_shape=tf.shape(self.network['pool1']))
            relu9_1 = tf.nn.relu(deconv4)
            crop4 = layers.cropping(relu9_1, self.network['pool1'].get_shape().as_list())
            concat4 = tf.concat([crop4, self.network['pool1']], axis=-1)
            prev_channel = concat4.get_shape().as_list()[-1]
            conv9_1 = layers.conv2d(concat4, W=[3, 3, prev_channel, curr_channel], b=[curr_channel])
            relu9_2 = tf.nn.relu(conv9_1)
            conv9_2 = layers.conv2d(relu9_2, W=[3, 3, curr_channel, curr_channel], b=[curr_channel])
            relu9_3 = tf.nn.relu(conv9_2)

            # Upsampling5
            prev_channel = relu9_3.get_shape().as_list()[-1]
            curr_channel = 3
            output_shape = tf.stack([tf.shape(ann_ph)[0], tf.shape(ann_ph)[1], tf.shape(ann_ph)[2], 3])
            deconv5 = layers.conv2d_transpose(relu9_3, W=[2, 2, curr_channel, prev_channel], b=[curr_channel], output_shape=output_shape)
            relu10_1 = tf.nn.relu(deconv5)            
            crop5 = layers.cropping(relu10_1, image_ph.get_shape().as_list())
            prev_channel = crop5.get_shape().as_list()[-1]
            conv10_1 = layers.conv2d(crop5, W=[3, 3, prev_channel, curr_channel], b=[curr_channel])
            relu10_2 = tf.nn.relu(conv10_1)
            conv10_2 = layers.conv2d(relu10_2, W=[3, 3, curr_channel, curr_channel], b=[curr_channel])
            relu10_3 = tf.nn.relu(conv10_2)

            # Final
            conv11 = layers.conv2d(relu10_3, W=[1, 1, curr_channel, curr_channel], b=[curr_channel])
            relu11 = tf.nn.relu(conv11)
            print('relu10_2\toutput: ', relu10_2.get_shape().as_list())
            self.predict = tf.argmax(relu11, axis=-1)
        return tf.expand_dims(self.predict, axis=-1), relu11

if __name__ == '__main__':
    img_ph = tf.placeholder(tf.float32, shape=[None, 1040, 780, 3])
    ann_ph = tf.placeholder(tf.int32, shape=[None, 1040, 780, 1])
    net = UNet(img_ph, ann_ph)
from model_def import FCN8_Def
import tensorlayer as tl
import tensorflow as tf

"""
    This program define the FCN-8 which is written by tensorflow and tensorlayer
    Moreover, this is the only one code which use tensorlayer to build the model
    Since my original code cannot converge to the corresponding output but whole black,
    I write this code

    You should use FCN.py rather than using FCN_fade.py to train the original model
"""

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

    def vgg_part(self, image_ph, base_filter_num=32):
        """
            Build the network toward VGG previous part
            If your computer didn't have enough RAM or GPU,
            you should set the size as smaller one
            (The base_filter_num in usual VGG-16 is 32)

            Arg:    image_ph        - The placeholder of images
                    base_filter_num - The base number of filter which can control the computation
        """
        # Import the definition of each layers
        current_layer = tl.layers.InputLayer(image_ph, name='fcn_input')
        _def = FCN8_Def(base_filter_num)

        # Build graph topology
        with tf.variable_scope('vgg'):
            for i, layer_name in enumerate(_def.layers):
                if layer_name[:4] == 'conv':
                    curr_channel = _def.num_filter_times[layer_name]
                    filter_size = (_def.filter_sizes[layer_name], _def.filter_sizes[layer_name])
                    current_layer = tl.layers.Conv2d(current_layer, curr_channel, filter_size=filter_size, name ='fcn_' + layer_name)
                elif layer_name[:4] == 'relu':
                    current_layer = tf.nn.relu(current_layer.outputs, name= layer_name)
                    current_layer = tl.layers.InputLayer(current_layer, name= 'fcn_' + layer_name + '_iput')
                elif layer_name[:4] == 'pool':
                    current_layer = tl.layers.MaxPool2d(current_layer, name = 'fcn' + layer_name)
                else:
                    print('invalid layer...')
                    exit()
                self.network[layer_name] = current_layer
    
    def formNet(self, image_ph):
        """
            Form the whole network of FCN-8
            The previous part is VGG-16 whose last three layer are conv layer
            The back part is deconv process

            Arg:    image_ph        - The placeholder of images
            Ret:    The prediction result and the logits
        """
        self.vgg_part(image_ph)
        current_layer = self.network['conv6_3']
        with tf.variable_scope('deconv'):
            deconv1_shape_list = self.network['pool4'].outputs.get_shape().as_list()
            deconv1_shape_tensor = tf.stack([tf.shape(self.network['pool4'].outputs)[0], deconv1_shape_list[1], deconv1_shape_list[2], deconv1_shape_list[3]])
            prev_channel = deconv1_shape_list[-1]
            out_size = (deconv1_shape_list[1], deconv1_shape_list[2])
            curr_channel = current_layer.outputs.get_shape().as_list()[-1]
            current_layer = tl.layers.DeConv2d(current_layer, prev_channel, out_size=out_size, batch_size = None, name='fcn_deconv1')
            current_layer = tl.layers.ReshapeLayer(current_layer, deconv1_shape_tensor, name='fcn_reshape1')
            current_layer = tl.layers.ElementwiseLayer([current_layer, self.network['pool4']], combine_fn = tf.add, name='fcn_up_add1')
            current_layer = tl.layers.InputLayer(tf.nn.relu(current_layer.outputs), name ='fcn_up_relu1')

            deconv2_shape_list = self.network['pool3'].outputs.get_shape().as_list()
            deconv2_shape_tensor = tf.stack([tf.shape(self.network['pool3'].outputs)[0], deconv2_shape_list[1], deconv2_shape_list[2], deconv2_shape_list[3]])
            prev_channel = deconv2_shape_list[-1]
            out_size = (deconv2_shape_list[1], deconv2_shape_list[2])
            curr_channel = current_layer.outputs.get_shape().as_list()[-1]
            current_layer = tl.layers.DeConv2d(current_layer, prev_channel, out_size=out_size, batch_size = None, name='fcn_deconv2')
            current_layer = tl.layers.ReshapeLayer(current_layer, deconv2_shape_tensor, name='fcn_reshape2')
            current_layer = tl.layers.ElementwiseLayer([current_layer, self.network['pool3']], combine_fn = tf.add, name='fcn_up_add2')
            current_layer = tl.layers.InputLayer(tf.nn.relu(current_layer.outputs), name ='fcn_up_relu2')

            img_shape = image_ph.get_shape().as_list()
            deconv3_shape_tensor = tf.stack([tf.shape(self.network['pool3'].outputs)[0], img_shape[1], img_shape[2], img_shape[3]])
            prev_channel = img_shape[3]
            out_size = (img_shape[1], img_shape[2])
            curr_channel = current_layer.outputs.get_shape().as_list()[-1]
            current_layer = tl.layers.DeConv2d(current_layer, prev_channel, filter_size=(16, 16), out_size=out_size, batch_size = None, strides = (8, 8), name='fcn_deconv3')
            current_layer = tl.layers.ReshapeLayer(current_layer, deconv3_shape_tensor, name='fcn_reshape3')
            current_layer = tf.nn.relu(current_layer.outputs)
            self.final_logits = tf.nn.relu(current_layer + 1e-8)              # Prevent nan loss
            self.predict = tf.argmax(self.final_logits, axis=-1)
        print('trainable: ', tf.trainable_variables())
        return tf.expand_dims(self.predict, axis=-1), self.final_logits

if __name__ == '__main__':
    img_ph = tf.placeholder(tf.float32, shape=[None, 35, 26, 3])
    ann_ph = tf.placeholder(tf.int32, shape=[None, 35, 26, 1])
    net = FCN8(img_ph, ann_ph)
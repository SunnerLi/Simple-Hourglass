import tensorlayer as tl
import tensorflow as tf

def deconv(_tensor, kernel_size = [3, 3, 256, 1024], output_shape = [32, 65, 49, 256]):
    with tf.name_scope('deconv'):
        w = tf.Variable(tf.random_normal(kernel_size))
        output_size = tf.stack([tf.shape(_tensor)[0], output_shape[1], output_shape[2], output_shape[3]])
        deconv = tf.nn.conv2d_transpose(_tensor, filter=w, output_shape=output_size, strides=[1, 2, 2, 1], padding='SAME', name=None)
        deconv = tf.reshape(deconv, output_size)
        return deconv

class Net(object):
    def work(self, ann_ph, predict):
        self.predict = predict
        self.ann_ph = ann_ph
        print(self.ann_ph.shape, self.predict.shape)
        # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.ann_ph, logits=self.predict))

        # Gradient explosion: https://github.com/MarvinTeichmann/KittiSeg/blob/master/optimizer/generic_optimizer.py
        ann_tensor = tf.reshape(ann_ph, [-1, tf.shape(predict)[-1]])
        predict_tensor = tf.reshape(predict, [-1, tf.shape(predict)[-1]])
        self.cross_entropy = -tf.reduce_sum(tf.multiply(ann_tensor, tf.log(predict_tensor)), reduction_indices=[1])

        print(self.cross_entropy)

        self.loss = tf.reduce_mean(self.cross_entropy)
        self.optimize = tf.train.AdamOptimizer().minimize(self.cross_entropy)
        self.optimizer = tf.train.AdamOptimizer()

        # Clip the gradient
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        crop_grads_and_vars = [(tf.clip_by_value(grad, -0.0001, 0.0001), var) for grad, var in grads_and_vars]
        self.train_op = self.optimizer.apply_gradients(crop_grads_and_vars)

class FCN8(Net):
    def __init__(self, img_ph, ann_ph):
        self.img_ph = img_ph
        self.ann_ph = ann_ph
        
        # Define VGG-16
        self.network = tl.layers.InputLayer(self.img_ph)
        self.network = tl.layers.Conv2d(self.network, n_filter=4, act = tf.nn.relu, name ='vgg_conv1')
        self.network = tl.layers.Conv2d(self.network, n_filter=4, act = tf.nn.relu, name ='vgg_conv2')
        self.pool1 = tl.layers.MaxPool2d(self.network, name='vgg_pool1')
        self.network = tl.layers.Conv2d(self.pool1, n_filter=8, act = tf.nn.relu, name ='vgg_conv3')
        self.network = tl.layers.Conv2d(self.network, n_filter=8, act = tf.nn.relu, name ='vgg_conv4')
        self.pool2 = tl.layers.MaxPool2d(self.network, name='vgg_pool2')
        self.network = tl.layers.Conv2d(self.pool2, n_filter=16, act = tf.nn.relu, name ='vgg_conv5')
        self.network = tl.layers.Conv2d(self.network, n_filter=16, act = tf.nn.relu, name ='vgg_conv6')
        self.network = tl.layers.Conv2d(self.network, n_filter=16, act = tf.nn.relu, name ='vgg_conv7')
        self.pool3 = tl.layers.MaxPool2d(self.network, name='vgg_pool3')
        self.network = tl.layers.Conv2d(self.pool3, n_filter=32, act = tf.nn.relu, name ='vgg_conv8')
        self.network = tl.layers.Conv2d(self.network, n_filter=32, act = tf.nn.relu, name ='vgg_conv9')
        self.network = tl.layers.Conv2d(self.network, n_filter=32, act = tf.nn.relu, name ='vgg_conv10')
        self.pool4 = tl.layers.MaxPool2d(self.network, name='vgg_pool4')
        self.network = tl.layers.Conv2d(self.pool4, n_filter=64, act = tf.nn.relu, name ='vgg_conv11')
        self.network = tl.layers.Conv2d(self.network, n_filter=64, act = tf.nn.relu, name ='vgg_conv12')
        self.network = tl.layers.Conv2d(self.network, n_filter=64, act = tf.nn.relu, name ='vgg_conv13')
        self.pool5 = tl.layers.MaxPool2d(self.network, name='vgg_pool5')

        # Define FC part
        batch, height, width, channel = self.pool5.outputs.shape
        self.network = tl.layers.Conv2d(self.pool5, n_filter=128, act = tf.nn.relu, name ='vgg_conv14')
        self.network = tl.layers.Conv2d(self.network, n_filter=128, act = tf.nn.relu, name ='vgg_conv15')

        # Define Deconv part
        self.network = deconv(self.network.outputs, kernel_size = [3, 3, 32, 128], output_shape = [32, 65, 49, 32])
        self.network = tl.layers.InputLayer(self.network, name ='input_layer1')
        self.network = tl.layers.ElementwiseLayer([self.network, self.pool4], combine_fn = tf.add, name='fcn_add1')
        
        self.network = deconv(self.network.outputs, kernel_size = [3, 3, 16, 32], output_shape = [32, 130, 98, 16])
        self.network = tl.layers.InputLayer(self.network, name ='input_layer2')        
        self.network = tl.layers.ElementwiseLayer([self.network, self.pool3], combine_fn = tf.add, name ='fcn_add2')
        
        batch_ann, height_ann, width_ann, channel_ann = ann_ph.shape
        self.network = tl.layers.DeConv2d(self.network, n_out_channel = int(channel_ann), filter_size=(7, 7), strides = (8, 8), out_size = (height_ann, width_ann), act = tf.nn.relu, name ='fcn_deconv3')        
        self.network = tl.layers.DeConv2d(self.network, n_out_channel = int(channel_ann), filter_size=(3, 3), strides = (1, 1), out_size = (height_ann, width_ann), act = tf.nn.softmax, name ='fcn_deconv4')        
        self.predict = self.network.outputs
        self.work(self.ann_ph, self.predict)
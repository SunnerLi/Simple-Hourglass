import tensorlayer as tl
import tensorflow as tf

class Net(object):
    def work(self, ann_ph, predict):
        self.predict = predict
        self.ann_ph = ann_ph
        print(self.ann_ph.shape, self.predict.shape)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.ann_ph, logits=self.predict))
        self.optimize = tf.train.AdamOptimizer().minimize(self.loss)

class FCN8(Net):
    def __init__(self, img_ph, ann_ph):
        self.img_ph = img_ph
        self.ann_ph = ann_ph
        
        # Define VGG-16
        self.network = tl.layers.InputLayer(img_ph)
        self.network = tl.layers.Conv2d(self.network, n_filter=32, act = tf.nn.relu, name ='vgg_conv1')
        self.network = tl.layers.Conv2d(self.network, n_filter=32, act = tf.nn.relu, name ='vgg_conv2')
        self.pool1 = tl.layers.MaxPool2d(self.network, name='vgg_pool1')
        self.network = tl.layers.Conv2d(self.pool1, n_filter=64, act = tf.nn.relu, name ='vgg_conv3')
        self.network = tl.layers.Conv2d(self.network, n_filter=64, act = tf.nn.relu, name ='vgg_conv4')
        self.pool2 = tl.layers.MaxPool2d(self.network, name='vgg_pool2')
        self.network = tl.layers.Conv2d(self.pool2, n_filter=128, act = tf.nn.relu, name ='vgg_conv5')
        self.network = tl.layers.Conv2d(self.network, n_filter=128, act = tf.nn.relu, name ='vgg_conv6')
        self.network = tl.layers.Conv2d(self.network, n_filter=128, act = tf.nn.relu, name ='vgg_conv7')
        self.pool3 = tl.layers.MaxPool2d(self.network, name='vgg_pool3')
        self.network = tl.layers.Conv2d(self.pool3, n_filter=256, act = tf.nn.relu, name ='vgg_conv8')
        self.network = tl.layers.Conv2d(self.network, n_filter=256, act = tf.nn.relu, name ='vgg_conv9')
        self.network = tl.layers.Conv2d(self.network, n_filter=256, act = tf.nn.relu, name ='vgg_conv10')
        self.pool4 = tl.layers.MaxPool2d(self.network, name='vgg_pool4')
        self.network = tl.layers.Conv2d(self.pool4, n_filter=512, act = tf.nn.relu, name ='vgg_conv11')
        self.network = tl.layers.Conv2d(self.network, n_filter=512, act = tf.nn.relu, name ='vgg_conv12')
        self.network = tl.layers.Conv2d(self.network, n_filter=512, act = tf.nn.relu, name ='vgg_conv13')
        self.pool5 = tl.layers.MaxPool2d(self.network, name='vgg_pool5')

        # Define FC part
        batch, height, width, channel = self.pool5.outputs.shape
        self.network = tl.layers.Conv2d(self.pool5, n_filter=1024, act = tf.nn.relu, name ='vgg_conv14')
        self.network = tl.layers.Conv2d(self.network, n_filter=1024, act = tf.nn.relu, name ='vgg_conv15')

        # Define Deconv part
        batch_pool4, height_pool4, width_pool4, channel_pool4 = self.pool4.outputs.shape
        self.network = tl.layers.DeConv2d(self.network, n_out_channel = int(channel_pool4), filter_size=(3, 3),  out_size = (height_pool4, width_pool4), act = tf.nn.relu, padding = 'SAME', name='fcn_deconv1')
        self.network = tl.layers.ElementwiseLayer([self.network, self.pool4], combine_fn = tf.add, name='fcn_add1')
        batch_pool3, height_pool3, width_pool3, channel_pool3 = self.pool3.outputs.shape
        self.network = tl.layers.DeConv2d(self.network, n_out_channel = int(channel_pool3), filter_size=(3, 3), out_size = (height_pool3, width_pool3), act = tf.nn.relu, padding = 'SAME', name='fcn_deconv2')
        self.network = tl.layers.ElementwiseLayer([self.network, self.pool3], combine_fn = tf.add, name ='fcn_add2')
        batch_ann, height_ann, width_ann, channel_ann = ann_ph.shape
        self.network = tl.layers.DeConv2d(self.network, n_out_channel = int(channel_ann), filter_size=(3, 3), strides = (8, 8), out_size = (height_ann, width_ann), act = tf.nn.softmax, padding = 'SAME', name ='fcn_deconv3')
        self.predict = self.network.outputs
        self.work(ann_ph, self.predict)


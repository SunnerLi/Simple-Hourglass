import tensorflow as tf

def get_weight(shape, stddev=0.1, name=None):
    init = tf.truncated_normal(shape=shape, stddev=stddev)
    if name == None:
        return tf.Variable(init)
    else:
        return tf.get_variable(name=name, initializer=init)

def get_bias(shape, name=None):
    init = tf.constant(0.01, shape=shape)
    if name == None:
        return tf.Variable(init)
    else:
        return tf.get_variable(name=name, initializer=init)

def conv2d(x, W, b, strides=[1, 1, 1, 1]):
    if type(W) == list:
        W = get_weight(W)
    if type(b) == list:
        b = get_weight(b)    
    conv = tf.nn.conv2d(x, W, strides=strides, padding='SAME')
    return tf.nn.bias_add(conv, b)

def conv2d_transpose(x, W, b, output_shape=None, stride=2):
    if type(W) == list:
        W = get_weight(W)
    if type(b) == list:
        b = get_weight(b)
    if output_shape == None:
        output_shape = x.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] = W.get_shape().as_list()[2]
    deconv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding="SAME")
    return tf.nn.bias_add(deconv, b)

def leaky_rely(x, alpha=0.01):
    return tf.maximum(alpha * x, x)

def max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1]):
    return tf.nn.max_pool(x, ksize=ksize, strides=strides, padding="SAME")

def batch_norm(x, phase_train=True, scope='bn'):
    """
        Modified from https://stackoverflow.com/a/34634291/2267819
    """
    channel = x.get_shape().as_list()[-1]
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[channel]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[channel]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def unpooling(x, output_shape=None):
    origin_shape = x.get_shape().as_list()
    output_shape = tf.stack([origin_shape[1] * 2, origin_shape[2] * 2])
    return tf.image.resize_bilinear(x, output_shape)

def crop_and_concat(tensor, prev):
    origin_height = tf.shape(tensor)[1]
    origin_width = tf.shape(tensor)[2]
    target_height = prev.get_shape().as_list()[1]
    target_width = prev.get_shape().as_list()[2]
    begin = [0, (origin_height - target_height) // 2, (origin_width - target_width) // 2, 0]
    size = [-1, target_height, target_width, -1]
    return tf.concat([tf.slice(tensor, begin, size), prev], axis=3)
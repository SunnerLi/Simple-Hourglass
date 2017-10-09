
class FCN8_Def(object):
    """
        This class define the layer of FCN-8
        To let the user control the computation, it provide user to set lower base_filter_num
        If you just set 2, it can speed up the procedure
    """
    layers = [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'pool5',
        'conv6_1', 'conv6_2', 'conv6_3'
    ]
    num_filter_times = {
        'conv1_1': 1, 'conv1_2': 1,
        'conv2_1': 2, 'conv2_2': 2,
        'conv3_1': 4, 'conv3_2': 4, 'conv3_3': 4,
        'conv4_1': 8, 'conv4_2': 8, 'conv4_3': 8,
        'conv5_1': 16, 'conv5_2': 16, 'conv5_3': 16,
        'conv6_1': 128, 'conv6_2': 128, 'conv6_3': 3
    }
    filter_sizes = {
        'conv1_1': 3, 'conv1_2': 3,
        'conv2_1': 3, 'conv2_2': 3,
        'conv3_1': 3, 'conv3_2': 3, 'conv3_3': 3,
        'conv4_1': 3, 'conv4_2': 3, 'conv4_3': 3,
        'conv5_1': 3, 'conv5_2': 3, 'conv5_3': 3,
        'conv6_1': 7, 'conv6_2': 1, 'conv6_3': 1
    }
    def __init__(self, base_filter_num):
        # Revised the number of filters
        for key, value in self.num_filter_times.items():
            if key != 'conv6_3':
                self.num_filter_times[key] *= base_filter_num

class UNet_Def(object):
    """
        This class define the layer of FCN-8
        To let the user control the computation, it provide user to set lower base_filter_num
        If you just set 2, it can speed up the procedure
    """
    layers = [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'pool5'
    ]
    num_filter_times = {
        'conv1_1': 1, 'conv1_2': 1,
        'conv2_1': 2, 'conv2_2': 2,
        'conv3_1': 4, 'conv3_2': 4, 
        'conv4_1': 8, 'conv4_2': 8, 
        'conv5_1': 16, 'conv5_2': 16
    }
    filter_sizes = {
        'conv1_1': 3, 'conv1_2': 3,
        'conv2_1': 3, 'conv2_2': 3,
        'conv3_1': 3, 'conv3_2': 3,
        'conv4_1': 3, 'conv4_2': 3,
        'conv5_1': 3, 'conv5_2': 3
    }
    def __init__(self, base_filter_num):
        pass
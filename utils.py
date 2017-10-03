from collections import Counter, OrderedDict
import numpy as np

"""
    This function provide to_categorical which just like keras.utils
"""

def to_categorical_4d(_input_tensor, pallete=None, verbose=1):
    """
        Change the representation to the one-hot format
        Just like the keras' to_categorical but in 4D shae

        Arg:    _input_tensor   - The input tensor want to change into one-hot format
                pallete         - The pallete object (default is None)
                verbose         - Show the percentage of completing                
        Ret:    The result tensor in 4D shape [batch_num, height, width, class_num]
    """ 
    # Build the pallete and create the zero array
    pallete = __buildPallete(_input_tensor, pallete)
    batch, height, width, channel = np.shape(_input_tensor)
    _result_tensor = np.zeros([batch, height, width, len(pallete)])

    # Fill the corresponding index as 1
    for i in range(batch):
        if verbose == 1:
            print('to_categorical_4d finish: ', i/batch)
        for j in range(height):
            for k in range(width):
                _result_tensor[i][j][k][pallete[tuple(_input_tensor[i, j, k, :].tolist())]] = 1
    return _result_tensor, pallete

def to_categorical_4d_reverse(_input_tensor, pallete):
    """
        Change the representation to the original format:
        [batch_num, height, width, RBG_channel_num(3)]
        The output is the color image rather than the index tensor

        Arg:    _input_tensor   - The tensor which you want to transfer to the original format
                pallete         - The pallete object which will be the mapping between index and color
        Ret:    The batch annotations with original color intensity
    """
    # Build reverse index tensor
    reverse_pallete = {pallete[x]: x for x in pallete}
    decode_map = np.argmax(_input_tensor, axis=-1)
    batch, height, width, channel = np.shape(_input_tensor)        
    _result_tensor = np.zeros([batch, height, width, 3])    

    # Mapping into origin color
    _result_tensor = np.reshape(np.copy(decode_map), [-1])
    _result_tensor = np.asarray(np.vectorize(reverse_pallete.get)(_result_tensor)).T
    return np.reshape(_result_tensor, [batch, height, width, 3])

def __buildPallete(_input_tensor, pallete):
    """
        Build the mapping of color and categorical index
        The warning will raise if you had built the mapping
        This function only accepts 4D tensor
        It's recommend you shouldn't use this function directly

        In fact, this function will create a new pallete.
        If the pallete isn't None, this function will give the warning to tell you.
        This function will append the new color into the existed pallete object

        Arg:    _input_tensor   - The input tensor want to change into one-hot format
                pallete         - The pallete object
        Ret:    The pallete mapping
    """
    if pallete != None:
        print('< ear_pen >  warning: You had built the pallete mapping')
    else:
        pallete = dict()
    if len(np.shape(_input_tensor)) != 4:
        print('< ear_pen >  error: The shape of tensor should be 4!')
        exit()
    unique_colors = np.unique(np.reshape(_input_tensor, [-1, 3]), axis=0)
    for i in range(len(unique_colors)):
        pallete[tuple(unique_colors[i].tolist())] = len(pallete)
    return pallete
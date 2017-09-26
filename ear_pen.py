import h5py
import os

def load_data(path='./ear_pen.h5'):
    """
        Load the Ear-pen dataset from the .h5 file

        Arg:    path    - The path of .h5 file
        Ret:    The 2 tuple which contains images and annotations
    """
    if not os.path.exists(path):
        print('You should generate the ear_pen hdf5 file first...')
        exit()
    f = h5py.File(path)
    return (f['train_x'], f['train_y']), (f['test_x'], f['test_y'])
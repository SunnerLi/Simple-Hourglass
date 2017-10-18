import _init_paths
from utils import to_categorical_4d, to_categorical_4d_reverse, denoising
from FCN import FCN8
from UNet import UNet
from RedNet import RedNet
from draw import drawRec
import tensorflow as tf
import numpy as np
import ear_pen
import math
import cv2

# Constant
batch_size = 1
video_name = 'move.mp4'
video_height = 104
video_width = 78

# Model path
# model_store_path = '../model/FCN-8/FCN-8.ckpt'
# model_store_path = '../model/UNet/UNet.ckpt'
model_store_path = '../model/RedNet/RedNet.ckpt'

def formShowImg(image, predictions, _map, counter):
    """
        image           - origin
        predictions     - without reverse
    """
    prediction = np.asarray(to_categorical_4d_reverse(predictions, _map), dtype=np.uint8)[0]
    print(np.shape(prediction))
    prediction = cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR)

    # -----------------------------------------------------------------------------------------
    # Do open and close operation to delete the noise
    # (default is commented, you should un-comment by yourself if you want to do)
    # -----------------------------------------------------------------------------------------
    # prediction = denoising(prediction)
    bboxImage = drawRec(image, prediction)
    show_img = np.concatenate((image, prediction, bboxImage), axis=1)

    # -----------------------------------------------------------------------------------------
    # Resize the showing image
    # (default is commented, you should un-comment by yourself if you want to do)
    # -----------------------------------------------------------------------------------------
    # show_img = cv2.resize(show_img, (np.shape(show_img)[1] * 2, np.shape(show_img)[0] * 2))
    return show_img

if __name__ == '__main__':
    img_ph = tf.placeholder(tf.float32, [None, video_height, video_width, 3])
    ann_ph = tf.placeholder(tf.int32, [None, video_height, video_width, 1])
    net = RedNet(img_ph, ann_ph)      # You should revise here if you switch to another model

    # Load data
    (train_img, train_ann), (test_img, test_ann) = ear_pen.load_data()
    train_ann, _map = to_categorical_4d(train_ann)

    # Work
    saver = tf.train.Saver()
    loss_list = []
    counter = 0
    with tf.Session() as sess:
        # Recover model
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, model_store_path)
        
        # Examine video
        cap = cv2.VideoCapture(video_name)
        while cap.isOpened():
            is_cap_open, frame = cap.read()
            fixed_img = cv2.resize(frame, (video_width, video_height))
            _pred = sess.run(net.prediction, feed_dict={
                img_ph: np.expand_dims(fixed_img, axis=0)
            })

            # Show predict annotation
            result_img = formShowImg(fixed_img, _pred, _map, counter)
            counter += 1
            cv2.imshow('predict result', result_img)
            cv2.waitKey(50)
        print('video capture close...')
from keras import backend as K
import time
from multiprocessing.dummy import Pool
K.set_image_data_format('channels_first')
import cv2
import os
import glob
import numpy as  np
from numpy import genfromtxt
import tensorflow as tf
from keras.models import load_model
import sys
from model import *
from utils import *
from face_utilities import prepare_database, detect_face, recognise_face


def triplet_loss_function(y_true, y_pred, alpha=0.3):
    """Calculate the triplet loss according to the FaceNet paper

    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.

    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """

    anchor = y_pred[0]
    positive = y_pred[1]
    negative = y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    return loss


if __name__ == '__main__':
    # model = model(input_shape=(3, 96,96))
    # model.compile(optimizer='adam', loss=triplet_loss_function, metrics=['accuracy'])
    # print('loading weight of model')
    # load_weights_from_FaceNet(model)
    # print('successfully load model')
    # model.save("model.h5")
    # print('model saved to disk')
    print('model load start')
    model = load_model('model.h5', custom_objects={'triplet_loss_function': triplet_loss_function})
    print('model loaded')
    while True:
        decision = input("Initiate face_recognition sequence press Y/N: ")
        # decision = sys.argv[1]
        if decision == ('y' or 'Y'):
            print('initialising webcam')
            image = detect_face('temp.jpg')
            database = prepare_database(model)
            face = recognise_face("temp.jpg", database, model)
            print(face)
            os.remove("temp.jpg")

        if decision == ('n' or 'N'):
            print("Face_recognition sequence closing....")
            break
        cv2.destroyAllWindows()






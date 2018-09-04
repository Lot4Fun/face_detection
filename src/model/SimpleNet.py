#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os

import keras
from keras.layers import Input, Dense, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
from keras.layers.core import Dropout
from keras import optimizers
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau, TensorBoard

from logging import DEBUG, INFO
from logging import getLogger

# Set logger
logger = getLogger('impulso')

logger.info(tf.__version__)
logger.info(keras.__version__)


class SimpleNet(object):

    def __init__(self, exec_type, hparams):
        logger.info('Begin init of Aggregator')
        self.exec_type = exec_type
        self.hparams = hparams


    def create_model(self):

        logger.info('Begin to create VGG16 model')

        logger.info('Input layer')
        input_h = self.hparams[self.exec_type]['resize']['height']
        input_w = self.hparams[self.exec_type]['resize']['width']
        inputs = Input(shape=(input_h, input_w, 3))

        logger.info('Block1')
        x = Conv2D(16, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block1_pool')(x)

        logger.info('Full Connection')
        flattened = Flatten(name='flatten')(x)
        x = Dense(128, activation='relu', name='fc1')(flattened)
        x = Dropout(0.5, name='dropout1')(x)

        logger.info('Output layer')
        predictions = Dense(input_h * input_w, activation='sigmoid', name='predictions')(x)

        logger.info('Create model')
        self.model = Model(inputs=inputs, outputs=predictions)

        logger.info('Finish creating ImpulsoNet model')


    def select_optimazer(self):

        logger.info('Create optimizer')
        sgd = optimizers.SGD(lr=0.01,
                             momentum=0.9,
                             decay=0.0005)
        
        self.selected_optimizer = sgd
    

    def compile(self):

        logger.info('Compile VGG16 model')
        self.model.compile(optimizer=self.selected_optimizer,
                           loss='mean_squared_error',
                           metrics=['accuracy'])
        
        self.model.summary()


if __name__ == '__main__':

    import datetime

    os.environ['IMPULSO_HOME'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')

    from logging import StreamHandler, FileHandler, Formatter

    # Set HOME directory.
    IMPULSO_HOME = os.environ['IMPULSO_HOME']

    # Set loger.
    log_date = datetime.datetime.today().strftime('%Y%m%d')
    log_path = os.path.join(IMPULSO_HOME, f'log/log_{log_date}.log')
    logger.setLevel(DEBUG)

    stream_handler = StreamHandler()
    file_handler = FileHandler(log_path)

    stream_handler.setLevel(INFO)
    file_handler.setLevel(DEBUG)

    handler_format = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(handler_format)
    file_handler.setFormatter(handler_format)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    modeler = ImpulsoNet()
    modeler.create_model()
    modeler.select_optimazer()
    modeler.compile()

    
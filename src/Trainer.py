#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import numpy as np
from .lib import utils

from logging import DEBUG
from logging import getLogger

logger = getLogger('impulso')
logger.setLevel(DEBUG)

# Set HOME directory.
IMPULSO_HOME = os.environ['IMPULSO_HOME']


class Trainer(object):

    def __init__(self, exec_type, hparams, model, model_id=None):
        logger.info('Begin init of Trainer')
        self.exec_type = exec_type
        self.hparams = hparams
        if model_id:
            self.hparams[exec_type]['fit']['initial_epoch'] = model_id
        else:
            self.hparams[exec_type]['fit']['initial_epoch'] = 0            
        self.model = model
        self.input_home = os.path.join(IMPULSO_HOME, 'datasets', f'{self.hparams["prepare"]["data_id"]}', f'{self.exec_type}')
        self.output_home = os.path.join(IMPULSO_HOME, 'experiments', f'{self.hparams["prepare"]["experiment_id"]}')

        logger.info('Check hparams.yaml')
        utils.check_hparams(self.exec_type, self.hparams)

        logger.info('Backup hparams.yaml and src')
        utils.backup_before_run(self.exec_type, self.hparams)

        os.makedirs(os.path.join(self.output_home, 'models'), exist_ok=True)
        logger.info('End init of Trainer')


    def load_data(self):
        logger.info('Load data')
        self.x_train = np.load(os.path.join(self.input_home, 'x', 'x.npy'))
        self.t_train = np.load(os.path.join(self.input_home, 't', 't.npy'))


    def append_callbacks(self):
        logger.info('Prepare callbacks')
        callbacks = []

        if 'ModelCheckpoint' in self.hparams[self.exec_type]['fit']['callbacks'].keys():
            if self.hparams[self.exec_type]['fit']['callbacks']['ModelCheckpoint']['enable']:
                logger.info('Enable: ModelCheckpoint')
                from keras.callbacks import ModelCheckpoint
                model_path = os.path.join(self.output_home, f'models')
                model_name = 'models.{epoch:05d}-{loss:.2f}-{acc:.2f}-{val_loss:.2f}-{val_acc:.2f}.hdf5'
                self.hparams[self.exec_type]['fit']['callbacks']['ModelCheckpoint']['hparams']['filepath'] = os.path.join(model_path, model_name)

                callback_hparams = self.hparams[self.exec_type]['fit']['callbacks']['ModelCheckpoint']['hparams']
                callback = ModelCheckpoint(filepath=callback_hparams['filepath'], 
                                           monitor=callback_hparams['monitor'],
                                           verbose=callback_hparams['verbose'],
                                           save_best_only=callback_hparams['save_best_only'],
                                           save_weights_only=callback_hparams['save_weights_only'],
                                           mode=callback_hparams['mode'],
                                           period=callback_hparams['period'])
                callbacks.append(callback)
            else:
                logger.info('Disable: ModelCheckpoint')
        
        if 'ReduceLROnPlateau' in self.hparams[self.exec_type]['fit']['callbacks'].keys():
            if self.hparams[self.exec_type]['fit']['callbacks']['ReduceLROnPlateau']['enable']:
                logger.info('Enable: ReduceLROnPlateau')
                from keras.callbacks import ReduceLROnPlateau
                callback_hparams = self.hparams[self.exec_type]['fit']['callbacks']['ReduceLROnPlateau']['hparams']
                callback = ReduceLROnPlateau(monitor=callback_hparams['monitor'],
                                             factor=callback_hparams['factor'],
                                             patience=callback_hparams['patience'],
                                             verbose=callback_hparams['verbose'],
                                             mode=callback_hparams['mode'],
                                             epsilon=callback_hparams['epsilon'],
                                             cooldown=callback_hparams['cooldown'],
                                             min_lr=callback_hparams['min_lr'])
                callbacks.append(callback)
            else:
                logger.info(f'Disable callback:{callback}')
            
        self.callbacks = callbacks


    def begin_train(self):
        hparams_fit = self.hparams[self.exec_type]['fit']
        self.model.fit(self.x_train,
                       self.t_train,
                       batch_size=hparams_fit['batch_size'],
                       epochs=hparams_fit['epochs'],
                       verbose=hparams_fit['verbose'],
                       validation_split=hparams_fit['validation_split'],
                       shuffle=hparams_fit['shuffle'],
                       initial_epoch=hparams_fit['initial_epoch'],
                       callbacks=self.callbacks)


if __name__ == '__main__':
    """add"""

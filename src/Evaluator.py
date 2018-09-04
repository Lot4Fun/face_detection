#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from .lib import utils as utils

from logging import DEBUG
from logging import getLogger

logger = getLogger('impulso')
logger.setLevel(DEBUG)

# Set HOME directory.
IMPULSO_HOME = os.environ['IMPULSO_HOME']


class Evaluator(object):

    def __init__(self, exec_type, hparams):
        logger.info('Begin init of Evaluator')
        self.exec_type = exec_type
        assert self.exec_type in ['validate', 'test'], 'Evaluator must be called in validate or test phase.'
        self.hparams = hparams
        self.y_home = os.path.join(IMPULSO_HOME, 'experiments', f'{self.hparams["prepare"]["experiment_id"]}', f'{self.exec_type}')
        if self.exec_type == 'validate':
            self.x_home = os.path.join(IMPULSO_HOME, 'datasets', f'{self.hparams["prepare"]["data_id"]}', f'{self.exec_type}', 'x')
            self.t_home = os.path.join(IMPULSO_HOME, 'datasets', f'{self.hparams["prepare"]["data_id"]}', f'{self.exec_type}', 't')
        elif self.exec_type == 'test':
            self.x_home = os.path.join(IMPULSO_HOME, 'datasets', f'{self.exec_type}', 'x')
            self.t_home = os.path.join(IMPULSO_HOME, 'datasets', f'{self.exec_type}', 't')
        self.output_home = self.y_home

        os.makedirs(os.path.join(self.output_home, 'scores'), exist_ok=True)
        os.makedirs(os.path.join(self.output_home, 'figures'), exist_ok=True)

        logger.info('Check hparams.yaml')
        utils.check_hparams(self.exec_type, self.hparams)

        logger.info('Backup hparams.yaml and src')
        utils.backup_before_run(self.exec_type, self.hparams)

        logger.info('End init of Estimator')


    def load_data(self):
        logger.info('Loading data...')
        self.x = np.load(os.path.join(self.x_home, 'x.npy'))
        self.y = np.load(os.path.join(self.y_home, 'y.npy'))
        self.t = np.load(os.path.join(self.t_home, 't.npy'))
        self.filename = np.load(os.path.join(self.x_home, 'filename.npy'))
        assert len(self.x) == len(self.y) == len(self.t) == len(self.filename), 'Length of x, y, t and filename are different.'


    def evaluate(self):
        logger.info('Evaluating...')
        scores = pd.DataFrame(columns=['FileName', 'RMS'])

        for x, y, t, filename in zip(self.x, self.y, self.t, self.filename):
            rms = np.sqrt(np.square(y - t).sum() / y.size)
            # Store with DataFrame
            idx = len(scores.index)
            scores.loc[idx, 'FileName'] = filename
            scores.loc[idx, 'RMS'] = rms

            # Save Image
            x_image = Image.fromarray(x)
            x_image -= np.min(x_image)
            x_image = np.minimum(x_image, 255)           

            resize_h = self.hparams['common']['resize']['height']
            resize_w = self.hparams['common']['resize']['width']            
            y_image = Image.fromarray(y.reshape(resize_h, resize_w))
            heatmap = y_image / np.max(y_image)
            y_image = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

            output_image = np.float32(y_image) + np.float32(x_image)
            output_image = 255 * output_image / np.max(output_image)

            cv2.imwrite(os.path.join(self.output_home, 'figures', filename), output_image)

        logger.info('Calculate total score')
        avg = scores['RMS'].mean()
        idx = len(scores.index)
        scores.loc[idx, 'FileName'] = 'Average'
        scores.loc[idx, 'RMS'] = avg

        scores.to_csv(os.path.join(self.output_home, 'scores', 'scores.csv'), index=False)


if __name__ == '__main__':
    """add"""

#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import numpy as np
from .lib import utils as utils

from logging import DEBUG
from logging import getLogger

logger = getLogger('impulso')
logger.setLevel(DEBUG)

# Set HOME directory.
IMPULSO_HOME = os.environ['IMPULSO_HOME']


class Estimator(object):

    def __init__(self, exec_type, hparams, model, x_dir=None, y_dir=None):
        logger.info('Begin init of Estimator')
        self.exec_type = exec_type
        self.hparams = hparams
        self.model = model
        # Set input_home
        if self.exec_type == 'test':
            self.input_home = os.path.join(IMPULSO_HOME, 'datasets', f'{self.exec_type}', 'x')
        elif self.exec_type == 'predict':
            assert os.path.exists(x_dir), 'Input data does not exist.'
            assert y_dir, 'y_dir is not spedified.'
            self.input_home = x_dir
        else:
            pass
        # Set output_home
        if self.exec_type == 'test':
            self.output_home = os.path.join(IMPULSO_HOME, 'experiments', f'{self.hparams["prepare"]["experiment_id"]}', f'{self.exec_type}')
        elif self.exec_type == 'predict':
            os.makedirs(y_dir, exist_ok=True)
            self.output_home = y_dir
        else:
            pass

        logger.info('Check hparams.yaml')
        utils.check_hparams(self.exec_type, self.hparams)

        logger.info('Backup hparams.yaml and src')
        utils.backup_before_run(self.exec_type, self.hparams)

        logger.info('End init of Estimator')
    
    
    def load_data(self):
        logger.info('Loading data...')
        if self.exec_type == 'test':
            x_path = os.path.join(self.input_home, 'x.npy')
            self.x = np.load(x_path)
        elif self.exec_type == 'predict':
            pass
            # predictのときは，画像を加工するところからやる
            # org_sizeを残しておかないと元に戻せない
            # 推論結果をどうやって可視化するかも課題（GradCAMみたいなヒートマップ？）
        else:
            pass


    def estimate(self):
        logger.info('Predicting...')
        self.y = self.model.predict(self.x, verbose=1)


    def save_results(self):
        os.makedirs(self.output_home, exist_ok=True)
        np.save(file=os.path.join(self.output_home, 'y.npy'), arr=self.y)

if __name__ == '__main__':
    """add"""

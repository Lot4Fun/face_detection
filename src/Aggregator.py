#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import glob
import json
import numpy as np
import cv2
from .lib import utils

from logging import DEBUG
from logging import getLogger

logger = getLogger('impulso')
logger.setLevel(DEBUG)

# Set HOME directory.
IMPULSO_HOME = os.environ['IMPULSO_HOME']


class Aggregator(object):

    def __init__(self, exec_type, hparams):
        logger.info('Begin init of Aggregator')
        self.exec_type = exec_type
        self.hparams = hparams
        self.hparams[exec_type]['data_id'] = utils.issue_id()
        self.hparams[exec_type]['output_train'] = os.path.join(IMPULSO_HOME,
                                                               'datasets',
                                                               self.hparams[exec_type]['data_id'],
                                                               'train')
        self.hparams[exec_type]['output_test'] = os.path.join(IMPULSO_HOME, 'datasets/test')

        logger.info('Check hparams.yaml')
        utils.check_hparams(self.exec_type, self.hparams)

        logger.info('Backup hparams.yaml and src')
        utils.backup_before_run(self.exec_type, self.hparams)
        logger.info('End init of Aggregator')


    def download_data(self):
        """
        cifar10_path = os.path.join(IMPULSO_HOME, 'org/cifar-10-python.tar.gz')
        if not os.path.exists(cifar10):
            if not os.path.exists(os.path.dirname(cifar10_path)):
                os.makedirs(os.path.dirname(cifar10_path), exist_ok=True)
        """
        pass


    def load_data(self):
        logger.info('Load FDDB dataset')
        with open(os.path.join(self.hparams[self.exec_type]['input_path'], 'bbox.json')) as f:
            bboxes = json.load(f)
        
        logger.info('Loading images...')
        images = []
        labels = []
        filenames = []
        for box in bboxes:
            image_path = os.path.join(self.hparams[self.exec_type]['input_path'], box['FileName'])
            # Input image
            resize_w = self.hparams['common']['resize']['width']
            resize_h = self.hparams['common']['resize']['height']
            image = cv2.imread(image_path)
            org_h, org_w, _ = image.shape
            image = cv2.resize(image, (resize_w, resize_h))
            images.append(image)
            # Label image
            label_image = np.zeros(resize_h * resize_w).reshape(resize_h, resize_w)
            for face in box['BBox']:
                left = int(face['Left'] * resize_w / org_w)
                top = int(face['Top'] * resize_h / org_h)
                width = int(face['Width'] * resize_w / org_w)
                height = int(face['Height'] * resize_h / org_h)
                label_image[top:top+height+1, left:left+width+1] = 1.
            labels.append(label_image.flatten())
            filenames.append(box['FileName'])

        logger.info('Split into train and test data')
        x = np.array(images)
        t = np.array(labels)
        filenames = np.array(filenames)
        self.x_train, self.x_test = np.split(x, [int(x.shape[0] * (1. - self.hparams[self.exec_type]['test_split']))])
        self.t_train, self.t_test = np.split(t, [int(t.shape[0] * (1. - self.hparams[self.exec_type]['test_split']))])
        self.train_filename, self.test_filename = np.split(filenames, [int(len(filenames) * (1. - self.hparams[self.exec_type]['test_split']))])
        assert len(self.x_train) == len(self.train_filename), 'Lengths of x_train and train_filename is different'
        assert len(self.x_test) == len(self.test_filename), 'Lengths of x_test and test_filename is different'
        logger.info('End loading FDDB dataset')


    def save_data(self):
        logger.info('Save data')
        train_x_dir = os.path.join(self.hparams['dataset']['output_train'], 'x')
        train_t_dir = os.path.join(self.hparams['dataset']['output_train'], 't')
        test_x_dir = os.path.join(self.hparams['dataset']['output_test'], 'x')
        test_t_dir = os.path.join(self.hparams['dataset']['output_test'], 't')
        
        for output_dir in [train_x_dir, train_t_dir, test_x_dir, test_t_dir]:
            os.makedirs(output_dir, exist_ok=True)
        
        np.save(file=os.path.join(train_x_dir, 'x.npy'), arr=self.x_train)
        np.save(file=os.path.join(train_t_dir, 't.npy'), arr=self.t_train)
        np.save(file=os.path.join(train_x_dir, 'filename.npy'), arr=self.train_filename)
        np.save(file=os.path.join(test_x_dir, 'x.npy'), arr=self.x_test)
        np.save(file=os.path.join(test_t_dir, 't.npy'), arr=self.t_test)
        np.save(file=os.path.join(test_x_dir, 'filename.npy'), arr=self.test_filename)

        logger.info('End saving data')


if __name__ == '__main__':
    """add"""

#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Import Machine Learning modules.
import os
os.environ['IMPULSO_HOME'] = os.path.dirname(os.path.abspath(__file__))

import sys
import datetime
import glob
import argparse
import src.lib.utils as utils
from src.Aggregator import Aggregator
from src.Preparer import Preparer
from src.Trainer import Trainer
from src.Estimator import Estimator
from src.Evaluator import Evaluator

from src.model.SimpleNet import SimpleNet

from logging import DEBUG, INFO
from logging import getLogger, StreamHandler, FileHandler, Formatter

# Set HOME directory.
IMPULSO_HOME = os.environ['IMPULSO_HOME']

# Set loger.
log_date = datetime.datetime.today().strftime('%Y%m%d')
log_path = os.path.join(IMPULSO_HOME, f'log/log_{log_date}.log')
logger = getLogger('impulso')
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


class Impulso(object):

    ### 後で追加したい機能
    # optimizerは外に出す
    # latest機能：data_id, experiment_idを指定しない場合は最新のデータ，experimentを使用する
    # Grid Search機能：python impulso.py grid でGrid Search用のhparamsのセットを作りたい

    def __init__(self, args, hparams_yaml='hparams.yaml'):
        logger.info('Begin init of Impulso')
        self.args = args
        if self.args.exec_type in ['dataset', 'prepare']:
            self.hparams_path = os.path.join(IMPULSO_HOME, f'hparams/{hparams_yaml}')
        else:
            self.hparams_path = os.path.join(IMPULSO_HOME, f'experiments/{self.args.experiment_id}/hparams/hparams.yaml')
        self.hparams = utils.load_hparams(self.hparams_path)
        logger.info('End init of Impulso')


    def dataset(self):
        logger.info('Begin dataset of Impulso')
        aggregator = Aggregator(self.args.exec_type, self.hparams)
        aggregator.load_data()
        aggregator.save_data()
        logger.info('DATA-ID: ' + aggregator.hparams[self.args.exec_type]['data_id'])
        logger.info('End dataset of Impulso')


    def prepare(self):
        logger.info('Begin prepare of Impuslo')
        preparer = Preparer(self.args.exec_type, self.hparams, self.args.data_id)
        logger.info('EXPERIMENT-ID: ' + preparer.hparams[self.args.exec_type]['experiment_id'])
        logger.info('End prepare of Impulso')


    def train(self):
        logger.info('Begin train of Impulso')
        trainer = Trainer(self.args.exec_type, self.hparams, self.model, self.args.model_id)
        trainer.load_data()
        trainer.append_callbacks()
        trainer.begin_train()
        logger.info('End train of Impulso')


    def estimate(self):
        logger.info('Begin estimate of Impulso')
        print('BEGIN: ESTIMATE')
        estimator = Estimator(self.args.exec_type, self.hparams, self.model, self.args.x_dir, self.args.y_dir)
        estimator.load_data()
        estimator.estimate()
        estimator.save_results()
        logger.info('End estimate of Impulso')


    def evaluate(self):
        logger.info('Begin test of Impulso')
        evaluator = Evaluator(self.args.exec_type, self.hparams)
        evaluator.load_data()
        evaluator.evaluate()
        logger.info('End test of Impulso')


    def load_model(self):
        logger.info('Load model')
        modeler = SimpleNet('common', self.hparams)
        modeler.create_model()
        modeler.select_optimazer()
        modeler.compile()

        if self.args.experiment_id and self.args.model_id: 
            weights = glob.glob(os.path.join(IMPULSO_HOME, 'experiments', self.args.experiment_id, 'weights', '*'))
            while weights:
                weight = weights.pop(0)
                i_weight = int(os.path.basename(weight).split('.')[1].split('-')[0])
                if self.args.model_id == i_weight:
                    logger.info('Load weight: ' + weight)
                    self.hparams[self.args.exec_type]['weight'] = weight
                    modeler.model.load_weights(weight)

        self.model = modeler.model
        

if __name__ == '__main__':

    logger.info('Prepare arguments.')
    parser = argparse.ArgumentParser()
    parser.add_argument('exec_type',
                        help='Execution type',
                        nargs=None,
                        default=None,
                        type=str,
                        choices=['dataset', 'prepare', 'train', 'test', 'predict'])
    parser.add_argument('-d', '--data_id',
                        help='Dataset ID',
                        nargs=None,
                        default=None,
                        type=str)
    parser.add_argument('-e', '--experiment_id',
                        help='Experiment ID',
                        nargs=None,
                        default=None,
                        type=str)
    parser.add_argument('-m', '--model_id',
                        help='The number of model',
                        nargs=None,
                        default=None,
                        type=int)
    parser.add_argument('-n', '--n_core',
                        help='The number of CPU core',
                        nargs=None,
                        default=1,
                        type=int)
    parser.add_argument('-x', '--x_dir',
                        help='Path to input data directory',
                        nargs=None,
                        default=None,
                        type=str)
    parser.add_argument('-y', '--y_dir',
                        help='Path to output data directory',
                        nargs=None,
                        default=None,
                        type=str)
    parser.add_argument('-t', '--t_dir',
                        help='Path to ground truth data directory',
                        nargs=None,
                        default=None,
                        type=str)
    args = parser.parse_args()

    logger.info('Check args')
    if args.exec_type == 'prepare':
        assert args.data_id, 'DATA-ID must be specified.'
    elif args.exec_type == 'train':
        assert args.experiment_id, 'EXPERIMENT-ID must be specified.'
    elif args.exec_type == 'test':
        assert args.experiment_id, 'EXPERIMENT-ID must be specified.'
        assert args.model_id, 'MODEL-ID must be specified.'
    logger.info(args)

    """
    logger.info('Set source path.')
    if args.exec_type in ['dataset', 'prepare']:
        sys.path.append(os.path.join(IMPULSO_HOME, 'src'))
    elif args.exec_type in ['train', 'validate', 'test', 'predict']:
        sys.path.append(os.path.join(IMPULSO_HOME, 'experiment_id'))
    """
    
    logger.info('Begin main processes.')
    impulso = Impulso(args, hparams_yaml='hparams.yaml')

    if args.exec_type == 'dataset':
        impulso.dataset()

    elif args.exec_type == 'prepare':
        impulso.prepare()

    elif args.exec_type == 'train':
        impulso.load_model()
        impulso.train()

    # Keras can split train data into train and validation data,
    # So, validation step is not necessary.
    elif args.exec_type == 'validate':
        pass
        #impulso.load_model()
        #impulso.estimate()
        #impulso.evaluate()

    elif args.exec_type == 'test':
        impulso.load_model()
        impulso.estimate()
        impulso.evaluate()

    elif args.exec_type == 'predict':
        impulso.load_model()
        impulso.estimate()
    
    else:
        pass

    logger.info('Finish main processes.')

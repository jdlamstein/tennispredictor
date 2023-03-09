"""
Train model
https://towardsdatascience.com/making-big-bucks-with-a-data-driven-sports-betting-strategy-6c21a6869171
"""

from models.model import Model
from preprocessing.pipeline import Dataspring
from param_tennis import Param
import torch
import os
import argparse
import pyfiglet
import wandb
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
__author__='Josh Lamstein'

class Deploy:
    def __init__(self, p, ckpt_path, csv):
        self.p = p
        self.ckpt_path = ckpt_path
        self.csv = csv

    def deploy(self):
        wandb_logger = WandbLogger()
        wandb_logger.experiment.config['timestring'] = self.p.timestring
        wandb_logger.experiment.config['learning_rate'] = self.p.learning_rate  # todo: cosine annealing
        wandb_logger.experiment.config['optimizer'] = self.p.optimizer
        wandb_logger.experiment.config['epochs'] = self.p.epochs
        wandb_logger.experiment.config['batch_size'] = self.p.batch_size
        Dat = Dataspring(self.csv)
        # savepath = os.path.join(self.p.model_dir, 'tennis_mlp_' + self.p.timestring + '.h5')
        dataset_train, dataset_val, dataset_test = Dat.build_dataset_with_labels()
        _model = Model(self.p.learning_rate)
        model= _model.load_from_checkpoint(self.ckpt_path)
        test_loader = DataLoader(dataset_test, batch_size=self.p.batch_size)
        model.eval()
        trainer = pl.Trainer(accelerator='gpu', devices=1,
                             logger=wandb_logger,
                             max_epochs=self.p.epochs,
                             default_root_dir=self.p.model_dir)
        trainer.test(model, test_loader)

    # def get_feature_importance(self, j, n):
    #     s = accuracy_score(y_test, y_pred)  # baseline score
    #     total = 0.0
    #     for i in range(n):
    #         perm = np.random.permutation(range(X_test.shape[0]))
    #         X_test_ = X_test.copy()
    #         X_test_[:, j] = X_test[perm, j]
    #         y_pred_ = clf.predict(X_test_)
    #         s_ij = accuracy_score(y_test, y_pred_)
    #         total += s_ij
    #     return s - total / n


if __name__ == '__main__':
    result = pyfiglet.figlet_format("Tennis Deploy", font="slant")
    print(result)
    csv = r'D:\Data\Sports\tennis\tennis_data\atp_database.csv'
    ckpt_path = r'D:\Data\Sports\tennis\models\tennis-main\2022_08_15_10_15_18\epoch=3-step=3312.ckpt'
    parser = argparse.ArgumentParser("Tennis Bets")
    parser.add_argument('--csv', action="store",
                        default=csv,
                        help='processed data csv',
                        dest='csv')
    parser.add_argument('--ckpt_path', action="store",
                        default=ckpt_path,
                        help='processed data csv',
                        dest='ckpt_path')

    args = parser.parse_args()
    print('ARGS:\n', args)
    Dep = Deploy(Param(None), args.model_path, args.csv)
    Dep.deploy()

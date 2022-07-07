"""
Train model
https://towardsdatascience.com/making-big-bucks-with-a-data-driven-sports-betting-strategy-6c21a6869171
"""

from models.model import Model
from preprocessing.pipeline import Dataspring
import param_tennis as param
import os
import argparse
import pyfiglet
import wandb
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger


class Train:
    def __init__(self, csv, p):
        self.p = p
        self.csv = csv

    def train(self):
        wandb_logger = WandbLogger()
        Dat = Dataspring(self.csv)
        # savepath = os.path.join(self.p.model_dir, 'tennis_mlp_' + self.p.timestring + '.h5')
        dataset_train, dataset_val, dataset_test = Dat.build_dataset_with_labels()
        model = Model(self.p.learning_rate)
        train_loader = DataLoader(dataset_train, batch_size=self.p.batch_size, shuffle=True)
        val_loader = DataLoader(dataset_val, batch_size=self.p.batch_size)
        test_loader = DataLoader(dataset_test, batch_size=self.p.batch_size)
        early_stop_callback = EarlyStopping(monitor="val_acc", min_delta=0.00, patience=4, verbose=False,
                                            mode="max")

        trainer = pl.Trainer(accelerator='gpu', devices=1,
                             logger=wandb_logger,
                             max_epochs = self.p.epochs,
                             default_root_dir=self.p.model_dir,
                             callbacks=[early_stop_callback])
        trainer.fit(model, train_loader, val_loader)
        trainer.test(model, test_loader)


if __name__ == '__main__':
    result = pyfiglet.figlet_format("Tennis Train", font="slant")
    print(result)
    csv = r'D:\Data\Sports\tennis\tennis_data\atp_database.csv'

    parser = argparse.ArgumentParser("Tennis Bets")
    parser.add_argument('--csv', action="store",
                        default=csv,
                        help='processed data csv',
                        dest='csv')

    args = parser.parse_args()
    print('ARGS:\n', args)
    Tr = Train(args.csv, param.Param(None))
    Tr.train()
    # todo: set gridsearch, hyperparam optimization
    # todo: linear regression
    # todo: incorportate odds, should bet and how much
    # todo: transformer for win percentage?
    # todo: data tests

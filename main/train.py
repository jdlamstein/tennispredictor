"""
Train model
"""
from pytorch_lightning.callbacks import ModelCheckpoint

from models.model import Model
from preprocessing.pipeline import Dataspring
from param_tennis import Param
import os
import argparse
import pyfiglet
import wandb
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

__author__='Josh Lamstein'
class Train:
    def __init__(self, p, csv):
        self.p = p
        self.csv = csv

    def train(self):
        wandb_logger = WandbLogger()
        wandb_logger.experiment.config['timestring'] = self.p.timestring
        wandb_logger.experiment.config['learning_rate'] = self.p.learning_rate  # todo: cosine annealing
        wandb_logger.experiment.config['optimizer'] = self.p.optimizer
        wandb_logger.experiment.config['epochs'] = self.p.epochs
        wandb_logger.experiment.config['batch_size'] = self.p.batch_size
        Dat = Dataspring(self.p, self.csv)
        # savepath = os.path.join(self.p.model_dir, 'tennis_mlp_' + self.p.timestring + '.h5')
        dataset_train, dataset_val, dataset_test = Dat.build_dataset_with_labels()
        model = Model(self.p.learning_rate)
        train_loader = DataLoader(dataset_train, batch_size=self.p.batch_size, shuffle=True)
        val_loader = DataLoader(dataset_val, batch_size=self.p.batch_size)
        test_loader = DataLoader(dataset_test, batch_size=self.p.batch_size)
        early_stop_callback = EarlyStopping(monitor="val_acc", min_delta=0.01, patience=4, verbose=False,
                                            mode="max")
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(self.p.model_dir, 'tennis-main', self.p.timestring),
                                              save_top_k=3, monitor="val_loss")

        trainer = pl.Trainer(accelerator='gpu', devices=1,
                             logger=wandb_logger,
                             max_epochs=self.p.epochs,
                             default_root_dir=self.p.model_dir,
                             callbacks=[early_stop_callback, checkpoint_callback])
        trainer.fit(model, train_loader, val_loader)
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
    result = pyfiglet.figlet_format("Tennis Train", font="slant")
    print(result)
    csv = '/Users/gandalf/Data/tennis/tennis_data/atp_database.csv'

    parser = argparse.ArgumentParser("Tennis Bets")
    parser.add_argument('--csv', action="store",
                        default=csv,
                        help='processed data csv',
                        dest='csv')
    parser.add_argument('--rootdir', default='/Users/gandalf/Data/tennis',
                        help='Parent directory for tennis analysis')

    args = parser.parse_args()
    print('ARGS:\n', args)
    Tr = Train(Param(args.rootdir), args.csv)
    Tr.train()
    # todo: incorporate odds, should bet and how much
    # todo: data tests

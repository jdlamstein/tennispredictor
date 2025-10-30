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

    def deploy(self, output_probabilities=True):
        wandb_logger = WandbLogger()
        wandb_logger.experiment.config['timestring'] = self.p.timestring
        wandb_logger.experiment.config['learning_rate'] = self.p.learning_rate  # todo: cosine annealing
        wandb_logger.experiment.config['optimizer'] = self.p.optimizer
        wandb_logger.experiment.config['epochs'] = self.p.epochs
        wandb_logger.experiment.config['batch_size'] = self.p.batch_size
        Dat = Dataspring(self.p, self.csv)

        # Check if this is a deploy CSV (no labels) or test set
        if 'deploy' in self.csv:
            Dat.load_metadata(os.path.join(self.p.resources_dir, 'meta.csv'))
            feats_deploy, lbls, player_names = Dat.prepare_dataset_deploy()
            _model = Model(self.p.learning_rate)
            model = _model.load_from_checkpoint(self.ckpt_path)
            model.eval()

            # Generate predictions with probabilities
            with torch.no_grad():
                feats_tensor = torch.Tensor(feats_deploy)
                log_probs = model(feats_tensor)
                probs = torch.exp(log_probs).numpy()
                preds = torch.argmax(log_probs, dim=1).numpy()

            results = []
            for i in range(len(preds)):
                result = {
                    'player1_name': player_names.player1_name.iloc[i],
                    'player2_name': player_names.player2_name.iloc[i],
                    'predicted_winner': int(preds[i]) + 1,
                    'player1_win_prob': float(probs[i][0]),
                    'player2_win_prob': float(probs[i][1])
                }
                results.append(result)
                print(f'{result["player1_name"]} vs {result["player2_name"]} - '
                      f'Winner: Player {result["predicted_winner"]} - '
                      f'Probabilities: P1={result["player1_win_prob"]:.3f}, P2={result["player2_win_prob"]:.3f}')

            # Save predictions
            import pandas as pd
            results_df = pd.DataFrame(results)
            output_path = os.path.join(self.p.data_dir, f'nn_predictions_{self.p.timestring}.csv')
            results_df.to_csv(output_path, index=False)
            print(f'\nSaved neural network predictions to {output_path}')

            return results_df
        else:
            # Original test set evaluation
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

if __name__ == '__main__':
    result = pyfiglet.figlet_format("Tennis Deploy", font="slant")
    print(result)
    # csv = r'D:\Data\Sports\tennis\tennis_data\atp_database.csv'
    csv = '/Users/gandalf/Data/tennis/tennis_data/deploy.csv'
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
    parser.add_argument('--rootdir', default='/Users/gandalf/Data/tennis',
                        help='Parent directory for tennis analysis')

    args = parser.parse_args()
    print('ARGS:\n', args)
    Dep = Deploy(Param(args.rootdir), args.ckpt_path, args.csv)
    Dep.deploy()

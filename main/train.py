"""
Train model
https://towardsdatascience.com/making-big-bucks-with-a-data-driven-sports-betting-strategy-6c21a6869171
"""

from models.model import Net
from preprocessing.pipeline import Dataspring
import param_tennis as param
import os
import argparse
import pyfiglet
import wandb


class Train:
    def __init__(self, csv, p):
        self.p = p
        self.csv = csv

    def learn_from_dict(self):
        Dat = Dataspring(self.csv)
        savepath = os.path.join(self.p.model_dir, 'tennis_mlp_' + self.p.timestring + '.h5')
        feats_train, feats_val, feats_test = Dat.dict_to_ds_with_labels()
        nn = Net(self.p)
        model = nn.mlp_from_dict(feats_train)

        model.fit(x=feats_train, y=Dat.labels_train, validation_data=(feats_val, Dat.labels_val),
                  batch_size=self.p.batch_size, epochs=self.p.epochs)
        if not os.path.exists(self.p.model_dir):
            os.makedirs(self.p.model_dir)
        model.save(savepath)

    # def learn_from_pd(self):
    #     Dat = Dataspring(self.csv)
    #     Dat.pandas_to_ds()
    #     nn = Net()
    #     model = nn.mlp_from_dict(feats_train)
    #
    #     model.fit(x=feats_train, y=Dat.label  s_train, epochs=self.p.epochs)


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
    Tr.learn_from_dict()
    # todo: set gridsearch, hyperparam optimization
    # todo: switch to pytorch
    # todo: linear regression
    # todo: incorportate odds, should bet and how much
    # todo: transformer for win percentage?
    # todo: data tests
    # todo: record on wandb

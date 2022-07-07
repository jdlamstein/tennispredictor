import tensorflow as tf
from preprocessing.pipeline import Dataspring
import param_tennis as param
import os
import numpy as np


class Deploy:
    def __init__(self, csv, loadpath):
        self.p = param.Param()
        self.deploy_csv = csv  # deploy csv of tournament
        self.model = tf.keras.models.load_model(loadpath)

    def learn_from_dict(self):
        Dat = Dataspring(self.deploy_csv)
        savepath = os.path.join(self.p.model_dir, 'tennis_mlp_' + self.p.timestring + '.h5')
        # feats_deploy,dummy_labels = Dat.dict_to_ds_deploy()
        feats_train, feats_val, feats_test = Dat.dict_to_ds_with_labels()
        feats_target = feats_test
        gt = Dat.labels_test.to_numpy()
        res = self.model.predict(x=feats_target)
        pred = np.reshape(np.round(np.array(res)), (-1,))
        pairings = pred == gt
        accuracy = np.mean(pairings)
        print('accuracy', accuracy)

    # def learn_from_pd(self):
    #     Dat = Dataspring(self.csv)
    #     Dat.pandas_to_ds()
    #     nn = Net()
    #     model = nn.mlp_from_dict(feats_train)
    #
    #     model.fit(x=feats_train, y=Dat.label  s_train, epochs=self.p.epochs)


if __name__ == '__main__':
    p = param.Param()
    csv = r'D:\Data\Sports\tennis\tennis_data\atp_database.csv'
    model = 'tennis_mlp_2021_07_09_21_51_58.h5'
    loadpath = os.path.join(p.model_dir, model)
    Tr = Deploy(csv, loadpath)
    Tr.learn_from_dict()

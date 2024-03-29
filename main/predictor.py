"""
Use random forest, adaboost, and other classifiers to predict data
"""
import pandas as pd
import param_tennis as param
import glob
import datetime
import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA
from preprocessing.pipeline import Dataspring
import matplotlib.pyplot as plt
import argparse
from param_tennis import Param

__author__='Josh Lamstein'
# matplotlib.interactive(False)
# plt = matplotlib.pyplot


class Predictor:
    def __init__(self, data, labels):

        self.classifiers = {"Nearest Neighbors": KNeighborsClassifier(3),
                            # "Linear SVM": SVC(kernel="linear", C=0.025),
                            # "RBF SVM": SVC(gamma=2, C=1),
                            # "Gaussian Process": GaussianProcessClassifier(1.0 * RBF(1.0)),
                            "Decision Tree": DecisionTreeClassifier(max_depth=100),
                            "Random Forest": RandomForestClassifier(max_depth=None, n_estimators=1000, random_state=42),
                            "Neural Net": MLPClassifier(alpha=1E-4, max_iter=4000),
                            "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
                            "Naive Bayes": GaussianNB(),
                            "QDA": QuadraticDiscriminantAnalysis()}

        # self.names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
        #               "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
        #               "Naive Bayes", "QDA"]
        #
        # self.classifiers = [
        #     KNeighborsClassifier(3),
        #     SVC(kernel="linear", C=0.025),
        #     SVC(gamma=2, C=1),
        #     GaussianProcessClassifier(1.0 * RBF(1.0)),
        #     DecisionTreeClassifier(max_depth=100),
        #     RandomForestClassifier(max_depth=None, n_estimators=1000, random_state=42),
        #     MLPClassifier(alpha=1E-4, max_iter=4000),
        #     AdaBoostClassifier(n_estimators=100, random_state=42),
        #     GaussianNB(),
        #     QuadraticDiscriminantAnalysis()]

        self.X = data  # pd.DataFrame
        self.y = labels  # np array

    def split_data_df(self, seed, train_size):
        X_train = self.X.sample(frac=train_size, random_state=seed)
        X_test = self.X.drop(X_train.index)
        y_train = self.y.loc[X_train.index]
        y_test = self.y.drop(X_train.index)
        return X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()
        # return X_train, X_test, y_train, y_test

    def split_data_np(self, seed, test_size):
        self.X = StandardScaler().fit_transform(self.X)

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=seed)

        return X_train, y_train, X_test, y_test

    def prediction(self, X_train, y_train, X_test, y_test, save_dir=None, save_bool=False):
        # for name, clf in zip(self.names, self.classifiers):
        for name, clf in self.classifiers.items():
            importances = []
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            y_pred = clf.predict(X_test)
            if (name == 'Random Forest') or (name == 'AdaBoost'):
                importances = clf.feature_importances_

            cm = confusion_matrix(y_test, y_pred)
            if save_dir is not None and save_bool:
                np.savez(os.path.join(save_dir, name + '.npz'), cm=cm, y_pred=y_pred, y_true=y_test, score=score,
                         importances=importances)
                with open(os.path.join(save_dir, name + '.pkl'), 'wb') as handle:
                    pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(save_dir, 'pred_scores.txt'), 'a') as fh:
                print(name, file=fh)
                print(score, file=fh)
                print(cm, file=fh)
            print(name)
            print(score)
            print(cm)
            if len(importances) > 0 and 0:

                std = np.std([tree.feature_importances_ for tree in clf.estimators_],
                             axis=0)
                indices = np.argsort(importances)[::-1]

                # Print the feature ranking
                print("Feature ranking:")

                for f in range(X_train.shape[1]):
                    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

                # Plot the feature importances of the forest
                plt.figure()
                plt.title("Feature importances")
                plt.bar(range(X_train.shape[1]), importances[indices],
                        color="r", yerr=std[indices], align="center")
                plt.xticks(range(X_train.shape[1]), indices, rotation='vertical')
                plt.xlim([-1, X_train.shape[1]])
                plt.margins(0.2)
                plt.subplots_adjust(bottom=0.15)
                plt.savefig(os.path.join(save_dir, name + '.png'))

                # plt.show()

    def apply_pca(self):
        redflag = False
        blueflag = False
        pca = PCA(n_components=2)
        pca_X = pca.fit_transform(self.X)
        print(np.shape(pca_X))
        # pca_X_train = pca.fit_transform(X_train)
        # pca_X_test = pca.fit_transform(X_test)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Principal Component 1', fontsize=15)
        ax.set_ylabel('Principal Component 2', fontsize=15)
        ax.set_title('2 Component PCA', fontsize=20)

        colors = ['b', 'r']
        for jdx, target in enumerate(self.y):
            if target == 0:
                color = 'b'
            else:  # winner
                color = 'r'
            line = ax.scatter(pca_X[jdx, 0], pca_X[jdx, 1], c=color, s=20)
            if color == 'r':
                red_line = line
            if color == 'b':
                blue_line = line
        ax.legend((blue_line, red_line), ('Loser', 'Winner'))
        ax.grid()
        plt.show()

        return pca_X


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='/Users/gandalf/Data/tennis/tennis_data/atp_database.csv',
                        help='Input csv generated from clean_data.py for training and analysis.')
    parser.add_argument('--parentdir', default='/Users/gandalf/Data/tennis', help='Save directory')
    parser.add_argument('--savedir', default='/Users/gandalf/Data/tennis/models/classifiers', help='Save directory')
    args = parser.parse_args()
    print(f"Arguments: {args}")
    DEBUG = False
    save_bool = True
    train_size = 0.8
    seed = 200
    Dat = Dataspring(Param(args.parentdir, props=None), args.csv)
    df_train, df_val, df_test, labels_train, labels_val, labels_test = Dat.prepare_dataset()
    if DEBUG:
        df_train = df_train.iloc[:1000]
        labels_train = labels_train.iloc[:1000]

    Pred = Predictor(df_train, labels_train)
    # Pred.apply_pca()

    X_train, y_train, X_test, y_test = Pred.split_data_np(seed, test_size=1 - train_size)
    Pred.prediction(X_train, y_train, X_test, y_test, args.savedir, save_bool)

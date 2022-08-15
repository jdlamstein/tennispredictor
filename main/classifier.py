"""Linear regression"""

import time
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from preprocessing.pipeline import Dataspring
import pyfiglet
import argparse
from param_tennis import Param
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
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
from sklearn.inspection import permutation_importance
from joblib import dump, load

class Classifier:
    def __init__(self, p, csv):
        self.p = p
        self.csv = csv

    def linreg(self):

        Dat = Dataspring(self.csv)
        feats_train, feats_val, feats_test, labels_train, labels_val, labels_test = Dat.prepare_dataset()
        # savepath = os.path.join(self.p.model_dir, 'tennis_mlp_' + self.p.timestring + '.h5')
        regr = LinearRegression()
        regr.fit(feats_train, labels_train)
        print("Coefficients: \n", regr.coef_)

        pred_val = np.round(regr.predict(feats_val))
        pred_test = np.round(regr.predict(feats_test))
        val_acc = np.sum(pred_val * labels_val) / len(labels_val)
        test_acc = np.sum(pred_test * labels_test) / len(labels_test)
        print("Mean squared error: %.2f" % mean_squared_error(labels_test, pred_test))
        print("Coefficient of determination: %.2f" % r2_score(labels_test, pred_test))

        print('Val accuracy: ', val_acc)
        print('Test accuracy: ', test_acc)

    def pca(self):
        Dat = Dataspring(self.csv)
        feats_train, feats_val, feats_test, labels_train, labels_val, labels_test = Dat.prepare_dataset()
        fig = plt.figure(1, figsize=(4, 3))
        plt.clf()

        X = feats_train
        y = labels_train

        plt.cla()

        pca = PCA(n_components=10)
        pca.fit(X)
        X = pca.transform(X)
        print('Explained variance ratio: ', pca.explained_variance_ratio_)
        print('Singular values: ', pca.singular_values_)
        # plt.scatter(X[:, 0], X[:, 1], c='blue', edgecolor='k')
        # plt.show()

    def classifiers(self, train_size=None):
        res = {}
        Dat = Dataspring(self.csv)
        feats_train, feats_val, feats_test, labels_train, labels_val, labels_test = Dat.prepare_dataset()
        if train_size is not None:
            print(f'Setting train length to: {train_size}')
        feats_train = feats_train[:train_size]
        labels_train = labels_train[:train_size]
        feats_test = feats_test[:train_size]
        labels_test = labels_test[:train_size]
        names = [
            "Nearest Neighbors",
            "Linear SVM",
            # "RBF SVM",
            "Gaussian Process",
            "Decision Tree",
            "Random Forest",
            "Neural Net",
            "AdaBoost",
            "Naive Bayes",
            "QDA",
        ]

        classifiers = [
            KNeighborsClassifier(3),
            SVC(kernel="linear", C=0.025),
            # SVC(gamma=2, C=1),
            GaussianProcessClassifier(1.0 * RBF(1.0)),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            MLPClassifier(alpha=1, max_iter=1000),
            AdaBoostClassifier(),
            GaussianNB(),
            QuadraticDiscriminantAnalysis(),
        ]
        # iterate over classifiers
        feature_names = Dat.columns

        for name, clf in zip(names, classifiers):
            # ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            start_time = time.time()
            clf.fit(feats_train, labels_train)
            # score_val = clf.score(feats_val, labels_val)
            score_test = clf.score(feats_test, labels_test)
            # print(f'val: {name} {score_val}')
            print(f'test: {name} {score_test}')
            res[name] = [score_test]
            elapsed_time = time.time() - start_time
            print(f"Elapsed time to compute fit and test: {elapsed_time:.3f} seconds")
            nm = name.replace(' ', '')
            savename = os.path.join(self.p.model_dir, 'classifiers', self.p.timestring, f'{nm}.joblib')
            savedir = os.path.join(self.p.model_dir, 'classifiers', self.p.timestring)
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            dump(clf, savename)
            print(f'Saved {name} to {savename}')
            start_import_time = time.time()
            result = permutation_importance(
                clf, feats_test, labels_test, n_repeats=5, random_state=42, n_jobs=3
            )
            # importances = clf.feature_importances_
            # std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
            elapsed_time = time.time() - start_import_time
            print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
            forest_importances = pd.Series(result.importances_mean, index=feature_names)

            fig, ax = plt.subplots()
            forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
            ax.set_title(f"{name}: Feature importances using MDI")
            ax.set_ylabel("Mean decrease in impurity")
            fig.tight_layout()
            plt.savefig(os.path.join(self.p.fig_dir, f'{name}_importance.tif'))
        pd.DataFrame(res).to_csv(os.path.join(self.p.model_dir, 'classifiers.csv'))


if __name__ == '__main__':
    program_name = pyfiglet.figlet_format("Tennis Classifier", font="slant")
    print(program_name)
    csv = r'D:\Data\Sports\tennis\tennis_data\atp_database.csv'

    parser = argparse.ArgumentParser("Tennis Bets")
    parser.add_argument('--csv', action="store",
                        default=csv,
                        help='processed data csv',
                        dest='csv')

    args = parser.parse_args()
    print('ARGS:\n', args)
    Tr = Classifier(Param(None), args.csv)
    # Tr.pca()
    Tr.classifiers(1000)

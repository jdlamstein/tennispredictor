"""Linear regression"""

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfiglet
from joblib import dump, load
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from param_tennis import Param
from preprocessing.pipeline import Dataspring

__author__ = 'Josh Lamstein'


class Classifier:
    def __init__(self, p, csv):
        self.p = p
        self.csv = csv

    def linreg(self):
        """Run linear regression on the dataset"""

        Dat = Dataspring(self.p, self.csv)
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
        Dat = Dataspring(self.p, self.csv)
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
            # "Gaussian Process",
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
            # GaussianProcessClassifier(1.0 * RBF(1.0)),
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
            scoring = ['r2', 'neg_mean_absolute_percentage_error', 'neg_mean_squared_error']
            r_multi = permutation_importance(
                clf, feats_test, labels_test, n_repeats=5, random_state=42, n_jobs=3, scoring=scoring
            )
            # importances = clf.feature_importances_
            # std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
            elapsed_time = time.time() - start_import_time
            print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
            for metric in r_multi:
                result = r_multi[metric]
                forest_importances = pd.Series(result.importances_mean, index=feature_names)

                fig, ax = plt.subplots()
                forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
                ax.set_title(f"Feature Importances using Permutation on {name}")
                ax.set_ylabel("Mean Accuracy Decrease")
                fig.tight_layout()
                plt.savefig(os.path.join(self.p.fig_dir, f'{name}_{metric}_importance.png'))

        pd.DataFrame(res).to_csv(os.path.join(self.p.model_dir, 'classifiers.csv'))
        return self.p.timestring

    def predictor(self, deploy_csv, meta_csv=None, classifier_timestring=None, classifier_name=None):
        """Run classifier on new game or test set"""
        # Load data for inference
        player_names = None
        if deploy_csv is None:
            Dat = Dataspring(self.p, self.csv)
            feats_train, feats_val, feats_test, labels_train, labels_val, labels_test = Dat.prepare_dataset()
        else:
            Dat = Dataspring(self.p, deploy_csv)
            Dat.load_metadata(meta_csv)
            feats_deploy, lbls, player_names = Dat.prepare_dataset_deploy()
            feats_test = feats_deploy
            labels_test = lbls

        # Load target classifier
        clf = load(os.path.join(self.p.model_dir, 'classifiers', classifier_timestring, f'{classifier_name}.joblib'))
        if labels_test is not None:
            score_test = clf.score(feats_test, labels_test)
            print(f'Predict Score: {classifier_name} {score_test}')
        preds = clf.predict(feats_test)
        for i, pred in enumerate(preds):
            print(f'{player_names.player1_name.iloc[i]} vs {player_names.player2_name.iloc[i]} : {pred}')
        print(f'predictions: {preds}')

        return preds


if __name__ == '__main__':
    program_name = pyfiglet.figlet_format("Tennis Classifier", font="slant")
    print(program_name)
    parser = argparse.ArgumentParser()
    timestring = None
    parser.add_argument('--csv',
                        # default='/Users/gandalf/Data/tennis/tennis_data/deploy.csv',
                        default='/Users/gandalf/Data/tennis/tennis_data/atp_database.csv',
                        help='Input csv generated from clean_data.py for training and analysis.')
    # parser.add_argument('--timestring', default='2023_03_08_13_50_22', help='Input timestring, directory name of classifiers in model folder.')
    parser.add_argument('--timestring', default='',
                        help='Input timestring, directory name of classifiers in model folder.')
    parser.add_argument('--rootdir', default='/Users/gandalf/Data/tennis',
                        help='Parent directory for tennis analysis')
    parser.add_argument('--classifier_name', default='AdaBoost', choices=[
        "Nearest Neighbors",
        "Linear SVM",
        "Gaussian Process",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
        "QDA",
    ],
                        help='Name classifier to run data on.')
    parser.add_argument('--train_length', default=10000, help='Number of samples with which to train classifiers')
    args = parser.parse_args()
    print(f"Arguments: {args}")
    p = Param(rootdir=args.rootdir, props=None)
    meta_csv = os.path.join(p.resources_dir, 'meta.csv')
    Tr = Classifier(p, args.csv)
    if args.timestring is None or len(args.timestring) < 1:
        new_timestring = Tr.classifiers(train_size=args.train_length)
        print(f'Classifiers saved under timestring: {new_timestring}')
    else:
        Tr.predictor(deploy_csv=args.csv, meta_csv=meta_csv, classifier_timestring=args.timestring,
                     classifier_name='AdaBoost')

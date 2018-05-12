import sys
import pandas as pd
import numpy as np
import re
import time
import os
import csv
import tabulate
import matplotlib.pyplot as plt

from sklearn import metrics

from sklearn.naive_bayes import MultinomialNB
from sklearn import svm, neighbors
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier #, RandomizedLasso
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessRegressor 

# from fragile families open source code 
def factorize(df):
    """Convert features of type 'object', e.g. 'string', to categorical
    variables or factors."""
    for col in df.columns:
        if df.loc[:,col].dtype == object:
            factors, values = pd.factorize(df[col])
            df.loc[:,col] = factors
    return df

def read_data():
    print "Reading data"
    train = pd.read_csv('train_2010_2017.csv')
    test = pd.read_csv('test_2018.csv')
    train = factorize(train)

    return train, test # includes factorizing train

def split_xy(train, test):
    
    train = train.drop(['W', 'L', 'T'], axis=1) # only doing this until emily fixes bincount 
    y_train = train['Outcome']
    X_train = train.drop('Outcome', axis=1)
    y_test = test['Outcome']
    X_test = test.drop('Outcome', axis=1)
    return X_train, y_train, X_test, y_test 

def remove_columns(train, test, takeout_vars):
    train = train.drop(takeout_vars, axis=1)
    test = test.drop(takeout_vars, axis=1)
    return train, test

def plot_confusion(cm, y_labels, cmap=plt.cm.Blues, filename='untitled.png'):
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    plt.show()

def bin_test(x_data, x_labels, y_data, y_labels):
    starttime = time.time()

    # binary models
    models = ["SVM_L", "100NN", "LR2", "P2", "HL2", "ADA", "DT", "GP", "ET"] #"SVM_L", "SVM_G", "P2", "DT",  "ADA_R", 
    clfs = [#MultinomialNB(), \
                svm.SVC(kernel = 'linear'), \
                # svm.SVC(kernel='rbf', degree=2), \
                neighbors.KNeighborsClassifier(n_neighbors=100), \
                LogisticRegression(), \
                Perceptron(penalty='l2',tol=None,max_iter=1000), \
                SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True,max_iter=5,tol=None), \
                AdaBoostClassifier(base_estimator=None, n_estimators=100), \
                DecisionTreeClassifier(), \
                # RandomForestClassifier(),  \
                GaussianProcessRegressor(),
                ExtraTreesClassifier() ]


    results = []

    for i in range(len(clfs)):
        print "model being tested: {0}".format(models[i])
        time_start = time.time()
        clf = clfs[i].fit(x_data, x_labels)
        predict = clf.predict(y_data)
        runtime = time.time() - time_start
        a = metrics.accuracy_score(y_labels, predict)
        p = metrics.precision_score(y_labels, predict)
        r = metrics.recall_score(y_labels, predict, average="macro")
        f = metrics.f1_score(y_labels, predict)

        # find outliers
        # data = [('challengeID', y_data['challengeID'].values),
        #         ('predicted', predict),
        #         ('label', y_labels.values)]
        # labels_and_predicted = pd.DataFrame.from_items(data)
        # outliers = y_data.merge(labels_and_predicted, on='challengeID')
        # outliers = outliers[outliers['label'] != outliers['predicted']]

        # num_mislabeled = outliers.shape[0]
        # a_new = -1
        # p_new = -1
        # r_new = -1
        # f_new = -1

        # if (outliers['label'].unique().size > 1):
        #     # train separate model on outliers
        #     mislabeled_labels = outliers['label']
        #     mislabeled_samples = outliers.drop(['label', 'predicted'], axis=1)

        #     (train_vars,validate_vars,train_outcomes,validate_outcomes) = train_test_split(mislabeled_samples,mislabeled_labels,test_size=0.2)

        #     clf_new = clfs[i].fit(train_vars, train_outcomes)
        #     validate_predicted = clf_new.predict(validate_vars)

        #     # evaluate
        #     a_new = metrics.accuracy_score(validate_outcomes, validate_predicted)
        #     p_new = metrics.precision_score(validate_outcomes, validate_predicted)
        #     r_new = metrics.recall_score(validate_outcomes, validate_predicted, average="macro")
        #     f_new = metrics.f1_score(validate_outcomes, validate_predicted)

        # results.append([models[i], a, p, r, f, runtime, num_mislabeled, a_new, p_new, r_new, f_new])
        results.append([models[i], a, p, r, f, runtime])
        # create confusion matrix 
        cm = metrics.confusion_matrix(y_labels, predict)
        plot_confusion(cm, y_labels, filename='{0}_confusion.png'.format(models[i]))
    print tabulate.tabulate(results, headers=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','Runtime'])
    print "Binary test took {0} secs".format(time.time() - starttime)
    results = pd.DataFrame(data=results)
    results.to_csv(PATH+'prediction_results')
    return results

def main(argv):
    PATH = '~/Documents/GitHub/march-madness/'
    train, test = read_data()
    takeout_vars = [u'PomeroyRank', u'Conf', u'AdjEM', u'AdjO',
       u'AdjD', u'AdjT', u'Luck', u'SOSAdjEM', u'OppO', u'OppD', u'NCSOSAdjEM',
       u'MooreRank', u'MooreSOS', u'MoorePR',u'OppPomeroyRank', u'OppConf', u'OppAdjEM', u'OppAdjO', u'OppAdjD',
       u'OppAdjT', u'OppLuck', u'OppSOSAdjEM', u'OppOppO', u'OppOppD',
       u'OppNCSOSAdjEM', u'OppMooreRank', u'OppW', u'OppL', u'OppT',
       u'OppMooreSOS', u'OppMoorePR']
    train, test = remove_columns(train, test, takeout_vars)
    X_train, y_train, X_test, y_test = split_xy(train, test)
    prediction_results = bin_test(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
  main(sys.argv[1:])
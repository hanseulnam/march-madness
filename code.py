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
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn import svm, neighbors
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier #RandomizedLasso
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessRegressor 
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.semi_supervised import LabelPropagation
from sklearn.neural_network import MLPClassifier

# from fragile families open source code 
def factorize(df):
    """Convert features of type 'object', e.g. 'string', to categorical
    variables or factors."""
    for col in df.columns:
        if df.loc[:,col].dtype == object:
            factors, values = pd.factorize(df[col])
            df.loc[:,col] = factors
    return df

def read_data(drop_ext=True):
    print "Reading data"
    ext_data_matchups = ['PomeroyRank', 'Conf', 'AdjEM', 'AdjO', 'AdjD', 'AdjT', 'Luck', 'SOSAdjEM', 'OppO', 'OppD', 'NCSOSAdjEM', 'MooreRank', 'MooreSOS', 'MoorePR', 'OppPomeroyRank', 'OppConf', 'OppAdjEM', 'OppAdjO', 'OppAdjD', 'OppAdjT', 'OppLuck', 'OppSOSAdjEM', 'OppOppO', 'OppOppD', 'OppNCSOSAdjEM', 'OppMooreRank', 'OppMooreSOS', 'OppMoorePR']
    ext_data_team = ['PomeroyRank', 'Conf', 'AdjEM', 'AdjO', 'AdjD', 'AdjT', 'Luck', 'SOSAdjEM', 'OppO', 'OppD', 'NCSOSAdjEM', 'MooreRank', 'MooreSOS', 'MoorePR']
    
    train = pd.read_csv('train_2010_2017.csv')
    test = pd.read_csv('test_2018.csv')
    team_data = pd.read_csv('team_info_2018.csv')
    if drop_ext: 
        train = train.drop(labels=ext_data_matchups, axis=1)
        test = test.drop(labels=ext_data_matchups, axis=1)
        team_data = team_data.drop(labels=ext_data_team, axis=1)

    train = factorize(train)
    test = factorize(test)
    team_data = factorize(team_data)

    teams = pd.read_csv('DataFiles/Teams.csv')

    return train, test, team_data, teams

def split_xy(train, test):
    y_train = train['Outcome']
    X_train = train.drop('Outcome', axis=1)
    y_test = test['Outcome']
    X_test = test.drop('Outcome', axis=1)

    return X_train, y_train, X_test, y_test 

def remove_columns(train, test, takeout_vars):
    train = train.drop(takeout_vars, axis=1)
    test = test.drop(takeout_vars, axis=1)
    return train, test

def validate_previous(train, year):
    test = train[train['Season']==year]
    test = test[test['Round']==1]
    train = train[train['Season']<year]
    X_train, y_train, X_test, y_test = split_xy(train, test)
    bin_test(X_train, y_train, X_test, y_test)

def plot_confusion(cm, y_test, cmap=plt.cm.Blues, filename='untitled.png'):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    plt.savefig('graphs/'+filename)
    # plt.show()
    plt.close()

def bin_test(X_train, y_train, X_test, y_test):
    starttime = time.time()

    # binary models
    models = ["BNB", "GNB", "LDA","SVM_G", "5NN", "LR2", "P2", "SGD","ADA", "DT", "RF", "DPGMM", "ET", "GMM", "MLP"] #"SVM_L", "SVM_G", "P2", "DT",  "ADA_R", 
    clfs = [BernoulliNB(), \
            GaussianNB(), \
            LinearDiscriminantAnalysis(), \
            svm.SVC(kernel='rbf', probability=True), \
            neighbors.KNeighborsClassifier(n_neighbors=5), \
            LogisticRegression(), \
            Perceptron(penalty='l2',tol=None,max_iter=1000), \
            SGDClassifier(tol=0.0001, power_t=0.4, average=True), \
            AdaBoostClassifier(base_estimator=None, n_estimators=100), \
            DecisionTreeClassifier(), \
            RandomForestClassifier(oob_score=True),  \
            BayesianGaussianMixture(n_components=2,max_iter=1000, weight_concentration_prior_type='dirichlet_process', tol=0.0001), \
            ExtraTreesClassifier(bootstrap=True, oob_score=True, n_estimators=4), \
            GaussianMixture(n_components=2, tol=0.0001, max_iter=1000, n_init=2), \
            MLPClassifier(activation='relu', alpha=0.00001, max_iter=1000)]


    results = []
    outlier_results=[]

    for i in range(len(clfs)):
        print "model being tested: {0}".format(models[i])
        time_start = time.time()
        clf = clfs[i].fit(X_train, y_train)
        predict = clf.predict(X_test)
        runtime = time.time() - time_start
        p = metrics.precision_score(y_test, predict)
        r = metrics.recall_score(y_test, predict, average="macro")
        f = metrics.f1_score(y_test, predict)

        # find outliers
        data = [('TeamID', X_test['TeamID'].values),
                ('predicted', predict),
                ('label', y_test.values)]
        labels_and_predicted = pd.DataFrame.from_items(data)
        outliers = X_test.merge(labels_and_predicted, on='TeamID')
        outliers = outliers[outliers['label'] != outliers['predicted']]

        num_mislabeled = outliers.shape[0]

        p_new = -1
        r_new = -1
        f_new = -1

        if (outliers['label'].unique().size > 1):
            # train separate model on outliers
            mislabeled_labels = outliers['label']
            mislabeled_samples = outliers.drop(['label', 'predicted'], axis=1)

            (train_vars,validate_vars,train_outcomes,validate_outcomes) = train_test_split(mislabeled_samples,mislabeled_labels,test_size=0.2)

            clf_new = clfs[i].fit(train_vars, train_outcomes)
            validate_predicted = clf_new.predict(validate_vars)

            # evaluate
            p_new = metrics.precision_score(validate_outcomes, validate_predicted)
            r_new = metrics.recall_score(validate_outcomes, validate_predicted, average="macro")
            f_new = metrics.f1_score(validate_outcomes, validate_predicted)
            outlier_results.append([models[i], p_new, r_new, f_new])

        results.append([models[i], p, r, f, runtime])
        # create confusion matrix 
        cm = metrics.confusion_matrix(y_test, predict)
        plot_confusion(cm, y_test, filename='{0}_confusion.png'.format(models[i]))

    print 
    print "All data models"
    print 
    print tabulate.tabulate(results, headers=['Classif', 'Precision', 'Recall', 'F1 Score','Runtime'])
    print 
    print "Outlier models"
    print 
    print tabulate.tabulate(outlier_results, headers=['Classif', 'Precision', 'Recall', 'F1 Score','Runtime'])
    print "Binary test took {0} secs".format(time.time() - starttime)
    return pd.DataFrame(data=results)

def create_matchups(team_data, pairings, round):
    opp_prefixes = ['Season', 'OppTeamID', 'OppW', 'OppL', 'OppAvgScore', 'OppAvgFGM', 'OppAvgFGA', 'OppAvgFGM3', 'OppAvgFGA3', 'OppAvgFTM', 'OppAvgFTA', 'OppAvgOR', 'OppAvgDR', 'OppAvgAst', 'OppAvgTO', 'OppAvgStl', 'OppAvgBlk', 'OppAvgPF', 'OppAvgOppScore', 'OppAvgOppFGM', 'OppAvgOppFGA', 'OppAvgOppFGM3', 'OppAvgOppFGA3', 'OppAvgOppFTM', 'OppAvgOppFTA', 'OppAvgOppOR', 'OppAvgOppDR', 'OppAvgOppAst', 'OppAvgOppTO', 'OppAvgOppStl', 'OppAvgOppBlk', 'OppAvgOppPF', 'OppSeed']
    
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    
    for p in pairings:        
        team_1 = p[0]
        team_1_data = team_data[(team_data['Season'] == 2018) & (team_data['TeamID'] == team_1)]
        team_1_data_opp = team_1_data.copy()
        team_1_data_opp.columns = opp_prefixes
        
        team_2 = p[1]
        team_2_data = team_data[(team_data['Season'] == 2018) & (team_data['TeamID'] == team_2)]
        team_2_data_opp = team_2_data.copy()
        team_2_data_opp.columns = opp_prefixes
        
        team1_v_team2 = team_1_data.merge(team_2_data_opp, how='outer', on='Season')
        team2_v_team1 = team_2_data.merge(team_1_data_opp, how='outer', on='Season')
        
        df1 = df1.append(team1_v_team2, ignore_index=True)
        df2 = df2.append(team2_v_team1, ignore_index=True)
        
    df = df1.append(df2, ignore_index=True)
    df['Round'] = 1  
    df = df.rename(columns={'Seed': 'TeamSeed', 'OppSeed': 'OppTeamSeed', 'AvgScore': 'AvgPoints', 'AvgOppScore': 'AvgOppPoints', 'OppAvgScore': 'OppAvgPoints'})
    df = df[['Season', 'Round', 'TeamID', 'OppTeamID', 'TeamSeed', 'OppTeamSeed', 'W', 'L', 'AvgPoints', 'AvgFGM', 'AvgFGA', 'AvgFGM3', 'AvgFGA3', 'AvgFTM', 'AvgFTA', 'AvgOR', 'AvgDR', 'AvgAst', 'AvgTO', 'AvgStl', 'AvgBlk', 'AvgPF', 'AvgOppPoints', 'AvgOppFGM', 'AvgOppFGA', 'AvgOppFGM3', 'AvgOppFGA3', 'AvgOppFTM', 'AvgOppFTA', 'AvgOppOR', 'AvgOppDR', 'AvgOppAst', 'AvgOppTO', 'AvgOppStl', 'AvgOppBlk', 'AvgOppPF', 'OppW', 'OppL', 'OppAvgPoints', 'OppAvgFGM', 'OppAvgFGA', 'OppAvgFGM3', 'OppAvgFGA3', 'OppAvgFTM', 'OppAvgFTA', 'OppAvgOR', 'OppAvgDR', 'OppAvgAst', 'OppAvgTO', 'OppAvgStl', 'OppAvgBlk', 'OppAvgPF', 'OppAvgOppScore', 'OppAvgOppFGM', 'OppAvgOppFGA', 'OppAvgOppFGM3', 'OppAvgOppFGA3', 'OppAvgOppFTM', 'OppAvgOppFTA', 'OppAvgOppOR', 'OppAvgOppDR', 'OppAvgOppAst', 'OppAvgOppTO', 'OppAvgOppStl', 'OppAvgOppBlk', 'OppAvgOppPF']]
    
    return df

def winners_to_matchups(winners):
    matchups = []
    for i in xrange(0,len(winners),2):
        team1 = winners[i]
        team2 = winners[i+1]
        matchups.append([team1, team2])
    return matchups

def predict_with_prob(classifier, matchups, team_dict):
    split = len(matchups) / 2
    # get win probabilities
    teams = matchups[['TeamID', 'OppTeamID']]
    win_probs = pd.DataFrame(data=classifier.predict_proba(matchups), columns=['Loss', 'Win'])
    results = pd.concat([teams, win_probs], axis=1)

    # compare predictions for each matchup from each POV
    results_1 = results.iloc[:split]
    results_1.loc[:,'Matchup'] = results_1.index
    results_2 = results.iloc[split:].reset_index()
    results_2.loc[:,'Matchup'] = results_2.index
    results_concat = results_1.join(results_2, on='Matchup', lsuffix='1', rsuffix='2')
    results_concat = results_concat[['TeamID1', 'OppTeamID1', 'Win1', 'Win2']]
    results_concat.columns = ['Team1', 'Team2', 'Win1', 'Win2']
    
    # standardize probabilities
    results_concat['Sum'] = results_concat['Win1'] + results_concat['Win2']
    results_concat['Win1Adj'] = results_concat['Win1'] / results_concat['Sum']
    results_concat['Win2Adj'] = results_concat['Win2'] / results_concat['Sum']

    # make predictions
    results_concat['Team1WinPred'] = np.where(results_concat['Win1Adj'] > results_concat['Win2Adj'], 1, 0)
    
    pred_winners = np.where(results_concat['Team1WinPred'] == 1, results_concat['Team1'], results_concat['Team2'])
    return pred_winners

def predict_bracket(classifier, X_train, y_train, matchups, rounds, team_data, team_dict, output=False):
    results = []
    
    for r in rounds:
        matchups_with_data = create_matchups(team_data, matchups, r)
        winner_ids = predict_with_prob(classifier, matchups_with_data, team_dict)
        results.append(winner_ids)
        winner_names = [team_dict[team_id] for team_id in winner_ids]
        
        if output: 
            print winner_names
            print

        if (r < 6):
            matchups = winners_to_matchups(winner_ids)
    
    return results

def score_bracket_espn(results, prediction, rounds):
    # see http://games.espn.com/tournament-challenge-bracket/2018/en/story?pageName=tcmen%5Chowtoplay
    points = [10, 20, 40, 80, 160, 320]
    total_pts = 0
    
    for rd, pts, pred_winners, act_winners in zip(rounds, points, prediction, results):
        num_correct = 0
        
        for pred, act in zip(pred_winners, act_winners):
            if (pred == act):
                num_correct += 1
        
        rd_pts = pts * num_correct
        total_pts += rd_pts
    
    return total_pts

def baseline_predict_bracket(matchups, rounds): 

    matchups = [[1438,1420], [1166, 1243], [1246, 1172], [1112, 1138], [1274, 1260], [1397, 1460], [1305, 1400], [1153, 1209], [1462, 1411], [1281, 1199], [1326, 1355], [1211, 1422], [1222, 1361], [1276, 1285], [1401, 1344], [1314, 1252], [1437, 1347], [1439, 1104], [1452, 1293], [1455, 1267], [1196, 1382], [1403, 1372], [1116, 1139], [1345, 1168], [1242, 1335], [1371, 1301], [1155, 1308], [1120, 1158], [1395, 1393], [1277, 1137], [1348, 1328], [1181, 1233]]
    results = []


    def higher_seed(matchups, X_test):
        pred_winners = []
        for i in matchups: 
            seed1 = X_test[X_test['TeamID'] == i[0]]['TeamSeed'].values
            seed2 = X_test[X_test['TeamID'] == i[1]]['TeamSeed'].values
            if seed1 < seed2: 
                pred_winners.append(i[0])
            else:
                pred_winners.append(i[1])
        return pred_winners
    
    
    for r in rounds:
        winner_ids = higher_seed(matchups, X_test)
        results.append(winner_ids)
        winner_names = [team_dict[team_id] for team_id in winner_ids]
        print winner_names
        print

        if (r < 6):
            matchups = winners_to_matchups(winner_ids)
    
    return results

def bracket_predict_clfs(X_train, y_train, matchups, rounds, team_data, team_dict, results, latex=False, \
            models=["BNB", "GNB", "LDA","SVM_G", "5NN", "LR2", "P2", "SGD","ADA", "DT", "RF", "DPGMM", "ET", "GMM", "MLP"], \
            clfs = [BernoulliNB(), \
            GaussianNB(), \
            LinearDiscriminantAnalysis(), \
            svm.SVC(kernel='rbf', probability=True), \
            neighbors.KNeighborsClassifier(n_neighbors=5), \
            LogisticRegression(), \
            AdaBoostClassifier(base_estimator=None, n_estimators=100), \
            DecisionTreeClassifier(), \
            RandomForestClassifier(oob_score=True),  \
            BayesianGaussianMixture(n_components=2,max_iter=1000, weight_concentration_prior_type='dirichlet_process', tol=0.0001), \
            ExtraTreesClassifier(bootstrap=True, oob_score=True, n_estimators=4), \
            GaussianMixture(n_components=2, tol=0.0001, max_iter=1000, n_init=2), \
            MLPClassifier(activation='relu', alpha=0.00001, max_iter=1000)]):

    table = []
    for i in range(len(clfs)):
        c = clfs[i]
        c.fit(X_train, y_train)
        predicted_bracket = predict_bracket(c, X_train, y_train, matchups, rounds, team_data, team_dict)
        espn_score = score_bracket_espn(results, predicted_bracket, rounds)
        print espn_score # 1150 pts would have been between 96.0-96.9%
        result = [models[i], espn_score]
        table.append(result)
    print tabulate.tabulate(table, headers=['Classif', 'ESPN score'])
    table = pd.DataFrame(data=table, columns=['Classif', 'ESPN score'])
    table.to_csv('bracket_prediction_ESPN.csv')
    if latex: 
        print "PRINTING LATEX TABLE BELOW"
        print table.to_latex(index=False)
    return table

def make_ensemble_clf(voting='soft', n_jobs=1):
    models = ["BNB", "GNB", "LDA","SVM_L","SVM_G", "5NN", "LR2", "P2", "SGD","ADA", "DT", "RF", "DPGMM", "ET", "GMM", "MLP"] #"SVM_L", "SVM_G", "P2", "DT",  "ADA_R", 
    clfs = [#BernoulliNB(), \
            GaussianNB(), \
            LinearDiscriminantAnalysis(), \
            svm.SVC(kernel='linear', probability=True), \
            svm.SVC(kernel='rbf', probability=True), \
            neighbors.KNeighborsClassifier(n_neighbors=5), \
            LogisticRegression(), \
            AdaBoostClassifier(base_estimator=None, n_estimators=100), \
            DecisionTreeClassifier(), \
            RandomForestClassifier(oob_score=True),  \
            #BayesianGaussianMixture(n_components=2,max_iter=1000, weight_concentration_prior_type='dirichlet_process', tol=0.0001), \
            ExtraTreesClassifier(bootstrap=True, oob_score=True, n_estimators=4), \
            #GaussianMixture(n_components=2, tol=0.0001, max_iter=1000, n_init=2), \
            #MLPClassifier(activation='relu', alpha=0.00001, max_iter=1000)
            ]
    
    estimators=[]
    for i in range(len(clfs)):
        est = (models[i], clfs[i])
        estimators.append(est)

    ensemble = VotingClassifier(estimators, voting=voting, n_jobs=n_jobs)
    return ensemble

def main(argv):

    ###### SET UP + HARD CODED VARIABLES ######
    starttime = time.time()
    PATH = '~/Documents/GitHub/march-madness/'
    train, test, team_data, teams = read_data()
    X_train, y_train, X_test, y_test = split_xy(train, test)
    team_dict = pd.Series(teams.TeamName.values,index=teams.TeamID).to_dict()
    rounds = [1, 2, 3, 4, 5, 6]
    results_2018 = [[1420, 1243, 1246, 1138, 1260, 1397, 1305, 1153, 1462, 1199, 1326, 1211, 1222, 1276, 1401, 1314, 1437, 1104, 1452, 1267, 1196, 1403, 1139, 1345, 1242, 1371, 1155, 1120, 1393, 1277, 1348, 1181],[1243, 1246, 1260, 1305, 1199, 1211, 1276, 1401, 1437, 1452, 1403, 1345, 1242, 1155, 1393, 1181],[1243, 1260, 1199, 1276, 1437, 1403, 1242, 1181],[1260, 1276, 1437, 1242],[1276, 1437],[1437]]
    matchups_2018 = [[1438,1420], [1166, 1243], [1246, 1172], [1112, 1138], [1274, 1260], [1397, 1460], [1305, 1400], [1153, 1209], [1462, 1411], [1281, 1199], [1326, 1355], [1211, 1422], [1222, 1361], [1276, 1285], [1401, 1344], [1314, 1252], [1437, 1347], [1439, 1104], [1452, 1293], [1455, 1267], [1196, 1382], [1403, 1372], [1116, 1139], [1345, 1168], [1242, 1335], [1371, 1301], [1155, 1308], [1120, 1158], [1395, 1393], [1277, 1137], [1348, 1328], [1181, 1233]]

    # ###### FIRST ROUND PREDICTIONS ######
    drop_ext = True
    bin_test(X_train, y_train, X_test, y_test) # includes outlier analysis 
    
    # ###### BRACKET PREDICTIONS FOR VARIETY OF CLASSIFIERS ###### 
    latex=True # if you want to print latex table
    bracket_predict_clfs(X_train, y_train, matchups_2018, rounds, team_data, team_dict, results_2018, latex=latex)

    ###### ENSEMBLE METHOD ######
    latex=True
    models = ['ESB1','ESB2','ESB3', 'ESB4']
    clfs = []
    for i in range(len(models)):
        esb = make_ensemble_clf(voting='soft', n_jobs=i+1)
        clfs.append(esb)
    bracket_predict_clfs(X_train, y_train, matchups_2018, rounds, team_data, team_dict, results_2018, latex=latex, models=models, clfs=clfs)
    
    print "Runtime: {0}".format(time.time() - starttime)

if __name__ == "__main__":
  main(sys.argv[1:])
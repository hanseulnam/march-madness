import numpy as np
import pandas as pd
import os
import seaborn as sns
#import matplotlib.pyplot as plt
os.chdir("/Users/eastonorbe/Desktop")
RegularSeason = pd.read_csv('regular_season.csv', encoding='latin-1')
YearList = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
IDLIST = range(1101, 1465,1)
DayNumLimit = 135
RS = np.asarray(RegularSeason)
print(RS)
RSTMatchup = np.delete(RS, [0,1], axis=1)
RSTMatchupYear = np.delete(RS, [1], axis=1)
RSTTeamsYear = np.delete(RS,[1,16,17,18,19,20,21,22,23,24,25,26,27,28,29], axis = 1)
RSTTeams = np.delete(RS, [0,1,16,17,18,19,20,21,22,23,24,25,26,27,28,29], axis = 1)
print(RSTMatchup.shape)
print(RSTTeams.shape)
KList = [2, 3, 5, 8, 10]

#Match-Up Styles
#Code used in eorbe Assignment #3
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
N = 20
distances = np.zeros((28,3139,N))
indices = np.zeros((3139,N))
mindist = np.zeros((3139,N))
totaldist = np.zeros((N))
for k in range(1,N):
    print(k)
    #RSTS = RSTMatchup[np.where(RSTMatchup[:,0]==y),:]
    RSTS = RSTMatchup
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(RSTS)
    for i in range(0,k): 
        a = kmeans.cluster_centers_[i,:]
        a = np.reshape(a, (1,28))
        for j in range(0,3139):
            b = RSTS[j,] 
            distances[i,j,k] = np.linalg.norm(a-b)
    
        indices[:,k] = np.argmin(distances[0:k,:,k], axis = 0)
        mindist[:,k] = np.min(distances[0:k,:,k], axis = 0)
    
        totaldist[k] = np.sum(mindist[:,k], axis = 0)/3139

totaldist=totaldist[1:N]
print(totaldist)
x = np.zeros((len(totaldist)))
for i in range(0,len(totaldist)):
    x[i] = i+1
plt.scatter(x,totaldist)
plt.plot(x, totaldist, '-o')
plt.title('Total Intra-Cluster Distance: Match-ups')
plt.xlabel('Number of Clusters')
plt.ylabel('Total Intra-Cluster Distance')
plt.show()

#Team Styles
#Code used in eorbe Assignment #3
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
N = 20
distances = np.zeros((28,3139,N))
indices = np.zeros((3139,N))
mindist = np.zeros((3139,N))
totaldist = np.zeros((N))
for k in range(1,N):
    print(k)
    RSTS = RSTTeams
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(RSTS)
    for i in range(0,k): 
        a = kmeans.cluster_centers_[i,:]
        a = np.reshape(a, (1,14))
        for j in range(0,3139):
            b = RSTS[j,] 
            distances[i,j,k] = np.linalg.norm(a-b)
    
        indices[:,k] = np.argmin(distances[0:k,:,k], axis = 0)
        mindist[:,k] = np.min(distances[0:k,:,k], axis = 0)
    
        totaldist[k] = np.sum(mindist[:,k], axis = 0)/3139

totaldist=totaldist[1:N]
print(totaldist)
x = np.zeros((len(totaldist)))
for i in range(0,len(totaldist)):
    x[i] = i+1
plt.scatter(x,totaldist)
plt.plot(x, totaldist, '-o')
plt.title('Total Intra-Cluster Distance: Teams')
plt.xlabel('Number of Clusters')
plt.ylabel('Total Intra-Cluster Distance')
plt.show()

np.where(RS[:,1]==1437)

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

kmeans_matchup = KMeans(n_clusters=4).fit(RSTMatchup)
kmeans_teams = KMeans(n_clusters=5).fit(RSTTeams)

scaler = StandardScaler(copy=True, with_mean=True, with_std=False)
CenteredMatchups = scaler.fit_transform(RSTMatchup)
CenteredTeams = scaler.fit_transform(RSTTeams)

#Matchups
pca = PCA(n_components = 2)
transmatchup = pca.fit_transform(CenteredMatchups)
pca.fit(CenteredMatchups)
print("Matchups", pca.components_)
#Teams
pca = PCA(n_components = 2)
transteams = pca.fit_transform(CenteredTeams)
pca.fit(CenteredTeams)
print("Teams", pca.components_)

#Visualization
color_array_matchup = ['red', 'green', 'blue', 'orange']
fig, ax = plt.subplots(1)
matchuplabels = kmeans_matchup.labels_
cbc = [color_array_matchup[i] for i in matchuplabels]
ax.scatter(transmatchup[:,0], transmatchup[:,1], color=cbc, alpha=0.4)
plt.title("Match-up Style Clusters")
plt.show()

color_array_teams = ['red', 'green', 'blue', 'orange','yellow']
fig, ax = plt.subplots(1)
teamlabels = kmeans_teams.labels_
cbc = [color_array_teams[i] for i in teamlabels]
ax.scatter(transteams[:,0], transteams[:,1], color=cbc, alpha=0.4)
ax.annotate('Villanova2018', xy=(21.8295691005, 8.91363084712), xytext=(21.8295691005, 8.91363084712), horizontalalignment='left', verticalalignment='top')
ax.annotate('Michigan2018', xy=(4.50687870895, 9.41057634619), xytext=(4.50687870895, 9.41057634619), horizontalalignment='right', verticalalignment='top')
ax.annotate('Kansas2018', xy=(16.3546880578, 9.06953582468), xytext=(16.3546880578, 9.06953582468), horizontalalignment='right', verticalalignment='top')
ax.annotate('Loyola2018', xy=(1.8468184172, 2.36264141747), xytext=(1.8468184172, 2.36264141747), horizontalalignment='left', verticalalignment='top')
plt.title("Team Play Style Clusters")
plt.show()

#import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import seaborn as sns

xvals = transteams[:,0]
yvals = transteams[:,1]
teamvals = RS[:,1]

# Create dataframe
df = pd.DataFrame({
'x': xvals,
'y': yvals,
'group': teamvals
})

for x,y,z in zip(xvals, yvals, teamvals):
    if (z == 1437):
        print("Villanova" + "x: " + str(x) + ", y: " + str(y))
        
 
# Create dataframe
df = pd.DataFrame({
'x': xvals,
'y': yvals,
'group': teamvals
})

for x,y,z in zip(xvals, yvals, teamvals):
    if (z == 1276):
        print("Michigan" + "x: " + str(x) + ", y: " + str(y))
        
# Create dataframe
df = pd.DataFrame({
'x': xvals,
'y': yvals,
'group': teamvals
})

for x,y,z in zip(xvals, yvals, teamvals):
    if (z == 1242):
        print("Kansas" + "x: " + str(x) + ", y: " + str(y))
        
        
# Create dataframe
df = pd.DataFrame({
'x': xvals,
'y': yvals,
'group': teamvals
})

for x,y,z in zip(xvals, yvals, teamvals):
    if (z == 1260):
        print("Loyola" + "x: " + str(x) + ", y: " + str(y))

#Matchups
for i in set(matchuplabels):
    cur_label = RSTMatchup[np.where(matchuplabels == i)]
    average_stats_matchup = np.mean(cur_label, axis = 0)
    print("Matchup Style Clusters")
    print(cur_label.shape)
    print(average_stats_matchup.shape)
    print(average_stats_matchup)

arr = []
arr2 = []
#Teams
for i in set(teamlabels):
    cur_label = RSTTeams[np.where(teamlabels == i)]
    arr.append(RS[np.where(teamlabels == i),1])
    arr2.append(RS[np.where(teamlabels == i),0])
    #print(cur_label)
    average_stats_teams = np.mean(cur_label, axis = 0)
    print("Team Styles")
    print(cur_label.shape)
    print(average_stats_teams.shape)
    print(average_stats_teams)

from collections import Counter

Top20Teams = [1112,1120,1153,1155,1181,1211,1242,1246,1276,1277,1314,1326,1345,1397,1403,1437,1438,1452,1455,1462]
Years = [2010,2011,2012,2013,2014,2015,2016,2017,2018]
for cluster in arr:
    counts = dict.fromkeys(Top20Teams, 0)
    cluster_vals = cluster[0]
    for val in cluster_vals:
        if val in Top20Teams:
            counts[val] += 1 
    print(counts)
    
print("Array2")
for cluster in arr2:
    counts2 = dict.fromkeys(Years, 0)
    cluster_vals = cluster[0]
    for val in cluster_vals:
        if val in Years:
            counts2[val] += 1
    print(counts2)
    
#Arizona-1112
#Auburn-1120
#Cincinatti-1153
#Clemson-1155
#Duke-1181
#Gonzaga-1211
#Kansas-1242
#Kentucky-1246
#Michigan-1276 #First Cluster
#Michigan St.-1277
#North Carolina-1314
#Ohio St.-1326
#Purdue-1345
#Tennessee-1397
#Texas Tech-1403
#Villanova-1437
#Virgina-1438
#West Virgina-1452
#Wichita St.-1455
#Xavier-1462


#First Cluster has 17.96
#Second Cluster has 21.59
#Third Cluster has 16.81
#Fourth Cluster has 22.7
#Fifth Cluster has 17.7
{2010: 55, 2011: 49, 2012: 55, 2013: 61, 2014: 43, 2015: 53, 2016: 94, 2017: 112, 2018: 158}
{2010: 24, 2011: 19, 2012: 16, 2013: 10, 2014: 28, 2015: 15, 2016: 88, 2017: 91, 2018: 97}
yaxis = [55+24, 49+19, 55+16, 61+10, 43+28, 53+15, 94+88, 112+91, 158+97]
xaxis = [2010,2011,2012,2013,2014,2015,2016,2017,2018]

plt.scatter(xaxis,yaxis)
plt.plot(xaxis, yaxis, '-o')
plt.title('Number of Teams Shooting More than 20 3s per Game on Average')
plt.xlabel('Season')
plt.ylabel('Number of Teams')
plt.show()


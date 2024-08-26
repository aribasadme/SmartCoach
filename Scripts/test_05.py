# -*- coding: utf-8 -*-
"""
Created on Tue May 12 15:42:30 2015

@author: aRa

PCA para identificar las componentes principales y poder agrupar/separar las
    clases feeling_good de feeling_bad
"""

import glob
import os
import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import dateutil.parser
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from mpl_toolkits.mplot3d import Axes3D


def convert(date):
    dt = dateutil.parser.parse(date).replace(tzinfo=None)
    epoch = datetime.utcfromtimestamp(0)
    delta = dt - epoch
    return delta.total_seconds()

# Generate Datasetes
path = "/Users/aRa/OneDrive/Documentos/ERASMUS/PFC/SmartCoach/Running_w_HR/APE_runs_oct26-mar22"
colnames = np.array(['time', 'elevation', 'distance', 'speed', 'HR'])
window_size_half = 3
Datasets_all = []

os.chdir(path)
for file in glob.glob("*.tab"):
    print "Processing " + file
    
    ds = np.genfromtxt(file, skip_header=1, delimiter='\t', converters={0: convert})
    ds[:,0] -= ds[0,0]                              # time reference set to 0
      
    if (ds[:,2].max() > 11000) == (ds[:,2].max() < 13000):
        ds = pd.DataFrame(ds,columns=colnames)
        
        slope = np.array([])
        dHR = np.array([])
        dspeed = np.array([])
        
        for i in ds.index:
            index = np.arange(i - window_size_half + 1, i + window_size_half + 1)
            index = index[(index >= 0) & (index < len(ds))]
            
            dataset_part = ds[['distance','elevation']].iloc[index].dropna()

            regr = lm.LinearRegression()
            regr.fit(dataset_part.distance[:,np.newaxis], np.array(dataset_part.elevation))
            
            slope = np.append(slope,regr.coef_)
            
            delta_speed = ds.speed.iloc[index].mean()
            dspeed = np.append(dspeed, delta_speed)
            
            delta_HR = (ds.HR.iloc[index[-1]] - ds.HR.iloc[index[0]]) / (190 - ds.HR.iloc[index[0]])
            dHR = np.append(dHR, delta_HR)
            
        dataset_new = ds.drop(['time','elevation','HR'], axis=1)
        dataset_new['distance'] = dataset_new['distance'] / dataset_new['distance'].iloc[-1]
        dataset_new['speed'] = dspeed
        dataset_new['slope'] = slope
        dataset_new['dHR'] = dHR
        Datasets_all.append(dataset_new)

datasets = pd.concat(Datasets_all,ignore_index=True)
datasets.sort(columns='distance', inplace=True)
datasets.index = np.arange(len(datasets))

path = "/Users/aRa/OneDrive/Documentos/ERASMUS/PFC/SmartCoach/Running_w_HR/TEST"

os.chdir(path)
file = 'feeling_bad_activity_726282538.tab'
print "Processing " + file

fb = np.genfromtxt(file, skip_header=1, delimiter='\t', converters={0: convert})
fb[:,0] -= fb[0,0]                              # time reference set to 0      
fb = pd.DataFrame(fb,columns=colnames)

slope_fb = np.array([])
dHR_fb = np.array([])
dspeed_fb = np.array([])

for i in fb.index:
    index = np.arange(i - window_size_half + 1, i + window_size_half + 1)
    index = index[(index >= 0) & (index < len(fb))]
    
    dataset_part = fb[['distance','elevation']].iloc[index].dropna()

    regr = lm.LinearRegression()
    regr.fit(dataset_part.distance[:,np.newaxis], np.array(dataset_part.elevation))
    
    slope_fb = np.append(slope_fb,regr.coef_)
    
    delta_speed_fb = fb.speed.iloc[index].mean()
    dspeed_fb = np.append(dspeed_fb, delta_speed_fb)
    
    delta_HR_fb = (fb.HR.iloc[index[-1]] - fb.HR.iloc[index[0]]) / (190 - fb.HR.iloc[index[0]])
    dHR_fb = np.append(dHR_fb, delta_HR_fb)
    
dataset_fb = fb.drop(['time','elevation','HR'], axis=1)
dataset_fb['distance'] = dataset_fb['distance'] / dataset_fb['distance'].iloc[-1]
dataset_fb['speed'] = dspeed_fb
dataset_fb['slope'] = slope_fb
dataset_fb['dHR'] = dHR_fb

# =================  PCA =================

# Scaling Data
ndata = np.array(datasets.drop(['distance'], axis=1))
scaler = MinMaxScaler().fit(ndata)

ndata = scaler.transform(ndata)
ndata_fb = scaler.transform(dataset_fb.drop(['distance'], axis=1))

pca = PCA(n_components=2)
ndata_r = pca.fit(ndata).transform(ndata)

pca_fb = PCA(n_components=2)
ndata_fb_r = pca_fb.fit(ndata_fb).transform(ndata_fb)

plt.figure()
plt.scatter(ndata_r[:,0], ndata_r[:,1], c='k', label='PCA Datasets', linewidth=0)
plt.scatter(ndata_fb_r[:,0], ndata_fb_r[:,1], c='b', label='PCA Feeling Bad',linewidth=0)
plt.legend(loc='best')

fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(ndata[:, 0], ndata[:, 1], ndata[:, 2], c='rgb',
           cmap=plt.cm.Paired)
ax.set_title("Feeling bad", fontsize=16, y=1.02)
ax.set_xlabel("Speed")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("Slope")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel(r'$\Delta$HR')
ax.w_zaxis.set_ticklabels([])

fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(ndata_fb)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c='rgb',
           cmap=plt.cm.Paired)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

plt.show()


# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 11:33:24 2015

@author: aRa
"""

"""
1. Estimate the race time using a simple statistics: i.e., mean, median speed and not considering slope

"""

import glob
import os
import numpy as np
import pandas as pd
import datetime as dt
import dateutil.parser

import sklearn.linear_model as lm

def convert(date):
    dt = dateutil.parser.parse(date).replace(tzinfo=None)
    epoch = dt.utcfromtimestamp(0)
    delta = dt - epoch
    return delta.total_seconds()

def speed_to_pace(speed):
    miuntes, seconds = divmod(100 / (6 * speed), 1)
    seconds *= 60
    return (miuntes, seconds)
    
path = "/Users/aRa/OneDrive/Documentos/ERASMUS/PFC/SmartCoach/Running_w_HR/APE_runs_oct26-mar22"
colnames = np.array(['time', 'elevation', 'distance', 'speed', 'HR'])
window_size_half = 3
Datasets_all = []

Datasets_all = []
os.chdir(path)
for file in glob.glob("*.tab"):
    print "Processing " + file
    ds = np.genfromtxt(file, skip_header=1, delimiter='\t', converters={0: convert})
    ds[:,0] -= ds[0,0]                              # time reference set to 0
      
    if (ds[:,2].max() > 10000) == (ds[:,2].max() < 13000):
        ds = pd.DataFrame(ds,columns=colnames)
        slope = np.array([])
        
        for i in ds.index:
            index = np.arange(i - window_size_half + 1, i + window_size_half + 1)
            index = index[(index >= 0) & (index < len(ds))]
            dataset_part = ds[['distance','elevation']].iloc[index].dropna()

            regr = lm.LinearRegression()
            regr.fit(dataset_part.distance[:,np.newaxis], np.array(dataset_part.elevation))
            
            slope = np.append(slope,100*regr.coef_)

        dataset_new = ds.drop(['time','elevation','HR'], axis=1)
        dataset_new['slope'] = slope
     
        Datasets_all.append(dataset_new)
        
datasets = pd.concat(Datasets_all,ignore_index=True)
datasets.sort(columns='distance', inplace=True)
datasets.index = np.arange(len(datasets))
print "Velocidad mediana: {:.2f} m/s".format(datasets.speed.median())
print "Velocidad media: {:.2f} m/s".format(datasets.speed.mean())
print "Meadian pace (not corrected): %d:%d min/km"%speed_to_pace(datasets.speed.median())
print "Mean pace (not corrected): %d:%d min/km"%speed_to_pace(datasets.speed.mean())

pace = []
for i in datasets.index:
    m,s = speed_to_pace(datasets.speed[i])[0], speed_to_pace(datasets.speed[i])[1]
    if datasets.slope[i] > 0:
        m += (s/60)
        p = m / (1 + 0.033 * datasets.slope[i])
        pace.append(p)
    elif datasets.slope[i] < 0:
        m += (s/60)
        p = m / (1 + 0.018 * datasets.slope[i])
        pace.append(p)
        
print "Median pace (corrected): {0:.0f}:{1:.0f} min/km".format(divmod(np.nanmedian(pace),1)[0],
                    divmod(np.nanmedian(pace),1)[1]*60)
print "Mean pace (corrected): {0:.0f}:{1:.0f} min/km".format(divmod(np.nanmean(pace),1)[0],
                  divmod(np.nanmean(pace),1)[1]*60)
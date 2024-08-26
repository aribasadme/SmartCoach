# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 15:00:58 2015

@author: aRa
"""

import glob
import os
import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import dateutil.parser
from datetime import datetime

path = "/Users/aRa/OneDrive/Documentos/ERASMUS/PFC/SmartCoach/Running_w_HR/APE_runs_oct26-mar22"

def convert(date):
    dt = dateutil.parser.parse(date).replace(tzinfo=None)
    epoch = datetime.utcfromtimestamp(0)
    delta = dt - epoch
    return delta.total_seconds()

colnames = np.array(['time', 'elevation', 'distance', 'speed', 'HR'])
Datasets_all = []

os.chdir(path)
for file in glob.glob("*.tab"):
    print "Processing " + file
    dataset = np.genfromtxt(file, skip_header=1,delimiter='\t', converters={0: convert})
    dataset[:,0] -= dataset[0,0]
    dataset = pd.DataFrame(dataset,columns=colnames)
    slope = np.array([])
    window_size_half = 8
    for j in dataset.index:
        index = np.arange(j-window_size_half+1, j+window_size_half+1)
        index = index[(index >= 0) & (index < len(dataset))]
        dataset_part = dataset.iloc[index].dropna()
        regr = lm.LinearRegression()
        regr.fit(dataset_part.distance[:,np.newaxis], np.array(dataset_part.elevation))
        slope = np.append(slope,regr.coef_)
    dataset['slope'] = slope
    if (len(dataset) > 300) == (len(dataset) < 900):
        Datasets_all.append(dataset)


def group_HR_by_slopes(slopes, HR, nbins):
    bins = np.linspace(slopes.min(), slopes.max(), num=nbins + 1)
#    bins = np.arange(slopes.min(),slopes.max()+stepsize, stepsize)
    bin_index = np.digitize(slopes, bins) - 1
    HR_per_bin = []
    for bi in range(len(bins)-1):
        HR_per_bin.append(HR[bin_index == bi].values)
    return HR_per_bin, bins

def group_speed_by_slopes(slopes, speed, nbins):
    bins = np.linspace(slopes.min(), slopes.max(), num=nbins + 1)
#    bins = np.arange(slopes.min(),slopes.max()+stepsize, stepsize)
    bin_index = np.digitize(slopes, bins) - 1
    speed_per_bin = []
    for bi in range(len(bins)-1):
        speed_per_bin.append(speed[bin_index == bi].values)
    return speed_per_bin, bins
       
"""
Intento 2 (OK) - plot HR(distancia) y boxplot de HR(slope) para cada carrera
#%pylab qt
for dataset_chunk in (Datasets_all[:len(Datasets_all)/2], Datasets_all[len(Datasets_all)/2:]):
    plt.figure(figsize=(10, 80))
    ndatasets = len(dataset_chunk)
    for i, ds in enumerate(dataset_chunk):
        ax = plt.subplot(ndatasets, 2, 2*i + 1)
      
#        Plot de HR en función de la distancia en % para cada carrera
#
        ax.plot(ds.distance / ds.distance.values[-1], ds.HR)
        plt.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75')
        plt.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75')        
        window = np.array([0.2, 0.5, 0.8])
        ax.axvline(x=window[0],color='orange')
        ax.axvline(x=window[2],color='orange')
        if i < len(dataset_chunk) - 1:
            plt.setp( ax.get_xticklabels(), visible=False)
        else:
            ticks = np.linspace(0, 1, num=11)
            plt.xticks(ticks, ['%.2f' % v for v in ticks], rotation=90)

        plt.hold(True)
        #ds_mid = ds.query('(distance >= 3000) and (distance <= 8000)')
        tot_dist = ds.distance.values[-1]
        ds_mid = ds.query('(distance >= %f) and (distance <= %f)' % (window[0] * tot_dist, window[2] * tot_dist))
        HR_per_bin, bins = group_HR_by_slopes(ds_mid.slope, ds_mid.HR, 10)
        bin_mid = (bins[1:] + bins[:-1])/2
        ds2 = pd.DataFrame(HR_per_bin,index=bin_mid)
        ax = plt.subplot(ndatasets, 2, 2*i + 2)
        ds2.T.boxplot(ax=ax, rot=90)
        if i < len(dataset_chunk) - 1:
            plt.setp( ax.get_xticklabels(), visible=False)
        else:
            plt.xticks(range(1, len(bin_mid) + 1), ['%.2f' % v for v in bin_mid])


"""

#%pylab qt
nbins=10
window = np.array([0.2, 0.5, 0.8])

plt.figure()
datasets = pd.concat(Datasets_all,ignore_index=True)
datasets.sort(columns='distance', inplace=True)
tot_dist = datasets.distance.values[-1]
ds_mid1 = datasets.query('(distance >= %f) and (distance <= %f)' % (window[0] * tot_dist, window[1] * tot_dist))
ds_mid2 = datasets.query('(distance >= %f) and (distance <= %f)' % (window[1] * tot_dist, window[2] * tot_dist))

HR_per_bin1, bins = group_HR_by_slopes(ds_mid1.slope, ds_mid1.HR, nbins)
bin_mid = (bins[1:] + bins[:-1])/2
ds2 = pd.DataFrame(HR_per_bin1,index=bin_mid)
ds2[0].fillna(0,inplace=True)
ax = plt.subplot(1, 2, 1)
ds2.T.boxplot(ax=ax, rot=90)
plt.title('First window (20% - 50%)', fontweight='bold')
plt.xlabel('Slope')
plt.ylabel('Heart Rate [bpm]')
plt.ylim([120,200])
plt.xticks(range(1, len(bin_mid) + 1), ['%.2f' % v for v in bin_mid])

HR_per_bin2, bins = group_HR_by_slopes(ds_mid2.slope, ds_mid2.HR, nbins)
bin_mid = (bins[1:] + bins[:-1])/2
ds3 = pd.DataFrame(HR_per_bin2,index=bin_mid)
ds3[0].fillna(0,inplace=True)
ax = plt.subplot(1, 2, 2)
ds3.T.boxplot(ax=ax, rot=90)
plt.title('Second window (50% - 80%)', fontweight='bold')
plt.xlabel('Slope')
plt.ylabel('Heart Rate [bpm]')
plt.ylim([120,200])
plt.xticks(range(1, len(bin_mid) + 1), ['%.2f' % v for v in bin_mid])
"""
"""
#%pylab qt
nbins=10
window = np.array([0.2, 0.5, 0.8])

datasets = pd.concat(Datasets_all,ignore_index=True)
datasets.sort(columns='distance', inplace=True)
tot_dist = datasets.distance.values[-1]
ds_mid = datasets.query('(distance >= %f) and (distance <= %f)' % (window[0] * tot_dist, window[2] * tot_dist))
ds_mid1 = datasets.query('(distance >= %f) and (distance <= %f)' % (window[0] * tot_dist, window[1] * tot_dist))
ds_mid2 = datasets.query('(distance >= %f) and (distance <= %f)' % (window[1] * tot_dist, window[2] * tot_dist))

#HR_per_bin, bins = group_HR_by_slopes(ds_mid.slope, ds_mid.HR, nbins=10, stepsize=0.04)
color = dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='Gray')

HR_per_bin1, _ = group_HR_by_slopes(ds_mid1.slope, ds_mid1.HR, 14)
speed_per_bin1,_ = group_HR_by_slopes(ds_mid1.slope, ds_mid1.HR, 14)
bin_mid = (bins[1:] + bins[:-1])/2
ds2_HR = pd.DataFrame(HR_per_bin1,index=bin_mid)
ds2_speed = pd.DataFrame(speed_per_bin1,index=bin_mid)
ds2_HR[0].fillna(0,inplace=True)
ds2_speed[0].fillna(0,inplace=True)

plt.figure()

ax1 = plt.subplot(1, 2, 1)
plt.title('First window (20% - 50%)', fontsize=16, y=1.02)
bp1 = ds2_HR.T.boxplot(ax=ax1, rot=90)
plt.setp(bp1['boxes'], color='DarkGreen')
plt.setp(bp1['whiskers'], color='DarkGreen')
plt.setp(bp1['fliers'], color='red', marker='+')
plt.setp(bp1['medians'], color='DarkOrange')
plt.setp(bp1['caps'],color='DarkGreen')
ax1.set_ylabel('Heart Rate ['r'$bpm$]', color='DarkGreen')
for tl in ax1.get_yticklabels():
    tl.set_color('DarkGreen')
ax1.set_ylim([120,200])

ax2 = ax1.twinx()
bp2 = ds2_speed.T.boxplot(ax=ax2, rot=90)
plt.setp(bp2['boxes'],color='DarkBlue')
plt.setp(bp2['whiskers'],color='DarkBlue')
plt.setp(bp2['caps'],color='DarkBlue')
ax2.set_ylabel('Speed ['r'$m/s$]',color='DarkBlue')
plt.xlabel('Slope')
for tl in ax2.get_yticklabels():
    tl.set_color('DarkBlue')
plt.xticks(range(1, len(bin_mid) + 1), ['%.2f' % v for v in bin_mid])

HR_per_bin2,_ = group_HR_by_slopes(ds_mid2.slope, ds_mid2.HR, 14)
speed_per_bin2,_ = group_HR_by_slopes(ds_mid2.slope, ds_mid2.HR, 14) 
bin_mid = (bins[1:] + bins[:-1])/2
ds3_HR = pd.DataFrame(HR_per_bin2,index=bin_mid)
ds3_speed = pd.DataFrame(speed_per_bin2,index=bin_mid)
ds3_HR[0].fillna(0,inplace=True)
ds3_speed[0].fillna(0,inplace=True)

ax1 = plt.subplot(1, 2, 2)
plt.title('Second window (50% - 80%)', fontsize=16, y=1.02)
bp1 = ds3_HR.T.boxplot(ax=ax1, rot=90)
plt.setp(bp1['boxes'], color='DarkGreen')
plt.setp(bp1['whiskers'], color='DarkGreen')
plt.setp(bp1['fliers'], color='red', marker='+')
plt.setp(bp1['medians'], color='DarkOrange')
plt.setp(bp1['caps'],color='DarkGreen')
ax1.set_ylabel('Heart Rate ['r'$bpm$]', color='DarkGreen')
for tl in ax1.get_yticklabels():
    tl.set_color('DarkGreen')
ax1.set_ylim([120,200])


ax2 = ax1.twinx()
bp2 = ds3_speed.T.boxplot(ax=ax2, rot=90)
plt.setp(bp2['boxes'],color='DarkBlue')
plt.setp(bp2['whiskers'],color='DarkBlue')
plt.setp(bp2['caps'],color='DarkBlue')
ax2.set_ylabel('Speed ['r'$m/s$]',color='DarkBlue')
plt.xlabel('Slope')
for tl in ax2.get_yticklabels():
    tl.set_color('DarkBlue')
plt.xticks(range(1, len(bin_mid) + 1), ['%.2f' % v for v in bin_mid])
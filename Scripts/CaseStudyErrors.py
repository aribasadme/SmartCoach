# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 17:41:12 2015

@author: aRa
"""

import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import datetime as dt
import dateutil.parser

def convert(date):
    dt = dateutil.parser.parse(date).replace(tzinfo=None)
    epoch = dt.utcfromtimestamp(0)
    delta = dt - epoch
    return delta.total_seconds()

def speed_to_pace(speed):
    miuntes, seconds = divmod(100 / (6 * speed), 1)
    seconds *= 60
    return (miuntes, seconds)

def convert_pace(time):
    pace = dt.datetime.strptime(time,'%M:%S')
    pace = pace.minute + pace.second / 60.
    return pace

def profile2D(data_in, res=25, pr=[25, 50, 75]):
    vx_counts, vx_edges = np.histogram(data_in[[0]],res)
    vx_factor = pd.cut(data_in.icol(0),res) 
    lower_intervals = np.array([])
    upper_intervals = np.array([])
    for i in range(len(vx_counts)):
        lower_intervals = np.append(lower_intervals, vx_edges[i])
        upper_intervals = np.append(upper_intervals, vx_edges[i+1])
    vx_intervals = np.array([lower_intervals, upper_intervals])
    vx_mids = vx_intervals.mean(0)
    data_split = data_in.icol(1).groupby(vx_factor).groups
    q = []
    for k,v in data_split.iteritems():
        q.append(np.percentile(data_in[[1]].loc[v],pr))
    quantiles1 = pd.DataFrame(q,index=[data_split.keys()],columns=[str(p)+'%' for p in pr])
    quantiles1.sort_index(inplace=True,ascending=False)
    quantiles2 = quantiles1.sort_index(ascending=True)
    quantiles = pd.concat([quantiles1[len(quantiles1)/2:], quantiles2[len(quantiles1)/2+1:]])
    return [quantiles, vx_mids, vx_intervals, vx_counts]
    

path = "/Users/aRa/OneDrive/Documentos/ERASMUS/PFC/SmartCoach/Running_w_HR/APE_runs_oct26-mar22"
window_size_half = 3
window_distances = np.array([0., 0.25, 0.5, 0.75, 1.])
colnames = np.array(['time', 'elevation', 'distance', 'speed', 'HR'])

Datasets_all = []
os.chdir(path)
for file in glob.glob("*.tab"):
    print "Processing " + file
    ds = np.genfromtxt(file, skip_header=1, delimiter='\t', converters={0: convert})
    ds[:,0] -= ds[0,0]                              # time reference set to 0
      
    if (ds[:,2].max() > 10000) == (ds[:,2].max() < 13000):
        ds = pd.DataFrame(ds,columns=colnames)
        slope = np.array([])
        pace = []
        
        for i in ds.index:
            index = np.arange(i - window_size_half + 1, i + window_size_half + 1)
            index = index[(index >= 0) & (index < len(ds))]
            dataset_part = ds[['distance','elevation']].iloc[index].dropna()

            regr = lm.LinearRegression()
            regr.fit(dataset_part.distance[:,np.newaxis], np.array(dataset_part.elevation))
            
            slope = np.append(slope,regr.coef_)

        dataset_new = ds.drop(['time','elevation','HR'], axis=1)
        dataset_new['slope'] = slope

        Datasets_all.append(dataset_new)

slope_min = -0.2                              #slope
slope_max = 0.2                               #slope
speed_min = 0                                #speed
speed_max = 5                                #speed
slope_resolution = 25
distance_step = 100

training_data = pd.concat(Datasets_all,ignore_index=True)
subset_data = training_data.loc[:,['slope','speed']]
subset_data_filtered = subset_data[(subset_data.slope < slope_max)
                                   &(subset_data.slope > slope_min)
                                   &(subset_data.speed > speed_min)
                                   &(subset_data.speed < speed_max)]

current_profile = profile2D(subset_data_filtered,pr=[50])

path = "/Users/aRa/OneDrive/Documentos/ERASMUS/PFC/SmartCoach/Running_w_HR/TEST/"
os.chdir(path)
file = 'Morat-Fribourg2014.tab'
validation_data = np.genfromtxt(file, skip_header=1, delimiter='\t', converters={0: convert})
validation_data[:,0] -= validation_data[0,0]
validation_data = pd.DataFrame(validation_data,columns=colnames)
slope = np.array([])
for i in validation_data.index:
    index = np.arange(i-window_size_half+1, i+window_size_half+1)
    index = index[(index >= 0) & (index < len(validation_data))]
    dataset_part = validation_data.iloc[index].dropna()
    regr = lm.LinearRegression()
    regr.fit(dataset_part.distance[:,np.newaxis], np.array(dataset_part.elevation))
    slope = np.append(slope,regr.coef_)
validation_data['slope'] = slope

efforts_index = np.array([])
for i in validation_data.index:
    sl = validation_data.slope[i]
    sp = validation_data.speed[i]
    speed_temp = current_profile[0].loc[(current_profile[2][0] < sl) == (sl <= current_profile[2][1])]
    if len(speed_temp) == 0:
        if sl <= current_profile[2][0][0]:
            speed_temp = current_profile[0].iloc[0]
        if sl > current_profile[2][1][len(current_profile[0])-1]:
            speed_temp = current_profile[0].iloc[len(current_profile[0])-1]
    if len(np.flatnonzero(np.array(speed_temp) >= sp)) == 0:
        efforts_index = np.append(efforts_index,len(speed_temp)) 
    else:
        efforts_index = np.append(efforts_index, np.flatnonzero(np.array(speed_temp) >= sp).min())

prediction = np.array([])
for i in np.arange(0,len(current_profile[0].columns)):
    q = current_profile[0].T.iloc[i]    
    for sl in validation_data.slope:
        temp = q.loc[(current_profile[2][0] < sl) == (sl <= current_profile[2][1])]
        if len(temp) == 0:
            if sl <= current_profile[2][0][0]:
                a = q[0]
                prediction = np.append(prediction, a)
            elif sl > current_profile[2][1][len(current_profile[0])-1]:
                a = q[len(current_profile[0])-1]
                prediction = np.append(prediction,a)
        prediction = np.append(prediction,temp)
prediction = np.reshape(prediction,(-1, len(current_profile[0].columns)),order='F')
speed_prediction_table = pd.DataFrame(prediction, columns=current_profile[0].columns)

distances = np.array(validation_data.distance)
distance_diffs = np.concatenate([[distances[0]], distances[1:] - distances[:len(distances)-1]])

matrix_distance_diffs = np.reshape(distance_diffs.repeat(len(current_profile[0].columns)),(-1,len(current_profile[0].columns)),order='C')
matrix_distance_diffs = pd.DataFrame(matrix_distance_diffs, columns=speed_prediction_table.columns)

times_prediction_table = matrix_distance_diffs / prediction

time_runner = np.array(validation_data.time)
times_diff = np.concatenate([[time_runner[0]], time_runner[1:] - time_runner[:len(time_runner)-1]])
#times_prediction = distance_diffs / speed_corrected
#times_prediction[times_prediction == np.inf] = 0
#times = np.arange(len(time_runner))
#t1 = np.vectorize(lambda x: times_prediction[:x+1].sum(axis=0))
#times_prediction_total = t1(times) 

cs4_error = np.ravel(times_prediction_table.values) - times_diff


# BOXPLOT HRF PREDICTION ERRORS

data = [cs1_error, cs2_error, cs3_error, cs4_error]
Methods = ['Case 1', 'Case 2', 'Case 3', 'Case 4']

fig, ax1 = plt.subplots()
bp = plt.boxplot(data)
ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
ax1.set_axisbelow(True)
ax1.set_ylabel(r'Error [$s$]')

numBoxes = len(data)
medians = [np.nanmedian(i) for i in data]
pos = np.arange(numBoxes)+1
upperLabels = [str(np.round(s, 2)) for s in medians]

# Set the axes ranges and axes labels
ax1.set_xlim(0.5, numBoxes+0.5)
top = ax1.get_ylim()[1]*0.93
bottom = -5
xtickNames = plt.setp(ax1, xticklabels=Methods)
plt.setp(xtickNames, rotation=45, fontsize=8)

ax1.set_title('Comparison of prediction time errors {}'.format(file[:-4]))

for tick,label in zip(range(numBoxes),ax1.get_xticklabels()):
    k = tick % 2
    ax1.text(pos[tick], top, upperLabels[tick],
        horizontalalignment='center', size='x-small',
           color='r')
           


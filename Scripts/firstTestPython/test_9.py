# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 16:49:58 2015

@author: aRa
"""
import glob
import os
import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import dateutil.parser
import datetime as dt

path = "/Users/aRa/OneDrive/Documentos/ERASMUS/PFC/SmartCoach/firstTestsR/data_corrected"

def convert(date):
    dt = dateutil.parser.parse(date).replace(tzinfo=None)
    epoch = dt.utcfromtimestamp(0)
    delta = dt - epoch
    return delta.total_seconds()
    
colnames = np.array(['time', 'elevation', 'distance', 'speed'])
Datasets_all = []

os.chdir(path)
for file in glob.glob("*.tab"):
    print "Processing " + file
    dataset = np.genfromtxt(file, skip_header=1,delimiter='\t', converters={0: convert})
    dataset[:,0] -= dataset[0,0]
    dataset = pd.DataFrame(dataset,columns=colnames)
    
    # Calculo del pendiente utilizando ventana deslizante de longitud 6
    slope = np.array([])
    window_size_half = 3
    for j in dataset.index:
        index = np.arange(j-window_size_half+1, j+window_size_half+1)
        index = index[(index >= 0) & (index < len(dataset))]
        dataset_part = dataset.iloc[index].dropna()
        regr = lm.LinearRegression()
        regr.fit(dataset_part.distance[:,np.newaxis], np.array(dataset_part.elevation))
        slope = np.append(slope,regr.coef_)
    dataset_new = pd.concat([dataset,pd.DataFrame(slope,columns=['slope'])], axis=1).dropna()
   
    Datasets_all.append(dataset_new)
   
#--------------------------------------------------------------------------------------------------

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
    
#--------------------------------------------------------------------------------------------------

slope_min = -0.2                             #slope
slope_max = 0.2                              #slope
speed_min = 0                                #speed
speed_max = 5                                #speed
slope_resolution = 25
distance_step = 100
n_files = len(Datasets_all)
probabilities = np.arange(10, 91, 1)

training_data = pd.concat(Datasets_all[0:24],ignore_index=True)
subset_data = training_data.loc[:,['slope','speed']]
subset_data_filtered = subset_data[(subset_data.slope < slope_max)&
                                    (subset_data.slope > slope_min)&
                                    (subset_data.speed > speed_min)&
                                    (subset_data.speed < speed_max)]
current_profile = profile2D(subset_data_filtered,pr=probabilities)

run_out = 26
validation_data = Datasets_all[run_out]

slopes = validation_data.slope
distances = np.array(validation_data.distance)
distance_diffs = np.concatenate([[distances[0]], 
                                 distances[1:] - distances[:len(distances)-1]])

# Getting efforts index
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

# ----------------------

prediction = np.array([])
for i in np.arange(0,len(current_profile[0].columns)):
    q = current_profile[0].T.iloc[i]    
    for sl in slopes:
        temp = q.loc[(current_profile[2][0] < sl) == (sl <= current_profile[2][1])]
        if len(temp) == 0:
            if sl <= current_profile[2][0][0]:
             prediction = np.append(prediction, q[0])
            elif sl > current_profile[2][1][len(current_profile[0])-1]:
                prediction = np.append(prediction, q[len(current_profile[0])-1])
        prediction = np.append(prediction,temp)

prediction = np.reshape(prediction,(-1, len(current_profile[0].columns)),order='F')
speed_prediction_table = pd.DataFrame(prediction, columns=current_profile[0].columns)
matrix_distance_diffs = np.reshape(distance_diffs.repeat(len(current_profile[0].columns)),
                                   (-1,len(current_profile[0].columns)),order='C')
times_prediction_table = matrix_distance_diffs / prediction
time_runner = np.array(validation_data.time)

times = np.arange(len(efforts_index))
t1 = np.vectorize(lambda x: time_runner[x] + times_prediction_table[:len(times_prediction_table)-1-x,
                                                                    int(efforts_index[x])].sum(axis=0))
t2 = np.vectorize(lambda x: times_prediction_table[:len(times_prediction_table)-1-x,
                                                   int(efforts_index[x])].sum(axis=0))
times_prediction_total = t1(times) / 60
times_prediction_to_goal = t2(times) / 60

times_prediction_total_averaged=[]
for i in np.arange(len(times_prediction_total)):
        idx = np.arange(i - 5 + 1, i)
        idx = idx[(idx >= 0) & (idx < len(times_prediction_total))] 
        tpt_tmp = np.average(times_prediction_total[idx])
        times_prediction_total_averaged.append(tpt_tmp)
        
laps = np.arange(0,10000,2000)
l1 = np.vectorize(lambda x: times_prediction_total_averaged[np.flatnonzero(validation_data.distance > x).min()])
l2 = np.vectorize(lambda x: times_prediction_to_goal[np.flatnonzero(validation_data.distance > x).min()])
laps_times_total = l1(laps)
laps_times_to_goal = l2(laps)
time_min = times_prediction_total.min()
time_max = times_prediction_total.max()

        
fig = plt.figure()
ax1 = fig.add_subplot(211)
im=ax1.plot(distances,validation_data.elevation,c='black',zorder=1)
ax1.set_title('Elevation',fontsize=16,y=1.02)
ax1.set_ylabel('Elevation ['r'$m$]')
ax1.grid(axis='y', alpha=0.5)
ax1.set_yticks(ax1.get_yticks()[::2])

ax2 = fig.add_subplot(212)
ax2.plot(distances, times_prediction_total_averaged,c='black',zorder=2)
ax2.scatter(laps,laps_times_total-0.1,c='white',zorder=1,linewidth=0)
ax2.axhline(y=time_runner[-1]/60, color='magenta')
ax2.text(7000,55,'Real time {}'.format(dt.timedelta(seconds=time_runner[-1])), 
         color='magenta', fontsize=10)
ax2.set_title('Estimated time',fontsize=16,y=1.02)
ax2.set_xlabel('Distance ['r'$m$]')
ax2.set_ylabel('Time ['r'$min$]')
ax2.grid(alpha=0.5)
ax2.set_ylim([time_min - (time_max-time_min)/2., time_max + (time_max-time_min)/2])
ax2.set_yticks(ax2.get_yticks()[::2])

ax1.set_xlim(ax2.get_xlim())
#for l in laps:
#    ax1.axvline(x=l,color='orange',zorder=2)
#    ax2.axvline(x=l,color='orange',zorder=2)

plt.tight_layout()

pp = PdfPages('../estimatedTimeRun{}_adaptive_copia.pdf'.format(run_out))
pp.savefig()
pp.close()



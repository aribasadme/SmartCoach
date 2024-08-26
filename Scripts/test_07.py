# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 17:08:23 2015

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

path = "/Users/aRa/OneDrive/Documentos/ERASMUS/PFC/SmartCoach/Running_w_HR/APE_runs_oct26-mar22"
window_size_half = 3
colnames = np.array(['time', 'elevation', 'distance', 'speed', 'HR'])

def convert(date):
    dt = dateutil.parser.parse(date).replace(tzinfo=None)
    epoch = dt.utcfromtimestamp(0)
    delta = dt - epoch
    return delta.total_seconds()
    

def speed_to_pace(speed):
    miuntes, seconds = divmod(100 / (6 * speed), 1)
    seconds *= 60
    return (miuntes, seconds)

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

slope_min = -0.2                              #slope
slope_max = 0.2                               #slope
speed_min = 0                                #speed
speed_max = 5                                #speed
slope_resolution = 25
distance_step = 100
n_files = len(Datasets_all)
probabilities = np.arange(10, 91, 1)

training_data = pd.concat(Datasets_all,ignore_index=True)
subset_data = training_data.loc[:,['slope','speed']]
subset_data_filtered = subset_data[(subset_data.slope < slope_max)
                                   &(subset_data.slope > slope_min)
                                   &(subset_data.speed > speed_min)
                                   &(subset_data.speed < speed_max)]

current_profile = profile2D(subset_data_filtered)

path = "/Users/aRa/OneDrive/Documentos/ERASMUS/PFC/SmartCoach/Running_w_HR/TEST"
os.chdir(path)
file = "Morat-Fribourg2014.tab"

dataset = np.genfromtxt(file, skip_header=1,delimiter='\t', converters={0: convert})
dataset[:,0] -= dataset[0,0]
dataset = pd.DataFrame(dataset,columns=colnames)
slope = np.array([])
for i in dataset.index:
    index = np.arange(i-window_size_half+1, i+window_size_half+1)
    index = index[(index >= 0) & (index < len(dataset))]
    dataset_part = dataset.iloc[index].dropna()
    regr = lm.LinearRegression()
    regr.fit(dataset_part.distance[:,np.newaxis], np.array(dataset_part.elevation))
    slope = np.append(slope,regr.coef_)

validation_data = pd.concat([dataset.dropna(), pd.DataFrame(slope,columns=['slope']).dropna()], axis=1)
slopes = validation_data.slope
distances = np.array(validation_data.distance)
distance_diffs = np.concatenate([[distances[0]], distances[1:] - distances[:len(distances)-1]])

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

efforts_index[np.flatnonzero(efforts_index == 3.)] = 2.

# ----------------------

prediction = np.array([])
for i in np.arange(0,len(current_profile[0].columns)):
    q = current_profile[0].T.iloc[i]    
    for sl in slopes:
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

matrix_distance_diffs = np.reshape(distance_diffs.repeat(len(current_profile[0].columns)),
                                   (-1,len(current_profile[0].columns)),order='C')
times_prediction_table = matrix_distance_diffs / prediction
times_prediction_start = times_prediction_table.sum(axis=0)

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
l1 = np.vectorize(lambda x: 
                    times_prediction_total[np.flatnonzero(validation_data.distance > x).min()])
l2 = np.vectorize(lambda x: 
                    times_prediction_to_goal[np.flatnonzero(validation_data.distance > x).min()])
laps_times_total = l1(laps)
laps_times_to_goal = l2(laps)
time_min = times_prediction_total.min()
time_max = times_prediction_total.max()

#print "Distancia de la carrera {0} {1:.2f} km".format(file[:-4], validation_data.distance.iloc[-1]/1000)
#print "Tiempo real de carrera {}".format(dt.timedelta(seconds=validation_data.time.iloc[-1]))
#print "Tiempo estimado de carrera {} (mediana ritmo)".format(dt.timedelta(seconds=int(times_prediction_start[1].values)))
#print 'Error: {0} ({1:0.1f}%)'.format(dt.timedelta(seconds=abs(validation_data.time.iloc[-1] - round(int(times_prediction_start.values)))),
#                                           abs(validation_data.time.iloc[-1] - round(int(times_prediction_start.values)))/validation_data.time.iloc[-1] * 100)
fig = plt.figure()
ax1 = fig.add_subplot(211)
im=ax1.plot(distances,validation_data.elevation,'k',zorder=1)
ax1.set_title('Elevation',fontsize=16,y=1.02)
ax1.set_ylabel('Elevation ['r'$m$]')
ax1.grid(axis='y', alpha=0.5)
#ax1.set_yticks(ax1.get_yticks()[::2])
plt.setp(ax1.get_xticklabels(), visible=False)

ax2 = fig.add_subplot(212)
ax2.plot(distances, times_prediction_total_averaged,c='black',zorder=1)
ax2.scatter(laps,laps_times_total-0.1,c='white',zorder=1,linewidth=0)
ax2.axhline(y=time_runner[-1]/60, color='magenta')
ax2.text(0.95,0.95,'Real time {}'.format(dt.timedelta(seconds=time_runner[-1])), 
         verticalalignment='top', horizontalalignment='right', transform=ax2.transAxes,
         color='magenta', fontsize=14)
ax2.set_title('Estimated time',fontsize=16,y=1.02)
ax2.set_xlabel('Distance ['r'$m$]')
ax2.set_ylabel('Time ['r'$min$]')
ax2.grid(alpha=0.5)
#ax2.set_ylim([time_min - (time_max-time_min)/2., time_max + (time_max-time_min)/2])
ax2.set_yticks(ax2.get_yticks()[::2])

ax1.set_xlim(ax2.get_xlim())
plt.tight_layout()
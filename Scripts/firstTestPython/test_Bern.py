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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import dateutil.parser
from datetime import datetime

path = "/Users/aRa/OneDrive/Documentos/ERASMUS/PFC/SmartCoach/firstTestsR/data_corrected"

def convert(date):
    dt = dateutil.parser.parse(date).replace(tzinfo=None)
    epoch = datetime.utcfromtimestamp(0)
    delta = dt - epoch
    return delta.total_seconds()
    
colnames = np.array(['time', 'elevation', 'distance', 'speed'])
window_size_half = 3
Datasets_all = []

os.chdir(path)
for file in glob.glob("*.tab"):
    print "Processing " + file
    dataset = np.genfromtxt(file, skip_header=1,delimiter='\t', converters={0: convert})
    dataset[:,0] -= dataset[0,0]
    dataset = pd.DataFrame(dataset,columns=colnames)
    
    # Calculo del pendiente utilizando ventana deslizante de longitud 6
    slope = np.array([])
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
#probabilities = np.arange(10, 91, 1)
probabilities = [25, 50, 75]
training_data = pd.concat(Datasets_all,ignore_index=True)
subset_data = training_data.loc[:,['slope','speed']]
subset_data_filtered = subset_data[(subset_data.slope < slope_max)&(subset_data.slope > slope_min)&(subset_data.speed > speed_min)&(subset_data.speed < speed_max)]

current_profile = profile2D(subset_data_filtered,pr=probabilities)

path = "/Users/aRa/OneDrive/Documentos/ERASMUS/PFC/SmartCoach/firstTestsR/APE_recent_races"
os.chdir(path)
file = "activity_502644583.tab"

dataset = np.genfromtxt(file, skip_header=1,delimiter='\t', converters={0: convert})
dataset[:,0] -= dataset[0,0]
dataset = pd.DataFrame(dataset,columns=colnames)
slope = np.array([])
y_lr = []
x_trs = []
ds_parts_dist = []
ds_parts_elev = []
for i in dataset.index:
    index = np.arange(i-window_size_half+1, i+window_size_half+1)
    index = index[(index >= 0) & (index < len(dataset))]
    dataset_part = dataset.iloc[index].dropna()
    ds_parts_dist.append(np.array(dataset_part.distance))
    ds_parts_elev.append(np.array(dataset_part.elevation))
    x_tr = np.linspace(dataset_part.distance.min(),
                       dataset_part.distance.max(),len(dataset_part))
    x_trs.append(x_tr)        
    regr = lm.LinearRegression()
    regr.fit(dataset_part.distance[:,np.newaxis], np.array(dataset_part.elevation))
    y_lr.append(regr.predict(x_tr[:,np.newaxis]))
    slope = np.append(slope,regr.coef_)

plt.plot(ds_parts_dist[2], ds_parts_elev[2], '--k')
plt.plot(x_trs[2], y_lr[2], 'g')

plt.plot(np.concatenate(ds_parts_dist), np.concatenate(ds_parts_elev), '--k')
plt.plot(np.concatenate(x_trs), np.concatenate(y_lr), 'g')

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

matrix_distance_diffs = np.reshape(distance_diffs.repeat(len(current_profile[0].columns)),(-1,len(current_profile[0].columns)),order='C')
matrix_distance_diffs = pd.DataFrame(matrix_distance_diffs, columns=speed_prediction_table.columns)

times_prediction_table = matrix_distance_diffs / prediction
times_prediction_start = times_prediction_table.sum(axis=0) / 60

fig = plt.figure(figsize=(5,3), dpi=72)
ax = fig.add_subplot(111)
ax.plot(probabilities,times_prediction_start,'k')
ax.set_title(r'Estimated time', y=1.02)
ax.set_xlabel('Effort (%)')
ax.set_ylabel('Time (min)')
ax.set_xticks((20,40,60,80))
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
#fig.savefig('estimatedTime_vs_effort_Bern.pdf', dpi=fig.dpi, bbox_inches='tight')


effort_index = 60
print "You want to finish in {:.1f} minutes".format(times_prediction_start.iloc[effort_index-10])
print "The application will set the effort to {} %".format(probabilities[effort_index-10])

speed_prediction = speed_prediction_table.icol(effort_index)

times_prediction = distance_diffs / speed_prediction # Error positions 88 & 89
time_runner = validation_data.time

times = np.arange(len(times_prediction))
t1 = np.vectorize(lambda x: time_runner[x] + times_prediction[:len(times_prediction_table)-1-x].sum(axis=0))
t2 = np.vectorize(lambda x: times_prediction[:len(times_prediction)-1-x].sum(axis=0))
times_prediction_total = t1(times) / 60 
times_prediction_to_goal = t2(times) / 60

laps = np.arange(0,15000,2000)
l1 = np.vectorize(lambda x: times_prediction_total[np.flatnonzero(validation_data.distance>x).min()])
l2 = np.vectorize(lambda x: times_prediction_to_goal[np.flatnonzero(validation_data.distance>x).min()])
laps_times_total = l1(laps)
laps_times_to_goal = l2(laps)
time_min = times_prediction_total.min()
time_max = times_prediction_total.max()
text_cord = np.linspace(time_min, time_max, 10)
efforts_probabilities = efforts_index + probabilities.min() - 1


#--------------------
fig = plt.figure(figsize=(14,12), dpi=72)
ax1 = fig.add_subplot(311)
im=ax1.scatter(distances,validation_data.elevation,c=efforts_probabilities,cmap=plt.cm.jet,linewidth=0)
divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right", size="1%", pad=0.05)
cbar1 = plt.colorbar(im, cax=cax1, drawedges=False, ticks=[20,40,60,80])
ax1.set_title('Personalized Effort',y=1.02)
ax1.set_ylabel('Elevation (m)')
ax1.set_yticks(ax1.get_yticks()[::2])

ax2 = fig.add_subplot(312)
ax2.plot(distances, times_prediction_total,c='black',zorder=1)
ax2.set_title('Estimated time (Effort={:.1f} %)'.format(probabilities[effort_index-10]),y=1.02)
#ax2.scatter(laps,laps_times_total-0.1,c='orange',zorder=3,linewidth=0)
ax2.set_ylim([82, 89])
ax2.set_ylabel('Time (min)')

ax3 = fig.add_subplot(313)
real = plt.plot(distances,validation_data.speed,c='black',zorder=2)
estimated = plt.plot(validation_data.distance, speed_prediction, c='magenta',zorder=2)
ax3.scatter(validation_data.distance, speed_prediction, c='white',linewidth=0, zorder=1)
plt.legend(['Real', 'Estimated'], loc=4, fontsize='small')
ax3.set_title('Speed',y=1.02)
ax3.set_xlabel('Distance (m)')
ax3.set_ylabel('Speed (m/s)')

for ax in [ax1, ax2, ax3]:
    ax.set_xlim([-1000, 17000])
    ax.grid(alpha=0.5)
    ax.set_xticks(np.arange(0,17000, 1000)[::2])
#for l in laps[::2]:
#    ax1.axvline(x=l,color='orange')
#    ax1.set_xticks((0,5000,10000,15000))
#    ax2.axvline(x=l,color='orange',zorder=2)
#    ax2.set_xticks((0,5000,10000,15000))
#    ax3.axvline(x=l,color='orange',zorder=3)
#    ax3.set_xticks((0,5000,10000,15000))
plt.tight_layout()
plt.draw()

#fig.savefig('../../estimatedTimeRunBern_user.pdf', bbox_inches='tight', dpi=fig.dpi)

# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 11:32:17 2015

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
window_distances = np.array([0., 0.25, 0.5, 0.75, 1.])
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
ds_mid1 = []
ds_mid2 = []
ds_mid3 = []
ds_mid4 = []

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
        

        ds_mid1.append(dataset_new.query('(distance >= %f) and (distance < %f)'
                    % (dataset_new.distance.iloc[-1]*window_distances[0], dataset_new.distance.iloc[-1]*window_distances[1])))
        ds_mid2.append(dataset_new.query('(distance >= %f) and (distance < %f)'
                    % (dataset_new.distance.iloc[-1]*window_distances[1], dataset_new.distance.iloc[-1]*window_distances[2])))
        ds_mid3.append(dataset_new.query('(distance >= %f) and (distance < %f)'
                    % (dataset_new.distance.iloc[-1]*window_distances[2], dataset_new.distance.iloc[-1]*window_distances[3])))
        ds_mid4.append(dataset_new.query('(distance >= %f) and (distance <= %f)' 
                    % (dataset_new.distance.iloc[-1]*window_distances[3], dataset_new.distance.iloc[-1]*window_distances[4])))

        Datasets_all.append(dataset_new)
   
#--------------------------------------------------------------------------------------------------

def profile2D(data_in, slopes=[], res=25, pr=[25, 50, 75]):
    vx_counts, vx_edges = np.histogram(data_in[[0]],res)
    if len(slopes) == 0:
        vx_factor = pd.cut(data_in.icol(0),res)
    else:
        vx_factor = pd.cut(slopes,res)
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
training_data.sort(columns='distance', inplace=True)

tr_data1 = pd.concat(ds_mid1, ignore_index=True)
tr_data2 = pd.concat(ds_mid2, ignore_index=True)
tr_data3 = pd.concat(ds_mid3, ignore_index=True)
tr_data4 = pd.concat(ds_mid4, ignore_index=True)

subset_data = training_data.loc[:,['slope','speed']]
subset_data_filtered = subset_data[(subset_data.slope < slope_max)&(subset_data.slope > slope_min)&(subset_data.speed > speed_min)&(subset_data.speed < speed_max)]

ss_data1 = tr_data1.loc[:,['slope','speed']]
ss_data1_filtered = ss_data1[(ss_data1.slope < slope_max)&(ss_data1.slope > slope_min)&(ss_data1.speed > speed_min)&(ss_data1.speed < speed_max)]

ss_data2 = tr_data2.loc[:,['slope','speed']]
ss_data2_filtered = ss_data2[(ss_data2.slope < slope_max)&(ss_data2.slope > slope_min)&(ss_data2.speed > speed_min)&(ss_data2.speed < speed_max)]

ss_data3 = tr_data3.loc[:,['slope','speed']]
ss_data3_filtered = ss_data3[(ss_data3.slope < slope_max)&(ss_data3.slope > slope_min)&(ss_data3.speed > speed_min)&(ss_data3.speed < speed_max)]

ss_data4 = tr_data4.loc[:,['slope','speed']]
ss_data4_filtered = ss_data4[(ss_data4.slope < slope_max)&(ss_data4.slope > slope_min)&(ss_data4.speed > speed_min)&(ss_data4.speed < speed_max)]

current_profile = profile2D(subset_data_filtered)
cp1 = profile2D(ss_data1_filtered,slopes=subset_data_filtered.slope)
cp2 = profile2D(ss_data2_filtered,slopes=subset_data_filtered.slope)
cp3 = profile2D(ss_data3_filtered,slopes=subset_data_filtered.slope)
cp4 = profile2D(ss_data4_filtered,slopes=subset_data_filtered.slope)

path = "/Users/aRa/OneDrive/Documentos/ERASMUS/PFC/SmartCoach/Running_w_HR/TEST"
slopes = os.chdir(path)
file = "test_activity_676359032.tab"

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
vd1 = validation_data[:150]
vd2 = validation_data[151:285]
vd3 = validation_data[286:420]
vd4 = validation_data[421:]

slopes = validation_data.slope
sl1 = vd1.slope
sl2 = vd2.slope
sl3 = vd3.slope
sl4 = vd4.slope

distances = np.array(validation_data.distance)
d1 = np.array(vd1.distance)
d2 = np.array(vd2.distance)
d3 = np.array(vd3.distance)
d4 = np.array(vd4.distance)

distance_diffs = np.concatenate([[distances[0]], distances[1:] - distances[:len(distances)-1]])
dd1 = np.concatenate([[d1[0]], d1[1:] - d1[:len(d1)-1]])
dd2 = np.concatenate([[d2[0]], d2[1:] - d2[:len(d2)-1]])
dd3 = np.concatenate([[d3[0]], d3[1:] - d3[:len(d3)-1]])
dd4 = np.concatenate([[d4[0]], d4[1:] - d4[:len(d4)-1]])


# Getting efforts index
efforts_index = np.array([])
eff_idx1 = np.array([])
eff_idx2 = np.array([])
eff_idx3 = np.array([])
eff_idx4 = np.array([])

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

for i in vd1.index:
    sl = vd1.slope[i]
    sp = vd1.speed[i]
    speed_temp = cp1[0].loc[(cp1[2][0] < sl) == (sl <= cp1[2][1])]
    if len(speed_temp) == 0:
        if sl <= cp1[2][0][0]:
            speed_temp = cp1[0].iloc[0]
        if sl > cp1[2][1][len(cp1[0])-1]:
            speed_temp = cp1[0].iloc[len(cp1[0])-1]
    if len(np.flatnonzero(np.array(speed_temp) >= sp)) == 0:
        eff_idx1 = np.append(eff_idx1,len(speed_temp)) 
    else:
        eff_idx1 = np.append(eff_idx1, np.flatnonzero(np.array(speed_temp) >= sp).min())


for i in vd2.index:
    sl = vd2.slope[i]
    sp = vd2.speed[i]
    speed_temp = cp2[0].loc[(cp2[2][0] < sl) == (sl <= cp2[2][1])]
    if len(speed_temp) == 0:
        if sl <= cp2[2][0][0]:
            speed_temp = cp2[0].iloc[0]
        if sl > cp2[2][1][len(cp2[0])-1]:
            speed_temp = cp2[0].iloc[len(cp2[0])-1]
    if len(np.flatnonzero(np.array(speed_temp) >= sp)) == 0:
        eff_idx2 = np.append(eff_idx2,len(speed_temp)) 
    else:
        eff_idx2 = np.append(eff_idx2, np.flatnonzero(np.array(speed_temp) >= sp).min())
eff_idx2[np.flatnonzero(eff_idx2 == 3.)] = 2.

for i in vd3.index:
    sl = vd3.slope[i]
    sp = vd3.speed[i]
    speed_temp = cp3[0].loc[(cp3[2][0] < sl) == (sl <= cp3[2][1])]
    if len(speed_temp) == 0:
        if sl <= cp3[2][0][0]:
            speed_temp = cp3[0].iloc[0]
        if sl > cp3[2][1][len(cp3[0])-1]:
            speed_temp = cp3[0].iloc[len(cp3[0])-1]
    if len(np.flatnonzero(np.array(speed_temp) >= sp)) == 0:
        eff_idx3 = np.append(eff_idx3,len(speed_temp)) 
    else:
        eff_idx3 = np.append(eff_idx3, np.flatnonzero(np.array(speed_temp) >= sp).min())

for i in vd4.index:
    sl = vd4.slope[i]
    sp = vd4.speed[i]
    speed_temp = cp4[0].loc[(cp4[2][0] < sl) == (sl <= cp4[2][1])]
    if len(speed_temp) == 0:
        if sl <= cp4[2][0][0]:
            speed_temp = cp4[0].iloc[0]
        if sl > cp4[2][1][len(cp3[0])-1]:
            speed_temp = cp4[0].iloc[len(cp4[0])-1]
    if len(np.flatnonzero(np.array(speed_temp) >= sp)) == 0:
        eff_idx4 = np.append(eff_idx4,len(speed_temp)) 
    else:
        eff_idx4 = np.append(eff_idx4, np.flatnonzero(np.array(speed_temp) >= sp).min())
        
# ----------------------

prediction = np.array([])
pred1 = np.array([])
pred2 = np.array([])
pred3 = np.array([])
pred4 = np.array([])

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
        
for i in np.arange(0,len(cp1[0].columns)):
    q = cp1[0].T.iloc[i]    
    for sl in sl1:
        temp = q.loc[(cp1[2][0] < sl) == (sl <= cp1[2][1])]
        if len(temp) == 0:
            if sl <= cp1[2][0][0]:
                pred1 = np.append(pred1, q[0])
            elif sl > cp1[2][1][len(cp1[0])-1]:
                pred1 = np.append(pred1, q[len(cp1[0])-1])
        pred1 = np.append(pred1,temp)

for i in np.arange(0,len(cp2[0].columns)):
    q = cp2[0].T.iloc[i]    
    for sl in sl2:
        temp = q.loc[(cp2[2][0] < sl) == (sl <= cp2[2][1])]
        if len(temp) == 0:
            if sl <= cp2[2][0][0]:
                pred2 = np.append(pred2, q[0])
            elif sl > cp2[2][1][len(cp2[0])-1]:
                pred2 = np.append(pred2, q[len(cp2[0])-1])
        pred2 = np.append(pred2,temp)
        
for i in np.arange(0,len(cp3[0].columns)):
    q = cp3[0].T.iloc[i]    
    for sl in sl3:
        temp = q.loc[(cp3[2][0] < sl) == (sl <= cp3[2][1])]
        if len(temp) == 0:
            if sl <= cp3[2][0][0]:
                pred3 = np.append(pred3, q[0])
            elif sl > cp3[2][1][len(cp3[0])-1]:
                pred3 = np.append(pred3, q[len(cp3[0])-1])
        pred3 = np.append(pred3,temp)

for i in np.arange(0,len(cp4[0].columns)):
    q = cp4[0].T.iloc[i]    
    for sl in sl4:
        temp = q.loc[(cp4[2][0] < sl) == (sl <= cp4[2][1])]
        if len(temp) == 0:
            if sl <= cp4[2][0][0]:
                pred4 = np.append(pred4, q[0])
            elif sl > cp4[2][1][len(cp4[0])-1]:
                pred4 = np.append(pred4, q[len(cp4[0])-1])
        pred4 = np.append(pred4,temp)
        
prediction = np.reshape(prediction,(-1, len(current_profile[0].columns)),order='F')
pred1 = np.reshape(pred1,(-1, len(cp1[0].columns)),order='F')
pred2 = np.reshape(pred2,(-1, len(cp2[0].columns)),order='F')
pred3 = np.reshape(pred3,(-1, len(cp3[0].columns)),order='F')
pred4 = np.reshape(pred4,(-1, len(cp4[0].columns)),order='F')

speed_prediction_table = pd.DataFrame(prediction, columns=current_profile[0].columns)
spt1 = pd.DataFrame(pred1, columns=cp1[0].columns)
spt2 = pd.DataFrame(pred2, columns=cp2[0].columns)
spt3 = pd.DataFrame(pred3, columns=cp3[0].columns)
spt4 = pd.DataFrame(pred4, columns=cp4[0].columns)

matrix_distance_diffs = np.reshape(distance_diffs.repeat(len(current_profile[0].columns)),(-1,len(current_profile[0].columns)),order='C')
matrix_distance_diffs = pd.DataFrame(matrix_distance_diffs, columns=speed_prediction_table.columns)
mdd1 = np.reshape(dd1.repeat(len(cp1[0].columns)),(-1,len(cp1[0].columns)),order='C')
mdd1 = pd.DataFrame(mdd1, columns=spt1.columns)
mdd2 = np.reshape(dd2.repeat(len(cp2[0].columns)),(-1,len(cp2[0].columns)),order='C')
mdd2 = pd.DataFrame(mdd2, columns=spt2.columns)
mdd3 = np.reshape(dd3.repeat(len(cp3[0].columns)),(-1,len(cp3[0].columns)),order='C')
mdd3 = pd.DataFrame(mdd3, columns=spt3.columns)
mdd4 = np.reshape(dd4.repeat(len(cp4[0].columns)),(-1,len(cp4[0].columns)),order='C')
mdd4 = pd.DataFrame(mdd4, columns=spt4.columns)

times_prediction_table = matrix_distance_diffs / prediction
tpt1 = mdd1 / pred1
tpt2 = mdd2 / pred2
tpt3 = mdd3 / pred3
tpt4 = mdd4 / pred4

times_prediction_start = times_prediction_table.sum(axis=0)
tps1 = tpt1.sum(axis=0)
tps2 = tpt2.sum(axis=0)
tps3 = tpt3.sum(axis=0)
tps4 = tpt4.sum(axis=0)

print 'Distancia de la carrera {0}: {1:.2f} km'.format(file[:-4], validation_data.distance.iloc[-1]/1000)
print 'Tiempo estimado de carrera {} (mediana ritmo)'.format(dt.timedelta(seconds=int(times_prediction_start[1])))
print 'Tiempo real de carrera {}'.format(dt.timedelta(seconds=validation_data.time.iloc[-1]))
print 'Error: {0} ({1:0.1f}%)\n'.format(dt.timedelta(seconds=abs(validation_data.time.iloc[-1] - round(int(times_prediction_start[1])))),
                                           abs(validation_data.time.iloc[-1] - round(int(times_prediction_start[1])))/validation_data.time.iloc[-1] * 100)

print 'Distancia del 0% al 25% de {0}: {1:.2f} km'.format(file[:-4],(vd1.distance.iloc[-1] - vd1.distance.iloc[0])/1000)
print 'Tiempo estimado de carrera {} (mediana ritmo)'.format(dt.timedelta(seconds=int(tps1[1])))
print 'Tiempo real de carrera {}'.format(dt.timedelta(seconds=(vd1.time.iloc[-1])))
print 'Error: {0} ({1:0.1f}%)\n'.format(dt.timedelta(seconds=abs(vd1.time.iloc[-1] - round(int(tps1[1])))),
                                           abs(vd1.time.iloc[-1] - round(int(tps1[1])))/vd1.time.iloc[-1] * 100)

print 'Distancia del 25% al 50% de {0} {1:.2f} km'.format(file[:-4],(vd2.distance.iloc[-1] - vd2.distance.iloc[0])/1000)
print 'Tiempo estimado de carrera {} (mediana ritmo)'.format(dt.timedelta(seconds=int(tps2[1])))
print 'Tiempo real de carrera {}'.format(dt.timedelta(seconds=(vd2.time.iloc[-1])))
print 'Error: {0} ({1:0.1f}%)\n'.format(dt.timedelta(seconds=abs(vd2.time.iloc[-1] - round(int(tps2[1])))),
                                           abs(vd2.time.iloc[-1] - round(int(tps2[1])))/vd2.time.iloc[-1] * 100)

print 'Distancia del 50% al 75% de {0} {1:.2f} km'.format(file[:-4],(vd3.distance.iloc[-1] - vd3.distance.iloc[0])/1000)
print 'Tiempo estimado de carrera {} (mediana ritmo)'.format(dt.timedelta(seconds=int(tps3[1])))
print 'Tiempo real de carrera {}'.format(dt.timedelta(seconds=(vd3.time.iloc[-1])))
print 'Error: {0} ({1:0.1f}%)\n'.format(dt.timedelta(seconds=abs(vd3.time.iloc[-1] - round(int(tps2[1])))),
                                           abs(vd3.time.iloc[-1] - round(int(tps3[1])))/vd3.time.iloc[-1] * 100)

print 'Distancia del 75% al 100% de {0} {1:.2f} km'.format(file[:-4],(vd4.distance.iloc[-1]- vd4.distance.iloc[0])/1000)
print 'Tiempo estimado de carrera {} (mediana ritmo)'.format(dt.timedelta(seconds=int(tps4[1])))
print 'Tiempo real de carrera {}'.format(dt.timedelta(seconds=(vd4.time.iloc[-1])))
print 'Error: {0} ({1:0.1f}%)\n'.format(dt.timedelta(seconds=abs(vd4.time.iloc[-1] - round(int(tps4[1])))),
                                           abs(vd4.time.iloc[-1] - round(int(tps4[1])))/vd4.time.iloc[-1] * 100)

time_runner = np.array(validation_data.time)
tr1 = np.array(vd1.time)
tr2 = np.array(vd2.time)
tr3 = np.array(vd3.time)
tr4 = np.array(vd4.time)

times = np.arange(len(efforts_index))
times1 = np.arange(len(eff_idx1))
times2 = np.arange(len(eff_idx2))
times3 = np.arange(len(eff_idx3))
times4 = np.arange(len(eff_idx4))

t1 = np.vectorize(lambda x: time_runner[x] + times_prediction_table[:len(times_prediction_table)-1-x].icol(int(efforts_index[x])).sum(axis=0))
t2 = np.vectorize(lambda x: times_prediction_table[:len(times_prediction_table)-1-x].icol(int(efforts_index[x])).sum(axis=0))

t1_1 = np.vectorize(lambda x: tr1[x] + tpt1[:len(tpt1)-1-x].icol(int(eff_idx1[x])).sum(axis=0))
t2_1 = np.vectorize(lambda x: tpt1[:len(tpt1)-1-x].icol(int(eff_idx1[x])).sum(axis=0))

t1_2 = np.vectorize(lambda x: tr2[x] + tpt2[:len(tpt2)-1-x].icol(int(eff_idx2[x])).sum(axis=0))
t2_2 = np.vectorize(lambda x: tpt2[:len(tpt2)-1-x].icol(int(eff_idx2[x])).sum(axis=0))

t1_3 = np.vectorize(lambda x: tr3[x] + tpt3[:len(tpt3)-1-x].icol(int(eff_idx3[x])).sum(axis=0))
t2_3 = np.vectorize(lambda x: tpt3[:len(tpt3)-1-x].icol(int(eff_idx3[x])).sum(axis=0))

t1_4 = np.vectorize(lambda x: tr4[x] + tpt4[:len(tpt4)-1-x].icol(int(eff_idx4[x])).sum(axis=0))
t2_4 = np.vectorize(lambda x: tpt4[:len(tpt4)-1-x].icol(int(eff_idx4[x])).sum(axis=0))

times_prediction_total = t1(times) / 60
tptotal1 = t1_1(times1) / 60
tptotal2 = t1_2(times2) / 60
tptotal3 = t1_3(times3) / 60
tptotal4 = t1_4(times4) / 60
times_prediction_to_goal = t2(times) / 60
tpg1 = t2_1(times1) / 60
tpg2 = t2_2(times2) / 60
tpg3 = t2_3(times3) / 60
tpg4 = t2_4(times4) / 60

laps = np.arange(0,10000,2000)

fig = plt.figure()
ax1 = fig.add_subplot(211)
im=ax1.plot(distances,validation_data.elevation,'k',zorder=1)
ax1.set_title('Elevation',fontsize=16,y=1.02)
ax1.set_ylabel('Elevation ['r'$m$]')
ax1.grid(axis='y', alpha=0.5)
ax1.set_yticks(ax1.get_yticks()[::2])

ax2 = fig.add_subplot(212)
ax2.plot(distances, times_prediction_total,c='black',zorder=1)
#ax2.scatter(laps,laps_times_total-0.1,c='orange',zorder=3,linewidth=0)
ax2.set_title('Estimated time',fontsize=16,y=1.02)
ax2.set_xlabel('Distance ['r'$m$]')
ax2.set_ylabel('Time ['r'$min$]')
ax2.grid(axis='y', alpha=0.5)
ax2.set_yticks(ax2.get_yticks()[::2])

ax1.set_xlim(ax2.get_xlim())
for l in laps:
    ax1.axvline(x=l,color='orange',zorder=2)
    ax2.axvline(x=l,color='orange',zorder=2)
plt.tight_layout()

fig = plt.figure()
ax1 = fig.add_subplot(211)
im=ax1.plot(distances,validation_data.elevation,'k',zorder=1)
ax1.set_title('Elevation',fontsize=16,y=1.02)
ax1.set_ylabel('Elevation ['r'$m$]')
ax1.grid(axis='y', alpha=0.5)
ax1.set_yticks(ax1.get_yticks()[::2])

ax2 = fig.add_subplot(212)
ax2.plot(np.concatenate([d1,d2,d3,d4]),np.concatenate([tpg1,tpg2,tpg3,tpg4]),'k',zorder=1)
#ax2.scatter(laps,laps_times_total-0.1,c='orange',zorder=3,linewidth=0)
ax2.set_title('Estimated time',fontsize=16,y=1.02)
ax2.set_xlabel('Distance ['r'$m$]')
ax2.set_ylabel('Time ['r'$min$]')
ax2.grid(axis='y', alpha=0.5)
ax2.set_yticks(np.array([-10.,  10.,  30.,  50.,  70.]))
ax2.set_xticks(np.array([ -2000.,      0.,   2000.,   4000.,   6000.,   8000.,  10000., 12000.]))
ax1.set_xlim(ax2.get_xlim())

for l in np.array((0,d1[-1],d2[-1],d3[-1],d4[-1])):
    ax1.axvline(x=l,color='orange',zorder=2)
    ax2.axvline(x=l,color='orange',zorder=2)

plt.tight_layout()
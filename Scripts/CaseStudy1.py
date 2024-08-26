# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 14:58:06 2015

@author: aRa
"""

#%pylab qt
import glob
import os
import numpy as np
import pandas as pd
import datetime as dt
import dateutil.parser
import matplotlib.pyplot as plt

def convert(date):
    dt = dateutil.parser.parse(date).replace(tzinfo=None)
    epoch = dt.utcfromtimestamp(0)
    delta = dt - epoch
    return delta.total_seconds()

def pace(speed):
    miuntes, seconds = divmod(100 / (6 * speed), 1)
    seconds *= 60
    return (miuntes, seconds)   

path = "/Users/aRa/OneDrive/Documentos/ERASMUS/PFC/SmartCoach/Running_w_HR/APE_runs_oct26-mar22"
colnames = np.array(['time', 'elevation', 'distance', 'speed', 'HR'])
means = []
medians = []

Datasets_all = []
os.chdir(path)
for file in glob.glob("*.tab"):
    ds = np.genfromtxt(file, skip_header=1, delimiter='\t', converters={0: convert})
    ds[:,0] -= ds[0,0]
      
    if (ds[:,2].max() > 10000) == (ds[:,2].max() < 13000):
        ds = pd.DataFrame(ds,columns=colnames)
        dataset_new = ds.drop(['time','elevation','HR'], axis=1)
        Datasets_all.append(dataset_new)

datasets = pd.concat(Datasets_all,ignore_index=True)
datasets.sort(columns='distance', inplace=True)
datasets.index = np.arange(len(datasets))
print "Velocidad mediana: {:.2f} m/s".format(datasets.speed.median())
print "Velocidad media: {:.2f} m/s".format(datasets.speed.mean())
print "Mediana ritmo carreras anteriores {0:.0f}:{1:.0f} min/km".format(pace(datasets.speed.median())[0],pace(datasets.speed.median())[1])
print "Ritmo promedio carreras anteriores {0:.0f}:{1:.0f} min/km".format(pace(datasets.speed.mean())[0],pace(datasets.speed.mean())[1])

path = "/Users/aRa/OneDrive/Documentos/ERASMUS/PFC/SmartCoach/Running_w_HR/TEST/"
os.chdir(path)
file == 'test_activity.tab'
validation_data = np.genfromtxt(file, skip_header=1, delimiter='\t', converters={0: convert})
validation_data[:,0] -= validation_data[0,0]
validation_data = pd.DataFrame(validation_data,columns=colnames)
distances = np.array(validation_data.distance)
distance_diffs = np.concatenate([[distances[0]], distances[1:] - distances[:len(distances)-1]])

time_runner = np.array(validation_data.time)
times_prediction = distance_diffs / datasets.speed.median()
times = np.arange(len(time_runner))
t1 = np.vectorize(lambda x: time_runner[x] + times_prediction[:len(times_prediction)-1-x].sum(axis=0))
times_prediction_total = t1(times) / 60

fig = plt.figure()
ax1 = fig.add_subplot(211)
im=ax1.plot(distances,validation_data.elevation,'k',zorder=1)
ax1.set_title('Elevation',fontsize=16,y=1.02)
ax1.set_ylabel('Elevation ['r'$m$]')
ax1.grid(axis='y', alpha=0.5)
ax1.set_yticks(ax1.get_yticks()[::2])
plt.setp(ax1.get_xticklabels(), visible=False)

ax2 = fig.add_subplot(212)
ax2.plot(distances, times_prediction_total,c='black',zorder=1)
#ax2.scatter(laps,laps_times_total-0.1,c='white',zorder=1,linewidth=0)
ax2.axhline(y=time_runner[-1]/60, color='magenta')
ax2.text(0.95,0.95,'Real time {}'.format(dt.timedelta(seconds=time_runner[-1])), 
         verticalalignment='top', horizontalalignment='right', transform=ax2.transAxes,
         color='magenta', fontsize=14)
ax2.set_title('Estimated time',fontsize=16,y=1.02)
ax2.set_xlabel('Distance ['r'$m$]')
ax2.set_ylabel('Time ['r'$min$]')
ax2.grid(alpha=0.5)
ax2.set_ylim([time_min - (time_max-time_min)/2., time_max + (time_max-time_min)/2])
ax2.set_yticks(ax2.get_yticks()[::2])
ax2.set_xticks(np.arange(-2000,14000, 2000))
ax1.set_xlim(ax2.get_xlim())
plt.tight_layout()
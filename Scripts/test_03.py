# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 12:02:59 2015

@author: aRa


Heatmaps of HR as a function of speed and slope slicing the data into four 
windows depending on the moment of the run

"""
import glob
import os
import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import matplotlib.mlab as ml
import dateutil.parser
from datetime import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable

path = "/Users/aRa/OneDrive/Documentos/ERASMUS/PFC/SmartCoach/Running_w_HR/APE_runs_oct26-mar22"

def convert(date):
    dt = dateutil.parser.parse(date).replace(tzinfo=None)
    epoch = datetime.utcfromtimestamp(0)
    delta = dt - epoch
    return delta.total_seconds()

colnames = np.array(['time', 'elevation', 'distance', 'speed', 'HR'])
Datasets_all = []
window_size_half = 3

os.chdir(path)
for file in glob.glob("*.tab"):
    print "Processing " + file
    dataset = np.genfromtxt(file, skip_header=1,delimiter='\t', converters={0: convert})
    dataset[:,0] -= dataset[0,0]
    dataset = pd.DataFrame(dataset,columns=colnames)
    
    if (dataset.distance.max() >
    11000) == (dataset.distance.max() < 13000):
        slope = np.array([])
        for j in dataset.index:
            index = np.arange(j-window_size_half+1, j+window_size_half+1)
            index = index[(index >= 0) & (index < len(dataset))]
            dataset_part = dataset.iloc[index].dropna()
            regr = lm.LinearRegression()
            regr.fit(dataset_part.distance[:,np.newaxis], np.array(dataset_part.elevation))
            slope = np.append(slope,regr.coef_)
    
        dataset['slope'] = slope        
        Datasets_all.append(dataset)

window = np.array([0., 0.25, 0.5, 0.75, 1.])

datasets = pd.concat(Datasets_all,ignore_index=True).dropna()
datasets.sort(columns='distance', inplace=True)

tot_dist = datasets.distance.values[-1]

ds_mid1 = datasets.query('(distance >= %f) and (distance <= %f)' % (window[0] * tot_dist, window[1] * tot_dist))
ds_mid2 = datasets.query('(distance >= %f) and (distance <= %f)' % (window[1] * tot_dist, window[2] * tot_dist))
ds_mid3 = datasets.query('(distance >= %f) and (distance <= %f)' % (window[2] * tot_dist, window[3] * tot_dist))
ds_mid4 = datasets.query('(distance >= %f) and (distance <= %f)' % (window[3] * tot_dist, window[4] * tot_dist))

x = [ds_mid1.speed, ds_mid2.speed, ds_mid3.speed, ds_mid4.speed]
y = [ds_mid1.slope, ds_mid2.slope, ds_mid3.slope, ds_mid4.slope]
z = [ds_mid1.HR, ds_mid2.HR, ds_mid3.HR, ds_mid4.HR]

binsize = 0.1
bins = 20

xmin = min(x[0].min(), x[1].min())
xmax = max(x[0].max(), x[1].max())
ymin = min(y[0].min(), y[1].min())
ymax = max(x[0].max(), x[1].max())

xx = [np.linspace(x[0].min(),x[0].max(), bins), np.linspace(x[1].min(),x[1].max(), bins),
      np.linspace(x[2].min(),x[2].max(), bins), np.linspace(x[3].min(),x[3].max(), bins)]
yy = [np.linspace(y[0].min(),y[0].max(), bins), np.linspace(y[1].min(),y[1].max(), bins),
      np.linspace(y[2].min(),y[2].max(), bins), np.linspace(y[3].min(),y[3].max(), bins)]
zz = [ml.griddata(x[0], y[0], z[0], xx[0], yy[0], interp='nn'),
      ml.griddata(x[1], y[1], z[1], xx[1], yy[1], interp='nn'),
      ml.griddata(x[2], y[2], z[2], xx[2], yy[2], interp='nn'),
      ml.griddata(x[3], y[3], z[3], xx[3], yy[3], interp='nn')]

      
zmin    = [zz[0][np.where(np.isnan(zz[0]) == False)].min(), 
           zz[1][np.where(np.isnan(zz[1]) == False)].min(),
           zz[2][np.where(np.isnan(zz[2]) == False)].min(), 
           zz[3][np.where(np.isnan(zz[3]) == False)].min()]
           
zmax    = [zz[0][np.where(np.isnan(zz[0]) == False)].max(), 
           zz[1][np.where(np.isnan(zz[1]) == False)].max(),
           zz[2][np.where(np.isnan(zz[2]) == False)].max(), 
           zz[3][np.where(np.isnan(zz[3]) == False)].max()]

extent = (min(x[0].min(),x[1].min()), max(x[0].max(),x[1].max()), min(y[0].min(),y[1].min()),max(y[0].max(),y[1].max()))

# Using pcolormesh
fig = plt.figure(figsize=(6,5), dpi=72)
ax1 = fig.add_subplot(221)
im1 = ax1.pcolormesh(xx[0], yy[0]*100, zz[0], cmap = plt.cm.jet,
               vmin=min(zmin), vmax=max(zmax))
divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right", size="5%", pad=0.05)
cbar1 = plt.colorbar(im1, cax=cax1)
ax1.set_ylabel('Slope (%)')

ax2 = fig.add_subplot(222)
im2 = ax2.pcolormesh(xx[1], yy[1]*100, zz[1], cmap = plt.cm.jet, 
               vmin=min(zmin), vmax=max(zmax))
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("right", size="5%", pad=0.05)
cbar2 = plt.colorbar(im2, cax=cax2)

ax3 = fig.add_subplot(223)
im3 = ax3.pcolormesh(xx[2], yy[2]*100, zz[2], cmap = plt.cm.jet, 
               vmin=min(zmin), vmax=max(zmax))
divider3 = make_axes_locatable(ax3)
cax3 = divider3.append_axes("right", size="5%", pad=0.05)
cbar3 = plt.colorbar(im3, cax=cax3)
ax3.set_xlabel('Speed (m/s)')
ax3.set_ylabel('Slope (%)')

ax4 = fig.add_subplot(224)
im4 = ax4.pcolormesh(xx[3], yy[3]*100, zz[3], cmap = plt.cm.jet, 
               vmin=min(zmin), vmax=max(zmax))
divider4 = make_axes_locatable(ax4)
cax4 = divider4.append_axes("right", size="5%", pad=0.05)
cbar4 = plt.colorbar(im4, cax=cax4)
ax4.set_xlabel('Speed (m/s)')
for ax, title in zip([ax1, ax2, ax3, ax4], ['(a)','(b)','(c)','(d)']):
    ax.set_xlim([0., 6.5])
    ax.set_ylim([-0.3, 0.3])
    ax.set_yticks(ax.get_yticks()*100)
    ax.set_title(title, y=1.02)

fig.tight_layout()
fig.savefig('../../heatmap_HR_as_speed_and_slope_windowed_3.pdf', dpi=fig.dpi, bbox_inches='tight')
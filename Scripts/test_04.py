# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 17:27:23 2015

@author: aRa

Primera parte: plotear los evolucion de la velocidad en función del tiempo para 
            una pendiente dada.
            
Segunda parte: plotear la evolucion de la velocidad en función de la Energía 
            consumida acumulada.
            
Energía consumida acumulada = Suma del gasto enerjético por metro y por kg. 

Tercera parte: plotear los heatmaps en función del delta de HR     

Cuarta parte: comparar con feeling bad

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
import sklearn.neighbors as neighbors
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.axes_grid1 import make_axes_locatable

path = "/Users/aRa/OneDrive/Documentos/ERASMUS/PFC/SmartCoach/Running_w_HR/APE_runs_oct26-mar22"

def cumulated_energy(run_cost, distance, wheigt):
    return run_cost * distance * weight
    
def cost_running(slope):
    return 155.4 * slope**5 - 30.4 * slope**4 - 43.3 * slope**3 + 46.3 * slope**2 + 19.5 * slope + 3.6
    
def convert(date):
    dt = dateutil.parser.parse(date).replace(tzinfo=None)
    epoch = datetime.utcfromtimestamp(0)
    delta = dt - epoch
    return delta.total_seconds()

def group_by_slopes(slopes, X, nbins):
    bins = np.linspace(slopes.min(), slopes.max(), num=nbins + 1)
    bin_index = np.digitize(slopes, bins) - 1
    X_per_bin = []
    for bi in range(len(bins)-1):
        X_per_bin.append(X[bin_index == bi].values)
    return X_per_bin, bins
    
# Generate Datasetes
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


# =================  PARTE 1 =================

speed_per_bin, bins = group_by_slopes(datasets.slope, datasets.speed, nbins=20)
distances_per_bin,_ = group_by_slopes(datasets.slope, datasets.distance, nbins=20)
bin_mid = (bins[1:] + bins[:-1])/2

# Masking slopes
#mask = np.flatnonzero((bin_mid >= -0.15) == (bin_mid <= 0.15))
mask = [7, 9, 11, 12]
window_smooth = 20

# Plot speed by distance in % for each slope in mask
plt.figure()
for i, sl in enumerate(bin_mid[mask]):
    distance = distances_per_bin[mask[i]]/distances_per_bin[mask[i]][-1]
    speed = speed_per_bin[mask[i]]
    speed_averages = []
  
  # Averaging slopes for smothing plot
    for j in np.arange(len(speed)):
        idx = np.arange(j - window_smooth + 1, j + window_smooth + 1)
        idx = idx[(idx >= 0) & (idx < len(speed))] 
        sp_tmp = np.average(speed[idx])
        speed_averages.append(sp_tmp)
        
    plt.plot(distance, speed_averages, label="m = "+"%.3f"%sl)    

#plt.title(r"Window = ${}$".format(window_smooth), fontsize=16, y=1.02)
plt.ylim([1, 5])
plt.xlabel('Distance ['r'$\%$]')
plt.ylabel("Speed "r'$[m/s]$')
plt.legend(loc='best',fontsize='small')
plt.grid(alpha=0.5)
#plt.savefig("../Speed_per_distance_window_%d"%window_smooth)


# =================  PARTE 2 =================

weight = 72

plt.figure()
for i, sl in enumerate(bin_mid[mask]):
    distance = distances_per_bin[mask[i]]/distances_per_bin[mask[i]][-1]
    speed = speed_per_bin[mask[i]]
    run_cost = cost_running(sl)
    E = []
    Suma = []
    speed_averages = []
    
  # Compute cumulated energy for each sample of distance    
    for d in distance:
        E.append(cumulated_energy(run_cost,d,weight))
        Suma.append(sum(E))

  # Averaging slopes for smothing plot
    for j in np.arange(len(speed)):
        idx = np.arange(j - window_smooth + 1, j + window_smooth + 1)
        idx = idx[(idx >= 0) & (idx < len(speed))] 
        sp_tmp = np.average(speed[idx])
        speed_averages.append(sp_tmp)
    plt.plot(Suma/Suma[-1], speed_averages, label="m = "+"%.3f"%sl)

plt.title('Energy consumption', fontsize=16, y=1.02)
plt.ylim([1, 5])
plt.xlabel('Energy')
plt.ylabel('Speed' r'$[m/s]$')
plt.legend(loc='best',fontsize='small')
plt.grid(alpha=0.5)
#plt.savefig("../Speed_per_cumulated_energy_window_%d"%window_smooth)

# =================  PARTE 3 =================

window_distances = np.array([0., 0.25, 0.5, 0.75, 1.])

ds_mid1 = datasets.query('(distance >= %f) and (distance < %f)'
            % (window_distances[0], window_distances[1]))
ds_mid2 = datasets.query('(distance >= %f) and (distance < %f)'
            % (window_distances[1], window_distances[2]))
ds_mid3 = datasets.query('(distance >= %f) and (distance < %f)'
            % (window_distances[2], window_distances[3]))
ds_mid4 = datasets.query('(distance >= %f) and (distance <= %f)' 
            % (window_distances[3], window_distances[4]))

x = [ds_mid1.speed, ds_mid2.speed, ds_mid3.speed, ds_mid4.speed]
y = [ds_mid1.slope, ds_mid2.slope, ds_mid3.slope, ds_mid4.slope]
z = [ds_mid1.dHR, ds_mid2.dHR, ds_mid3.dHR, ds_mid4.dHR]

binsize = 0.1
bins = 20

xmin = min(x[0].min(), x[1].min(), x[2].min(), x[3].min())
xmax = max(x[0].max(), x[1].max(), x[2].max(), x[3].max())
ymin = min(y[0].min(), y[1].min(), y[2].min(), y[3].min())
ymax = max(x[0].max(), x[1].max(), y[2].max(), y[3].max())

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


# Using pcolormesh
xLim = [1.5, 5.]
yLim = [-0.3, 0.3]
fig = plt.figure(figsize=(12,4), dpi=72)
"""
ax1 = fig.add_subplot(221)
im1 = ax1.pcolormesh(xx[0], yy[0]*100, zz[0], cmap = plt.cm.jet,
               vmin=zmin[0], vmax=zmax[0])
ax1.set_xlim(xLim)
ax1.set_ylim(yLim)
ax1.set_yticks(ax1.get_yticks()*100)
ax1.set_xlabel('Speed ['r'$m$]')
ax1.set_ylabel('Slope ['r'$\%$]')
ax1.set_title(r'$\Delta$''HR 1st Window (0% - 25%)', fontsize=16, y=1.02)
"""
ax2 = fig.add_subplot(131)
im2 = ax2.pcolormesh(xx[1], yy[1]*100, zz[1], cmap = plt.cm.jet, 
                     vmin=min(zmin[1],zmin[2],zmin[3]), vmax=max(zmax[1],zmax[2],zmax[3]))

ax3 = fig.add_subplot(132)
im3 = ax3.pcolormesh(xx[2], yy[2]*100, zz[2], cmap = plt.cm.jet, 
                     vmin=min(zmin[1],zmin[2],zmin[3]), vmax=max(zmax[1],zmax[2],zmax[3]))
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("right", size="5%", pad=0.05)
cbar2 = plt.colorbar(im2, cax=cax2)
divider3 = make_axes_locatable(ax3)
cax3 = divider3.append_axes("right", size="5%", pad=0.05)
cbar3 = plt.colorbar(im3, cax=cax3)

ax4 = fig.add_subplot(133)
im4 = ax4.pcolormesh(xx[3], yy[3]*100, zz[3], cmap = plt.cm.jet, 
                     vmin=min(zmin[1],zmin[2],zmin[3]), vmax=max(zmax[1],zmax[2],zmax[3]))
divider4 = make_axes_locatable(ax4)
cax4 = divider4.append_axes("right", size="5%", pad=0.05)
cbar4 = plt.colorbar(im4, cax=cax4)
for ax, title in zip([ax2, ax3, ax4], ['(a)','(b)','(c)']):
    ax.set_xlabel('Speed (m/s)')
    ax.set_xlim(xLim)
    ax.set_ylim(yLim)
    ax.set_yticks(ax.get_yticks()*100)
    ax.set_title(title, y=1.02)
ax2.set_ylabel('Slope (%)')
plt.tight_layout()
fig.savefig("../../heatmapDeltaHR3windowsFiltered.pdf", dpi=fig.dpi, bbox_inches='tight')
# =================  PARTE 4 =================

path = "/Users/aRa/OneDrive/Documentos/ERASMUS/PFC/SmartCoach/Running_w_HR/TEST"

os.chdir(path)
file = "Morat_Fribourg2014.tab"
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


ndata = np.array(datasets.drop(['distance'], axis=1))
scaler = MinMaxScaler().fit(ndata)
ndata = scaler.transform(ndata)

ndata_fb = scaler.transform(dataset_fb.drop(['distance'], axis=1))

nn = neighbors.NearestNeighbors()
nn.fit(ndata)

dist, ind = nn.kneighbors(ndata_fb, n_neighbors=2)
dist = np.squeeze(dist)
ind = np.squeeze(ind)
nearest_hr = ind[:,0]
        
plt.figure()
#plt.plot(dataset_fb.distance, dataset_fb.dHR, label='Feeling bad')
#plt.plot(dataset_fb.distance, datasets.dHR.iloc[nearest_hr],
#         label='Nearest approximation', c='r')
plt.scatter(datasets.distance, datasets.dHR, label='Database', c='k', linewidth=0)
plt.scatter(dataset_fb.distance, dataset_fb.dHR, label='Feeling Bad', c='b', linewidth=0)
plt.scatter(datasets.distance.iloc[nearest_hr], datasets.dHR.iloc[nearest_hr],
            label='Nearest approximation', c='g', linewidth=0)
plt.legend(loc=4, fontsize='small')
plt.xlabel('Distance ['r'$\%$]')
plt.ylabel(r'$\Delta$''HR/HR')

plt.figure()
plt.scatter(datasets.slope, datasets.dHR, label='Database', c='k', linewidth=0)
plt.scatter(dataset_fb.slope, dataset_fb.dHR, label='Feeling Bad', c='b', linewidth=0)
plt.legend(loc='best', fontsize='small')
plt.xlabel('Slope ['r'$\%$]')
plt.ylabel(r'$\Delta$''HR/HR')

plt.figure()
plt.scatter(datasets.speed, datasets.dHR, label='Database', c='k', linewidth=0)
plt.scatter(dataset_fb.speed, dataset_fb.dHR, label='Feeling Bad', c='b', linewidth=0)
plt.legend(loc='best', fontsize='small')
plt.xlabel('Speed ['r'$m$]')
plt.ylabel(r'$\Delta$''HR/HR')


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.itervalues():
        sp.set_visible(False)

fig, host = plt.subplots(figsize=(6,4), dpi=72)
fig.subplots_adjust(right=0.75)

par1 = host.twinx()
par2 = host.twinx()

par2.spines["right"].set_position(("axes", 1.1))
make_patch_spines_invisible(par2)
par2.spines["right"].set_visible(True)

p1, = host.plot(fb.time/60, fb.elevation, color='k', label='Elevation['r'$m$]',alpha=0.2)
host.fill_between(fb.time/60, 0, fb.elevation, color='k', alpha=0.2)
p2, = par1.plot(fb.time/60, fb.HR, color='k', linestyle=':', label='HR['r'$bpm$]')
p3, = par2.plot(fb.time/60, fb.speed, color='k', linestyle='--', label='Speed['r'$m/s$]')

host.set_xlim([min(fb.time/60), max(fb.time/60)])
#host.set_ylim([650, 1000])

host.set_xlabel('Time ['r'$min$]')

tkw = dict(size=4, width=1.5)
host.tick_params(axis='y', colors=p1.get_color(), **tkw)
par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
host.tick_params(axis='x', **tkw)

host.legend((p1, p2, p3),(p1.get_label(), p2.get_label(), p3.get_label()),
           bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.,
            fontsize='small')


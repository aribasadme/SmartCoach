# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 10:01:34 2015

@author: aRa
"""

import glob
import os
import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import dateutil.parser
from datetime import datetime

path = "/Users/aRa/OneDrive/Documentos/ERASMUS/PFC/SmartCoach/Running_w_HR/APE_runs_oct26-mar22"

def convert(date):
    """
    Converts a date from strings and calculates the time delta since epoch timestamp

    Parameters
    ----------
    date : any daytime format
    
    Returns
    -------
    delta : datetime.timedelta
        timedelta from epoch to date
    """
    dt = dateutil.parser.parse(date).replace(tzinfo=None)
    
    epoch = datetime.utcfromtimestamp(0)
    delta = dt - epoch
   
    return delta.total_seconds()
    
colnames = np.array(['time', 'elevation', 'distance', 'speed', 'HR'])
Datasets_all = []

# Set directory from where to load files
os.chdir(path)

for file in glob.glob("*.tab"):
    print "Processing " + file
    
    # Read the table from file
    dataset = np.genfromtxt(file, skip_header=1,delimiter='\t', 
                            converters={0: convert})
    
    # Set time reference from the first entry                        
    dataset[:,0] -= dataset[0,0]
    
    dataset = pd.DataFrame(dataset,columns=colnames)
    slope = np.array([])
    
    # Computes slope with a window of size 2*window_size_half
    window_size_half = 8    
    for i in dataset.index:
        index = np.arange(i - window_size_half + 1, i + window_size_half + 1)
        index = index[(index >= 0) & (index < len(dataset))]
        dataset_part = dataset.iloc[index].dropna()
        regr = lm.LinearRegression()
        regr.fit(dataset_part.distance[:,np.newaxis], 
                 np.array(dataset_part.elevation))
        slope = np.append(slope,regr.coef_)
    
    # Add new column slope to the dataset
    dataset['slope'] = slope
    
    if (len(dataset) > 300) == (len(dataset) < 900):
        Datasets_all.append(dataset)


dist_w_half = 5

dataset_part = dataset[['time','speed','HR','slope']]
for i in dataset_part:
    index = np.arange(i - dist_w_half + 1, i + dist_w_half + 1)
    index = index[(index >= 0) & (index < len(dataset_part))]
    ds_tmp = dataset_part.iloc[index]
    regr.fit(ds_tmp.HR[:,np.newaxis])
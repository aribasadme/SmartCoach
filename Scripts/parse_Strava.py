# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 14:51:22 2015

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
import matplotlib.mlab as ml
from mpl_toolkits.axes_grid1 import make_axes_locatable

def convert(time):
    pace = dt.datetime.strptime(time,'%M:%S')
    pace = pace.minute + pace.second / 60.
    return pace

path = "/Users/aRa/OneDrive/Documentos/ERASMUS/PFC/SmartCoach/"
os.chdir(path)
file = 'APEruns_Strava_GAP_pace.xlsx'
Strava = pd.read_excel(file, converters={0: convert, 1: convert})
import numpy as np
import pandas as pd
import glob
import os
import dateutil.parser as dt
import xml.etree.ElementTree as ET

#path = "/Users/aRa/OneDrive/Documentos/ERASMUS/PFC/SmartCoach/firstTestsR/APE_recent_races"
#path = "/Users/aRa/OneDrive/Documentos/ERASMUS/PFC/SmartCoach/firstTestsR/data_corrected"
path = "/Users/aRa/OneDrive/Documentos/ERASMUS/PFC/SmartCoach/Running_w_HR/APE_runs_oct26-mar22"

namespace="{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}"
tpx_namespace="{http://www.garmin.com/xmlschemas/ActivityExtension/v2}"
colnames = np.array(['time', 'elevation', 'distance', 'speed'])

def get_child_text_or_none(parent, childname):
    child = parent.find(childname)
    if child is not None:
        return child.text
    else:
        return None

os.chdir(path)
for file in glob.glob("*.tcx"):
    print "Parsing " + file
    tree = ET.parse(file)
    root = tree.getroot()
    lapsxml = root[0][0]	   # Contain tag 'Activity' & sons
    time = np.array([])
    elevation = np.array([])
    distance = np.array([])
    speed = np.array([])
    
    for lap in lapsxml.findall(namespace+"Lap"):
        tracks = lap.findall(namespace+"Track")
        if len(tracks) > 0:
            track = tracks[0]
            for trackpoint in track.findall(namespace+"Trackpoint"):
                time = np.append(time, dt.parse(get_child_text_or_none(trackpoint, namespace+"Time"), ignoretz=True))
                elevation  = np.append(elevation, get_child_text_or_none(trackpoint, namespace+"AltitudeMeters"))
                speed = np.append(speed, get_child_text_or_none(trackpoint,
                    "./{0}Extensions/{1}TPX/{1}Speed".format(namespace, tpx_namespace)))
                distance = np.append(distance, get_child_text_or_none(trackpoint, namespace+"DistanceMeters"))
    
    laps = np.array([time, elevation, distance,speed])    
    activity = pd.DataFrame([time,elevation,distance,speed], colnames).T        
    
    activityfile = open(file[:-4]+" copia"+".tab", 'w')    
    acivity_csv = activity.to_csv(activityfile,sep='\t',na_rep='NA',index=False, float_format='%.12f')
    activityfile.close()

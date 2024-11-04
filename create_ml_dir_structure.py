#!/usr/bin/python
'''Read from PyMongo, make simple model and export for CoreML'''

# users might be interested in the following 
# documentation from Apple on gathering and training and app for activity detection (WWDC 2019)
# https://wwdcnotes.com/documentation/wwdcnotes/wwdc19-426-building-activity-classification-models-in-create-ml/

# this script creates a dierectory structure that CreateML can read from and make use of for 
# acitvity classification based on our uploaded data to mongoDB. 
# Becasue of this, you do NOT need to use python 3.8 (as we no longer depend on turi)

# database imports
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
import os

# model imports
import numpy as np
import pandas as pd

# export 
#import coremltools


dsids = [5]
client  = MongoClient(serverSelectionTimeoutMS=50)
db = client.turidatabase

def get_features_and_labels(dsid):
    # create feature vectors from database
    features=[]
    labels=[]
    for a in db.labeledinstances.find({"dsid":dsid}): 
        feat_row = np.array([float(val) for val in a['feature']])
        features.append(feat_row.reshape((1,-1)))
        labels.append(a['label'])

    features = np.vstack(features)
    labels = np.array(labels).reshape((-1,1))

    # convert to dictionary for tc
    data = {'target':labels, 'sequence': features}
    
    return data
  

# ================================================
#      Main Script
#-------------------------------------------------

data = {'target':[], 'sequence': []}
for idx,dsid in enumerate(dsids):
    print("Getting data from Mongo db for dsid=", dsid)
    data_tmp = get_features_and_labels(dsid)
    print("Found",len(data_tmp['sequence']),"labels and feature vectors")

    if idx != 0:
        data['sequence'] = np.vstack((data['sequence'],data_tmp['sequence']))
        data['target'] = np.vstack((data['target'],data_tmp['target']))
    else:
        data = data_tmp

# close the mongo connection, now that we have the data
client.close() 

classes = np.unique(data['target'])

print("Found",len(data['sequence']),"labels and feature vectors")
print("Unique classes found:",classes)


print("Exporting Dataset structure for CreateML")
# format should create directories for each class
path_to_save = "../data"
for y in classes:
    dir_name = f"{path_to_save}/{y}/"
    print(dir_name) 
    if not os.path.exists(dir_name):
        os.mkdir(dir_name) # make directory 

# examples for a given class should be a csv with each column being the feature values
# since we flattened the data, we need to unflatten for 50 point of x/y/z accel data
num_instances = {x:0 for x in classes} # number of examples written for each class

for X,y in zip(data['sequence'],data['target']):
    filename = f"{path_to_save}/{y[0]}/{num_instances[y[0]]}.csv"
    print(filename) # label is used to specify the directory for saving

    # restructure X data, make into table data, save csv
    x_data = X[0::3].copy().reshape((50,1)) # get every first element from x/y/z
    y_data = X[1::3].copy().reshape((50,1)) # get every second element from x/y/z
    z_data = X[2::3].copy().reshape((50,1)) # get every third element from x/y/z 
    features = np.hstack((x_data,y_data,z_data))


    df = pd.DataFrame(data=features,columns=["x","y","z"])
    df.to_csv(filename)

    num_instances[y[0]] += 1














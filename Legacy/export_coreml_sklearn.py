#!/usr/bin/python
'''Read from PyMongo, make simple model and export for CoreML'''

# database imports
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

# model imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

import numpy as np

# export 
import coremltools


# To use the scikit-learn outputs, you need to have the following installed:
# these can be done on an install without using Rosetta! No need to be i386
# python3 -m pip install --upgrade pip 
# pip3 install numpy   
# pip3 install scikit-learn==1.1.2
# pip3 install coremltools
# pip3 install pymongo


dsid = 1
client  = MongoClient(serverSelectionTimeoutMS=50)
db = client.turidatabase


# create feature/label vectors from database
X=[];
y=[];
for a in db.labeledinstances.find({"dsid":dsid}): 
    X.append([float(val) for val in a['feature']])
    y.append(a['label'])    


print("Found",len(y),"labels and",len(X),"feature vectors")
print("Unique classes found:",np.unique(y))

clf_rf = RandomForestClassifier(n_estimators=150)
clf_svm = SVC()
clf_pipe = Pipeline([("SCL", StandardScaler()),
	("SVC",SVC())])
clf_gb = GradientBoostingClassifier()

print("Training Model", clf_rf)

clf_rf.fit(X,y)
clf_svm.fit(X,y)
clf_pipe.fit(X,y)
clf_gb.fit(X,y)

print("Exporting to CoreML")

coreml_model = coremltools.converters.sklearn.convert(
	clf_rf) 

# save out as a file
coreml_model.save('../RandomForestAccel.mlmodel')


coreml_model = coremltools.converters.sklearn.convert(
	clf_svm) 

# save out as a file
coreml_model.save('../SVMAccel.mlmodel')

coreml_model = coremltools.converters.sklearn.convert(
	clf_pipe) 

# save out as a file
coreml_model.save('../PipeAccel.mlmodel')

coreml_model = coremltools.converters.sklearn.convert(
	clf_gb) 

# save out as a file
coreml_model.save('../GradientAccel.mlmodel')
 

# close the mongo connection

client.close() 
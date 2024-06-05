#!/usr/bin/python
'''
In this example, we will use FastAPI as a gateway into a MongoDB database. We will use a REST style 
interface that allows users to initiate GET, POST, PUT, and DELETE requests. These commands will 
also be used to control certain functionalities with machine learning, using the ReST server to
function as a machine learning as a service, MLaaS provider. 

Specifically, we are creating an app that can take in motion sampled data and labels for 
segments of the motion data

The swift code for interacting with the interface is also available through the SMU MSLC class 
repository. 
Look for the https://github.com/SMU-MSLC/SwiftHTTPExample with branches marked for FastAPI and
turi create

To run this example in localhost mode only use the command:
fastapi dev fastapi_turicreate.py

Otherwise, to run the app in deployment mode (allowing for external connections), use:
fastapi run fastapi_turicreate.py

External connections will use your public facing IP, which you can find from the inet. 
A useful command to find the right public facing ip is:
ifconfig |grep "inet "
which will return the ip for various network interfaces from your card. 
'''

# For this to run properly, MongoDB should be running
#    To start mongo use this: brew services start mongodb-community@6.0
#    To stop it use this: brew services stop mongodb-community@6.0

# This App uses a combination of FastAPI and Motor (combining tornado/mongodb) which have documentation here:
# FastAPI:  https://fastapi.tiangolo.com 
# Motor:    https://motor.readthedocs.io/en/stable/api-tornado/index.html

# Maybe the most useful SO answer for FastAPI parallelism:
# https://stackoverflow.com/questions/71516140/fastapi-runs-api-calls-in-serial-instead-of-parallel-fashion/71517830#71517830
# Chris knows what's up 



import os
from typing import Optional, List
from enum import Enum

# FastAPI imports
from fastapi import FastAPI, Body, HTTPException, status
from fastapi.responses import Response
from pydantic import ConfigDict, BaseModel, Field, EmailStr
from pydantic.functional_validators import BeforeValidator

from typing_extensions import Annotated

# Motor imports
from bson import ObjectId
import motor.motor_asyncio
from pymongo import ReturnDocument

# Machine Learning, Turi and Sklearn Imports
import turicreate as tc
from sklearn.neighbors import KNeighborsClassifier

from joblib import dump, load
import pickle
import numpy as np


# define some things in API
async def db_lifespan(app: FastAPI):
    # Motor API allows us to directly interact with a hosted MongoDB server
    # In this example, we assume that there is a single client 
    # First let's get access to the Mongo client that allows interactions locally 
    app.mongo_client = motor.motor_asyncio.AsyncIOMotorClient()

    # new we need to create a database and a collection. These will create the db and the 
    # collection if they haven't been created yet. They are stored upon the first insert. 
    db = app.mongo_client.turidatabase
    app.collection = db.get_collection("labeledinstances")

    app.clf = [] # start app with no classifier

    yield 

    app.mongo_client.close()


# Create the FastAPI app
app = FastAPI(
    title="Machine Learning as a Service",
    summary="An application using FastAPI to add a ReST API to a MongoDB for data and labels collection.",
    lifespan=db_lifespan,
)



# Represents an ObjectId field in the database.
# It will be represented as a `str` on the model so that it can be serialized to JSON.
PyObjectId = Annotated[str, BeforeValidator(str)]



#========================================
#   Data store objects from pydantic 
#----------------------------------------
# These allow us to create a schema for our database and access it easily with FastAPI
# That might seem odd for a document DB, but its not! Mongo works faster when objects
# have a similar schema. 

'''Create the data model and use strong typing. This also helps with the use of intellisense.
'''
class LabeledDataPoint(BaseModel):
    """
    Container for a single labeled data point.
    """

    # This will be aliased to `_id` when sent to MongoDB,
    # but provided as `id` in the API requests and responses.
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    feature: List[float] = Field(...) # feature data as array
    label: str = Field(...) # label for this data
    dsid: int = Field(..., le=50) # dataset id, for tracking different sets
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_schema_extra={ # provide an example for FastAPI to show users
            "example": {
                "feature": [-0.6,4.1,5.0,6.0],
                "label": "Walking",
                "dsid": 2,
            }
        },
    )


class LabeledDataPointCollection(BaseModel):
    """
    A container holding a list of instances.

    This exists because providing a top-level array in a JSON response can be a [vulnerability](https://haacked.com/archive/2009/06/25/json-hijacking.aspx/)
    """

    datapoints: List[LabeledDataPoint]


class FeatureDataPoint(BaseModel):
    """
    Container for a single labeled data point.
    """

    # This will be aliased to `_id` when sent to MongoDB,
    # but provided as `id` in the API requests and responses.
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    feature: List[float] = Field(...) # feature data as array
    dsid: int = Field(..., le=50) # dataset id, for tracking different sets
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_schema_extra={ # provide an example for FastAPI to show users
            "example": {
                "feature": [-0.6,4.1,5.0,6.0],
                "dsid": 2,
            }
        },
    )



#===========================================
#   FastAPI methods, for interacting with db 
#-------------------------------------------
# These allow us to interact with the REST server. All interactions with mongo should be 
# async, allowing the API to remain responsive even when servicing longer queries. 


@app.post(
    "/labeled_data/",
    response_description="Add new labeled datapoint",
    response_model=LabeledDataPoint,
    status_code=status.HTTP_201_CREATED,
    response_model_by_alias=False,
)
async def create_datapoint(datapoint: LabeledDataPoint = Body(...)):
    """
    Insert a new data point. Let user know the range of values inserted

    A unique `id` will be created and provided in the response.
    """
    
    # insert this datapoint into the database
    new_label = await app.collection.insert_one(
        datapoint.model_dump(by_alias=True, exclude=["id"])
    )

    # send back info about the record
    created_label = await app.collection.find_one(
        {"_id": new_label.inserted_id}
    )
    # also min/max of array, rather than the entire to array to save some bandwidth
    # the datapoint variable is a pydantic model, so we can access with properties
    # but the output of mongo is a dictionary, so we need to subscript the entry
    created_label["feature"] = [min(datapoint.feature), max(datapoint.feature)]

    return created_label


@app.get(
    "/labeled_data/{dsid}",
    response_description="List all labeled data in a given dsid",
    response_model=LabeledDataPointCollection,
    response_model_by_alias=False,
)
async def list_datapoints(dsid: int):
    """
    List all of the data for a given dsid in the database.

    The response is unpaginated and limited to 1000 results.
    """
    return LabeledDataPointCollection(datapoints=await app.collection.find({"dsid": dsid}).to_list(1000))


@app.get(
    "/max_dsid/",
    response_description="Get current maximum dsid in data",
    response_model=int,
    response_model_by_alias=False,
)
async def show_max_dsid():
    """
    Get the maximum dsid currently used 
    """

    if (
        datapoint := await app.collection.find_one(sort=[("dsid", -1)])
    ) is not None:
        return datapoint["dsid"]

    raise HTTPException(status_code=404, detail=f"No datasets currently created.")



@app.delete("/labeled_data/{dsid}", 
    response_description="Delete an entire dsid of datapoints.")
async def delete_dataset(dsid: int):
    """
    Remove an entire dsid from the database.
    REMOVE AN ENTIRE DSID FROM THE DATABASE, USE WITH CAUTION.
    """

    # replace any underscores with spaces (to help support others)

    delete_result = await app.collection.delete_many({"dsid": dsid})

    if delete_result.deleted_count > 0:
        return {"num_deleted_results":delete_result.deleted_count}

    raise HTTPException(status_code=404, detail=f"DSID {dsid} not found")




#===========================================
#   Machine Learning methods (Turi)
#-------------------------------------------
# These allow us to interact with the REST server with ML from Turi. 

@app.get(
    "/train_model_turi/{dsid}",
    response_description="Train a machine learning model for the given dsid",
    response_model_by_alias=False,
)
async def train_model_turi(dsid: int):
    """
    Train the machine learning model using Turi
    """

    # convert data over to a scalable dataframe

    datapoints = await app.collection.find({"dsid": dsid}).to_list(length=None)

    if len(datapoints) < 2:
        raise HTTPException(status_code=404, detail=f"DSID {dsid} has {len(datapoints)} datapoints.") 

    # convert to dictionary and create SFrame
    data = tc.SFrame(data={"target":[datapoint["label"] for datapoint in datapoints], 
        "sequence":np.array([datapoint["feature"] for datapoint in datapoints])}
    )
        
    # create a classifier model  
    model = tc.classifier.create(data,target="target",verbose=0)# training
    
    # save model for use later, if desired
    model.save("../models/turi_model_dsid%d"%(dsid))

    # save this for use later 
    app.clf = model 

    return {"summary":f"{model}"}


@app.post(
    "/predict_turi/",
    response_description="Predict Label from Datapoint",
)
async def predict_datapoint_turi(datapoint: FeatureDataPoint = Body(...)):
    """
    Post a feature set and get the label back

    """

    # place inside an SFrame (that has one row)
    data = tc.SFrame(data={"sequence":np.array(datapoint.feature).reshape((1,-1))})

    if(app.clf == []):
        print("Loading Turi Model From file")
        app.clf = tc.load_model("../models/turi_model_dsid%d"%(datapoint.dsid))

        # TODO: what happens if the user asks for a model that was never trained?
        #       or if the user asks for a dsid without any data? 
        #       need a graceful failure for the client...


    pred_label = app.clf.predict(data)
    return {"prediction":str(pred_label)}


#===========================================
#   Machine Learning methods (Scikit-learn)
#-------------------------------------------
# These allow us to interact with the REST server with ML from Turi. 

@app.get(
    "/train_model_sklearn/{dsid}",
    response_description="Train a machine learning model for the given dsid",
    response_model_by_alias=False,
)
async def train_model_sklearn(dsid: int):
    """
    Train the machine learning model using Scikit-learn
    """

    # convert data over to a scalable dataframe

    datapoints = await app.collection.find({"dsid": dsid}).to_list(length=None)

    if len(datapoints) < 2:
        raise HTTPException(status_code=404, detail=f"DSID {dsid} has {len(datapoints)} datapoints.") 

    # convert to dictionary and create SFrame
    labels = [datapoint["label"] for datapoint in datapoints] 
    features = [datapoint["feature"] for datapoint in datapoints]
        
    # create a classifier model  
    model = KNeighborsClassifier(n_neighbors=1)

    model.fit(features,labels) # training
    yhat = model.predict(features)
    acc = sum(yhat==labels)/float(len(labels))

    # just write this to model files directory
    dump(model, '../models/sklearn_model_dsid%d.joblib'%(dsid))

    # save this for use later 
    app.clf = model 

    return {"summary":f"KNN classifier with accuracy {acc}"}


@app.post(
    "/predict_sklearn/",
    response_description="Predict Label from Datapoint",
)
async def predict_datapoint_sklearn(datapoint: FeatureDataPoint = Body(...)):
    """
    Post a feature set and get the label back

    """

    # place inside an SFrame (that has one row)
    data = np.array(datapoint.feature).reshape((1,-1))

    if(app.clf == []):
        print("Loading Sklearn Model From file")
        tmp = load('../models/sklearn_model_dsid%d.joblib'%(dsid)) 
        app.clf = pickle.loads(tmp['model'])

        # TODO: what happens if the user asks for a model that was never trained?
        #       or if the user asks for a dsid without any data? 
        #       need a graceful failure for the client...


    pred_label = app.clf.predict(data)
    return {"prediction":str(pred_label)}




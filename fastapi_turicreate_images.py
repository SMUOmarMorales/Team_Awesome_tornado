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
which will return the ip for various network interfaces from your card. If you get something like this:
inet 10.9.181.129 netmask 0xffffc000 broadcast 10.9.191.255 
then your app needs to connect to the netmask (the first ip), 10.9.181.129
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
from fastapi import FastAPI, Body, HTTPException, status, File, UploadFile
from fastapi.responses import Response
from pydantic import ConfigDict, BaseModel, Field, EmailStr
from pydantic.functional_validators import BeforeValidator
from bson import ObjectId, Binary
import base64

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
async def custom_lifespan(app: FastAPI):
    # Motor API allows us to directly interact with a hosted MongoDB server
    # In this example, we assume that there is a single client 
    # First let's get access to the Mongo client that allows interactions locally 

    uri = "mongodb+srv://omarcastelan:seFEZm1yn2EsKGyZ@smu8392coylef2024.l1ff5.mongodb.net/?retryWrites=true&w=majority&appName=SMU8392CoyleF2024"
    
    app.mongo_client = motor.motor_asyncio.AsyncIOMotorClient(uri)  # Update with your MongoDB URI

    # Testing connection
    try:
        app.mongo_client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)

    # connect to our databse

    db = app.mongo_client.turiDatabase
    app.collection = db.get_collection("test_images")

    app.clf = {} # Start app with dictionary, empty classifier

    yield 

    # anything after the yield can be used for clean up

    app.mongo_client.close()

# Create the FastAPI app
app = FastAPI(
    title="Machine Learning as a Service",
    summary="An application using FastAPI to add a ReST API to a MongoDB for data and labels collection.",
    lifespan=custom_lifespan,
)

# Represents an ObjectId field in the database.
# It will be represented as a `str` on the model so that it can be serialized to JSON.

# Annotated in python allows you to declare the type of a reference 
# and provide additional information related to it.
#   below we are declaring a "string" type with the annotation from BeforeValidator for a string type
#   this is the expectec setup for the pydantic Field below
# The validator is a pydantic check using the @validation decorator
# It specifies that it should be a strong before going into the validator
# we are not really using any advanced functionality, though, so its just boiler plate syntax
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

# Pydantic Model for Image Metadata
class ImageMetadata(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    filename: str = Field(..., description="Name of the image file")
    content_type: str = Field(..., description="Content type of the image (e.g., image/png)")


#===========================================
#   FastAPI methods, for interacting with db 
#-------------------------------------------
# These allow us to interact with the REST server. All interactions with mongo should be 
# async, allowing the API to remain responsive even when servicing longer queries. 

# Upload image
@app.post(
    "/upload_image/",
    response_description="Upload an image to MongoDB",
    response_model=ImageMetadata,
    status_code=status.HTTP_201_CREATED,
)
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an image to the MongoDB database.

    Stores the image as BSON binary data along with metadata.
    """
    # Read the image file as bytes
    file_bytes = await file.read()

    # Create the document with binary data and metadata
    image_document = {
        "filename": file.filename,
        "content_type": file.content_type,
        "image_data": Binary(file_bytes),  # BSON Binary for storing file bytes
    }

    # Insert the document into the collection
    result = await app.collection.insert_one(image_document)

    # Respond with the inserted metadata
    return ImageMetadata(
        id=str(result.inserted_id),
        filename=file.filename,
        content_type=file.content_type,
    )

# Endpoint to retrieve an image by ID
@app.get(
    "/images/{image_id}",
    response_description="Retrieve an image from MongoDB by its ID",
    responses={
        200: {"content": {"image/png": {}}},
        404: {"description": "Image not found"},
    },
)
async def get_image(image_id: str):
    """
    Retrieve an image from the MongoDB database by its ID.

    Returns the image as binary data with the appropriate content type.
    """
    try:
        # Find the image document by its ObjectId
        image = await app.collection.find_one({"_id": ObjectId(image_id)})

        if not image:
            raise HTTPException(status_code=404, detail="Image not found")

        # Return the image as a binary response
        return Response(
            content=image["image_data"],
            media_type=image["content_type"],
            headers={"Content-Disposition": f"inline; filename={image['filename']}"},
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid ID format: {e}")

# # Endpoint to list all stored images (metadata only)
# @app.get(
#     "/list_images/",
#     response_description="List all image metadata",
#     response_model=list[ImageMetadata],
# )
# async def list_images():
#     """
#     List all images stored in the database (metadata only).
#     """
#     images = await app.collection.find({}, {"image_data": 0}).to_list(100)
#     return [ImageMetadata(**img) for img in images]

# @app.post(
#     "/labeled_data/",
#     response_description="Add new labeled datapoint",
#     response_model=LabeledDataPoint,
#     status_code=status.HTTP_201_CREATED,
#     response_model_by_alias=False,
# )
# async def create_datapoint(datapoint: LabeledDataPoint = Body(...)):
#     """
#     Insert a new data point. Let user know the range of values inserted

#     A unique `id` will be created and provided in the response.
#     """
    
#     # insert this datapoint into the database
#     new_label = await app.collection.insert_one(
#         datapoint.model_dump(by_alias=True, exclude=["id"])
#     )

#     # send back info about the record
#     created_label = await app.collection.find_one(
#         {"_id": new_label.inserted_id}
#     )
#     # also min/max of array, rather than the entire to array to save some bandwidth
#     # the datapoint variable is a pydantic model, so we can access with properties
#     # but the output of mongo is a dictionary, so we need to subscript the entry
#     created_label["feature"] = [min(datapoint.feature), max(datapoint.feature)]

#     return created_label

# @app.get(
#     "/labeled_data/{dsid}",
#     response_description="List all labeled data in a given dsid",
#     response_model=LabeledDataPointCollection,
#     response_model_by_alias=False,
# )
# async def list_datapoints(dsid: int):
#     """
#     List all of the data for a given dsid in the database.

#     The response is unpaginated and limited to 1000 results.
#     """
#     return LabeledDataPointCollection(datapoints=await app.collection.find({"dsid": dsid}).to_list(1000))

# @app.get(
#     "/max_dsid/",
#     response_description="Get current maximum dsid in data",
#     response_model_by_alias=False,
# )
# async def show_max_dsid():
#     """
#     Get the maximum dsid currently used 
#     """

#     if (
#         datapoint := await app.collection.find_one(sort=[("dsid", -1)])
#     ) is not None:
#         return {"dsid":datapoint["dsid"]}

#     raise HTTPException(status_code=404, detail=f"No datasets currently created.")

# @app.delete("/labeled_data/{dsid}", 
#     response_description="Delete an entire dsid of datapoints.")
# async def delete_dataset(dsid: int):
#     """
#     Remove an entire dsid from the database.
#     REMOVE AN ENTIRE DSID FROM THE DATABASE, USE WITH CAUTION.
#     """

#     # replace any underscores with spaces (to help support others)

#     delete_result = await app.collection.delete_many({"dsid": dsid})

#     if delete_result.deleted_count > 0:
#         return {"num_deleted_results":delete_result.deleted_count}

#     raise HTTPException(status_code=404, detail=f"DSID {dsid} not found")

# #===========================================
# #   Machine Learning methods (Turi)
# #-------------------------------------------
# # These allow us to interact with the REST server with ML from Turi. 

# @app.get(
#     "/train_model_turi/{dsid}",
#     response_description="Train a machine learning model for the given dsid",
#     response_model_by_alias=False,
# )
# async def train_model_turi(dsid: int):
#     """
#     Train the machine learning model using Turi
#     """

#     # convert data over to a scalable dataframe

#     datapoints = await app.collection.find({"dsid": dsid}).to_list(length=None) # Call to dictionary

#     if len(datapoints) < 2:
#         raise HTTPException(status_code=404, detail=f"DSID {dsid} has {len(datapoints)} datapoints.") 

#     # convert to dictionary and create SFrame
#     data = tc.SFrame(data={"target":[datapoint["label"] for datapoint in datapoints], 
#         "sequence":np.array([datapoint["feature"] for datapoint in datapoints])}
#     )
        
#     # create a classifier model  
#     model = tc.classifier.create(data,target="target",verbose=0)# training
    
#     # save model for use later, if desired
#     model.save(f"../models/turi_model_dsid{dsid}") #Save model by DSID

#     # save this for use later 
#     app.clf[dsid] = model

#     # return {"summary":f"KNN classifier with accuracy {acc}"}

#     return {"summary":f"{model}"}

# @app.post(
#     "/predict_turi/",
#     response_description="Predict Label from Datapoint",
# )
# async def predict_datapoint_turi(datapoint: FeatureDataPoint = Body(...)):
#     """
#     Post a feature set and get the label back

#     """

#     # place inside an SFrame (that has one row)
#     data = tc.SFrame(data={"sequence":np.array(datapoint.feature).reshape((1,-1))})

#     # if(app.clf == []):
#     #     print("Loading Turi Model From file")
#     #     app.clf = tc.load_model("../models/turi_model_dsid%d"%(datapoint.dsid))

#     #     # TODO: what happens if the user asks for a model that was never trained?
#     #     #       or if the user asks for a dsid without any data? 
#     #     #       need a graceful failure for the client...
    
#     if datapoint.dsid not in app.clf:
#         try:
#             # Attempt to load the model from a saved file if it exists
#             app.clf[datapoint.dsid] = tc.load_model(f"../models/turi_model_dsid{datapoint.dsid}")
#         except FileNotFoundError:
#             # Return an error if the model is not found
#             raise HTTPException(status_code=404, detail=f"Model for DSID {datapoint.dsid} not found. Please train the model first.")


#     pred_label = app.clf[datapoint.dsid].predict(data)
#     return {"prediction":str(pred_label)}
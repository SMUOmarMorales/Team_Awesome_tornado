#!/usr/bin/python
'''
Team Awesome Lab 5 DEV Prototype
11/23/2024
'''

# For this to run properly, MongoDB should be running
# No longer using local server but running out of our Atlas Mongo instance.

# This App uses a combination of FastAPI and Motor (combining tornado/mongodb) which have documentation here:
# FastAPI:  https://fastapi.tiangolo.com 
# Motor:    https://motor.readthedocs.io/en/stable/api-tornado/index.html

# Maybe the most useful SO answer for FastAPI parallelism:
# https://stackoverflow.com/questions/71516140/fastapi-runs-api-calls-in-serial-instead-of-parallel-fashion/71517830#71517830
# Chris knows what's up 

#Imports
import os
from typing import Optional, List
from enum import Enum
import tempfile

# FastAPI imports
from fastapi import FastAPI, Body, HTTPException, status, File, UploadFile
from fastapi.responses import Response
from pydantic import ConfigDict, BaseModel, Field, EmailStr
from pydantic.functional_validators import BeforeValidator
from bson import ObjectId, Binary
import base64
from typing import List  # Import List from typing for older Python versions
from typing_extensions import Annotated

# Motor imports
from bson import ObjectId
import motor.motor_asyncio
from pymongo import ReturnDocument

# Machine Learning, Turi and Sklearn Imports
import turicreate as tc
# from sklearn.neighbors import KNeighborsClassifier

# from joblib import dump, load
# import pickle
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
    app.collection = db.get_collection("turi_train_model_dev")

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

PyObjectId = Annotated[str, BeforeValidator(str)]

#========================================
#   Data store objects from pydantic 
#----------------------------------------
# These allow us to create a schema for our database and access it easily with FastAPI
# That might seem odd for a document DB, but its not! Mongo works faster when objects
# have a similar schema. 

'''Create the data model and use strong typing. This also helps with the use of intellisense.
'''

# Pydantic Model for Image Metadata BASE MODEL
class Labeled_ImageMetadata(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    filename: str = Field(..., description="Name of the image file")
    label: str = Field(..., description="Label for predictive model")
    content_type: str = Field(..., description="Content type of the image (e.g., image/png)")
    dsid: int = Field(..., description="Dataset ID to categorize images")
    # model_config = ConfigDict(
    # populate_by_name=True,
    # arbitrary_types_allowed=True,
    # json_schema_extra={ # provide an example for FastAPI to show users
    #     "example": {
    #         "feature": [-0.6,4.1,5.0,6.0],
    #         "label": "Walking",
    #         "dsid": 2,
    #     }
    # },
    # )
    
class LabeledDataPointCollection(BaseModel):
    """
    A container holding a list of instances.

    This exists because providing a top-level array in a JSON response can be a [vulnerability](https://haacked.com/archive/2009/06/25/json-hijacking.aspx/)
    """

    datapoints: List[Labeled_ImageMetadata]
    
class FeatureDataPoint(BaseModel):
    """
    Container for a single labeled data point.
    """

    # This will be aliased to `_id` when sent to MongoDB,
    # but provided as `id` in the API requests and responses.
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    filename: str = Field(..., description="Name of the image file")
    content_type: str = Field(..., description="Content type of the image (e.g., image/png)")
    dsid: int = Field(..., description="Dataset ID to categorize images")
    # model_config = ConfigDict(
    # populate_by_name=True,
    # arbitrary_types_allowed=True,
    # json_schema_extra={ # provide an example for FastAPI to show users
    #     "example": {
    #         "feature": [-0.6,4.1,5.0,6.0],
    #         "label": "Walking",
    #         "dsid": 2,
    #     }
    # },
    # )

#===========================================
#   FastAPI methods, for interacting with db 
#-------------------------------------------
# These allow us to interact with the REST server. All interactions with mongo should be 
# async, allowing the API to remain responsive even when servicing longer queries. 

# Upload image FROM phone
@app.post("/upload_image_phone/")   # Fully derived from ChatGPT
async def upload_image(
        data: dict = Body(...,
                          example={"feature": "<base64_image_data>",
                                   "label": "some_label", "dsid": 5})
):
    """
    Accept base64 image data and metadata.
    """
    try:
        # Decode the base64 image
        image_data = base64.b64decode(data["feature"])

        # Create the document with binary data and metadata
        image_document = {
            "filename": "uploaded_image.jpg",
            "content_type": "image/jpeg",
            "image_data": Binary(image_data),
            # BSON Binary for storing file bytes
            "label": data["label"],
            "dsid": data["dsid"]
        }

        # Insert the document into the collection
        result = await app.collection.insert_one(image_document)

        return {"message": "Image uploaded successfully",
                "id": str(result.inserted_id)}
    except Exception as e:
        raise HTTPException(status_code=400,
                            detail=f"Error processing image: {e}")


# Upload image using the web API, for DEV/UAT purpose
@app.post(
    "/upload_image_dev/",
    response_description="Upload an image from the server API point",
    response_model=Labeled_ImageMetadata,
    status_code=status.HTTP_201_CREATED,
)
async def upload_image(file: UploadFile = File(...), dsid: int = 0, label = "dog"):
    """
    Upload an image to the MongoDB database with a specified dataset ID.
    """
    # Read the image file as bytes
    file_bytes = await file.read()

    # Create the document with binary data, metadata, and dataset ID
    image_document = {
        "filename": file.filename,
        "content_type": file.content_type,
        "label": label.lower(), # Ensure label is included
        "image_data": Binary(file_bytes),  # BSON Binary for storing file bytes
        "dsid": dsid,  # Dataset ID
    }

    # Insert the document into the collection
    result = await app.collection.insert_one(image_document)

    # Respond with the inserted metadata
    return Labeled_ImageMetadata(
        id=str(result.inserted_id),
        filename=file.filename,
        content_type=file.content_type,
        label = label.lower(),
        dsid=dsid,
    )

# Endpoint to retrieve an image by image ID
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

# Endpoint to list all stored images (metadata only)
@app.get(
    "/list_images/",
    response_description="List all image metadata or filter by dsid",
    response_model=List[Labeled_ImageMetadata],
)
async def list_images(dsid: Optional[int] = 0):
    """
    List all images stored in the database (metadata only) or filter by `dsid`.
    """
    query = {"dsid": dsid} if dsid is not None else {}
    images = await app.collection.find(query, {"image_data": 0}).to_list(100)

    if not images:
        raise HTTPException(status_code=404, detail="No images found")

    return [Labeled_ImageMetadata(**img) for img in images]

#===========================================
#   Machine Learning methods (Turi)
#-------------------------------------------
# These allow us to interact with the REST server with ML from Turi. 

# testing DEV MODE
@app.get(
    "/train_model_turi/{dsid}",
    response_description="Train a Turi Image Create learning model for the given dsid using the data stored there",
    response_model_by_alias=False,
)
async def train_model_turi(dsid: int):
    """
    Train the machine learning model using Turi
    """

    # convert data over to a scalable dataframe

    datapoints = await app.collection.find({"dsid": dsid}).to_list(length=None)

    if len(datapoints) < 2:
        raise HTTPException(status_code=404, detail=f"DSID {dsid} has {len(datapoints)} datapoints.",
        ) 
    
    try:     # Convert MongoDB documents to TuriCreate SFrame and store bytes as SFrame

        # Create Empty frames to store data
        images = []
        labels = []

        for datapoint in datapoints:
            try:
                        
                image_bytes = base64.b64decode(datapoint["image_data"])
                
                               # Validate image data
                try:
                    Image.open(BytesIO(image_bytes)).verify()
                except Exception as e:
                    raise ValueError(f"Invalid image data for document {datapoint['_id']}")

                # Save the bytes to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                    temp_file.write(image_bytes)
                    temp_file_path = temp_file.name
                
                print("I'm here")
                turi_image = tc.Image(temp_file_path)
                images.append(turi_image)
                labels.append(datapoint["label"])
                
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing image data for document {datapoint['_id']}: {e}",
                    )

         # Create SFrame with image and label columns
        data = tc.SFrame({"image": images, "label": labels})
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error converting MongoDB documents to SFrame: {e}",
        )
    
    try: # Try to train model
        # Train an image classifier model
        model = tc.image_classifier.create(data, target="label", verbose=True)

        # Save the trained model to disk
        model_path = f"../models/turi_image_model_dsid{dsid}"
        model.save(model_path)

        # Cache the model in memory for immediate use
        app.clf[dsid] = model
        
        # save model for use later, if desired
        model.save("../models/turi_model_dsid%d"%(dsid))

        return {
            "message": "Model trained successfully",
            "summary": str(model),
            "model_path": model_path,
            "summary":f"{model}",
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error training the image classifier model: {e}",
        )


@app.post(
    "/predict_turi/",
    response_description="Predict Label from Image",
)
async def predict_image_turi(file: UploadFile = File(...),dsid: int = 0):
    
    """
    Post an image and get the label back.
    """
    
        # Read the image file as bytes
    image_bytes = await file.read()

    try:
        
        temp_filename = f"/tmp/{file.filename}"
        with open(temp_filename, "wb") as temp_file:
            temp_file.write(image_bytes)
        data = tc.SFrame({"image": [turi_image]})
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image data: {e}")

 # Load the model if it's not already in memory
    if dsid not in app.clf:
        try:
            # Attempt to load the model from a saved file if it exists
            model_path = f"../models/turi_image_model_dsid{dsid}"
            
            if not os.path.exists(model_path): raise HTTPException(status_code=404, detail=f"Model file {model_path} does not exist. Please train the model first.")
            
            app.clf[dsid] = tc.load_model(model_path)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Model for DSID {dsid} not found. Please train the model first.")

    # Perform prediction using the model
    try:
        pred_label = app.clf[dsid].predict(data)
        return {"prediction": str(pred_label[0])}  # Return the first prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")










## Additional methods



#Count the number of images by DSID#
# @app.get(
#     "/count_images/",
#     response_description="Get the total number of images stored",
# )
# async def count_images(dsid: int):
#     """
#     Count the total number of images stored in the database, optionally filtered by `dsid`.
#     """
#     query = {"dsid": dsid} if dsid is not None else {}
#     count = await app.collection.count_documents(query)
#     return {"total_images": count}

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




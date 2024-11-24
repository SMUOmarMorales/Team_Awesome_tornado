import turicreate as tc
import numpy as np
from celery_app import celery_app
from motor.motor_asyncio import AsyncIOMotorClient

# The model logic (ML functions) has been transferred here 
# The file turicreate_fastapi_celery.py is updated to call the Celery tasks below and manage the responses

# MongoDB URI for fetching data
MONGO_URI = "mongodb+srv://omarcastelan:seFEZm1yn2EsKGyZ@smu8392coylef2024.l1ff5.mongodb.net/?retryWrites=true&w=majority&appName=SMU8392CoyleF2024"

@celery_app.task
def train_model_turi_task(dsid):
    """
    Celery task to train the model asynchronously.
    """
    # Connect to MongoDB
    client = AsyncIOMotorClient(MONGO_URI)
    db = client.turiDatabase
    collection = db.get_collection("carine_celery_test_dev")

    # Fetch data for the given DSID
    datapoints = collection.find({"dsid": dsid})
    datapoints = list(datapoints)  # Convert cursor to list

    if len(datapoints) < 2:
        return {"error": f"DSID {dsid} has insufficient datapoints."}

    # Prepare data for Turi Create
    data = tc.SFrame({
        "target": [datapoint["label"] for datapoint in datapoints],
        "sequence": np.array([datapoint["feature"] for datapoint in datapoints])
    })

    # Train the model
    model = tc.classifier.create(data, target="target", verbose=0)

    # Save the model for later use
    model.save(f"../models/turi_model_dsid{dsid}")

    # Cleanup and close MongoDB connection
    client.close()

    return {"summary": f"Model trained for DSID {dsid}", "accuracy": model.evaluate(data)["accuracy"]}


@celery_app.task
def predict_datapoint_turi_task(dsid, feature):
    """
    Celery task to predict labels asynchronously.
    """
    # Load the saved model
    try:
        model = tc.load_model(f"../models/turi_model_dsid{dsid}")
    except FileNotFoundError:
        return {"error": f"Model for DSID {dsid} not found. Please train it first."}

    # Prepare feature for prediction
    data = tc.SFrame(data={"sequence": np.array(feature).reshape((1, -1))})

    # Predict the label
    pred_label = model.predict(data)
    return {"prediction": str(pred_label)}

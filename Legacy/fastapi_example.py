#!/usr/bin/python
'''
In this example, we will use FastAPI as a gateway into a MongoDB database. We will use a REST style 
interface that allows users to initiate GET, POST, PUT, and DELETE requests.

Specifically, we are creating an app that tracks entering marvel characters from a user.
Then the user can ask for a list of various characters by their name.

The swift code for interacting with the interface is also available through the SMU MSLC class repository. 
Look for the https://github.com/SMU-MSLC/SwiftHTTPExample with branches marked for FastAPI

To run this example in localhost mode only use the command:
fastapi dev fastapi_example.py

Otherwise, to run the app in deployment mode (allowing for external connections), use:
fastapi run fastapi_example.py

External connections will use your public facing IP, which you can find from the inet. 
A useful command to find the right public facing ip is:
ifconfig |grep "inet "
which will return the ip for various network interfaces from your card. 
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

# Moreover, the following code was manipulated from a Mongo+FastAPI tutorial available here:
# There are quite a few changes from that code, so it will be quite different. 
# Motor allows us to asynchronously interact with the database, allowing FastAPI to service
# requests while mongo commands are performed. 
#   https://github.com/mongodb-developer/mongodb-with-fastapi/tree/master


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


# Create the FastAPI app
app = FastAPI(
    title="Marvel Character API",
    summary="A sample application showing how to use FastAPI to add a ReST API to a MongoDB collection.",
)

# Motor API allows us to directly interact with a hosted MongoDB server
# In this example, we assume that there is a single client 
# First let's get access to the Mongo client that allows interactions locally 
client = motor.motor_asyncio.AsyncIOMotorClient()

# new we need to create a database and a collection. These will create the db and the 
# collection if they haven't been created yet. They are stored upon the first insert. 
db = client.mcu
character_collection = db.get_collection("character")

# Represents an ObjectId field in the database.
# It will be represented as a `str` on the model so that it can be serialized to JSON.
PyObjectId = Annotated[str, BeforeValidator(str)]

# for storing the kind of character they are
class KindEnum(str, Enum):
    hero = 'hero'
    villain = 'villain'
    grey = 'gray'


# create a quick hello world for connecting
@app.get("/")
def read_root():
    return {"Hello:": "Mobile Sensing World"}


#========================================
#   Data store objects from pydantic 
#----------------------------------------
# These allow us to create a schema for our database and access it easily with FastAPI
# That might seem odd for a document DB, but its not! Mongo works faster when objects
# have a similar schema. 

'''Create the MCU character model and use strong typing. This alos helps with the use of intellisense.
'''
class CharacterModel(BaseModel):
    """
    Container for a single marvel character record.
    """

    # The primary key for the CharacterModel, stored as a `str` on the instance.
    # This will be aliased to `_id` when sent to MongoDB,
    # but provided as `id` in the API requests and responses.
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    name: str = Field(...) # our character's name, no restrictions
    power: str = Field(...) # a super power, if they have one
    level: int = Field(..., le=5) # class of character (1 to 5 limit)
    kind: str = Field(...) # Enum for good, bad, mixed characters 
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_schema_extra={ # provide an example for FastAPI to show users
            "example": {
                "name": "Natash Romanov",
                "power": "None",
                "level": 1,
                "kind": 'hero',
            }
        },
    )


class UpdateCharacterModel(BaseModel):
    """
    A set of optional updates to be made to a document in the database.
    """

    name: Optional[str] = None
    power: Optional[str] = None
    level: Optional[int] = None
    kind: Optional[str] = None
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str},
        json_schema_extra={
            "example": {
                "name": "Natash Romanov",
                "power": "None",
                "level": 1,
                "kind": 'hero',
            }
        },
    )


class CharacterCollection(BaseModel):
    """
    A container holding a list of `CharacterModel` instances.

    This exists because providing a top-level array in a JSON response can be a [vulnerability](https://haacked.com/archive/2009/06/25/json-hijacking.aspx/)
    """

    characters: List[CharacterModel]


#========================================
#   FastAPI methods, for interacting with db 
#----------------------------------------
# These allow us to interact with the REST server. All interactions with mongo should be 
# async, allowing the API to remain responsive even when servicing longer queries. 


@app.post(
    "/characters/",
    response_description="Add new character",
    response_model=CharacterModel,
    status_code=status.HTTP_201_CREATED,
    response_model_by_alias=False,
)
async def create_character(character: CharacterModel = Body(...)):
    """
    Insert a new character record. 
    Update the character if a character by that name already exists.
    Return the newly created character to the connected client

    A unique `id` will be created and provided in the response.
    """

    new_character = await character_collection.find_one_and_update(
        {"name":character.name}, 
        {"$set":character.model_dump(by_alias=True, exclude=["id"])}, # set all fields except id
        upsert=True, # insert if nothing found.
        return_document=ReturnDocument.AFTER)

    return new_character


@app.get(
    "/characters/",
    response_description="List all characters",
    response_model=CharacterCollection,
    response_model_by_alias=False,
)
async def list_characters():
    """
    List all of the characters data in the database.

    The response is unpaginated and limited to 1000 results.
    """
    return CharacterCollection(characters=await character_collection.find().to_list(1000))


@app.get(
    "/characters/{name}",
    response_description="Get a single mcu character",
    response_model=CharacterModel,
    response_model_by_alias=False,
)
async def show_character(name: str):
    """
    Get the record for a specific character, looked up by `name`.
    Any spaces in the name should be replaced by underscores.
    """

    # replace any underscores with spaces
    name_query = name.replace("_"," ")
    if (
        character := await character_collection.find_one({"name": name_query})
    ) is not None:
        return character

    raise HTTPException(status_code=404, detail=f"Character {name_query} not found")


@app.put(
    "/characters/{name}",
    response_description="Update a character",
    response_model=CharacterModel,
    response_model_by_alias=False,
)
async def update_character(name: str, character: UpdateCharacterModel = Body(...)):
    """
    Update individual fields of an existing character record.

    Only the provided fields will be updated.
    Any missing or `null` fields will be ignored.
    """
    
    # replace any underscores with spaces
    name_query = name.replace("_"," ")

    # only get fields that exist (from the put request)
    character = {
        k: v for k, v in character.model_dump(by_alias=True).items() if v is not None
    }

    # if we still have some fields to update
    if len(character) >= 1:
        # this is similar to the insert, but the user speifically
        # wants to update a character. We will query, but NOT insert if not found. 
        update_result = await character_collection.find_one_and_update(
            {"name": name_query},
            {"$set": character},
            return_document=ReturnDocument.AFTER,
        )
        if update_result is not None:
            return update_result
        else:
            # return a common sense error to them 
            raise HTTPException(status_code=404, detail=f"Character {name_query} not found")

    # The update is empty, but we should still return the matching document:
    if (existing_character := await character_collection.find_one({"name": name_query})) is not None:
        return existing_character

    raise HTTPException(status_code=404, detail=f"Character {name_query} not found")


# @app.delete("/characters/{name}", response_description="Delete a character")
# async def delete_character(name: str):
#     """
#     Remove a single character record from the database.
#     """

#     # replace any underscores with spaces (to help support others)
#     name_query = name.replace("_"," ")

#     delete_result = await character_collection.delete_one({"name": name_query})

#     if delete_result.deleted_count == 1:
#         return Response(status_code=status.HTTP_204_NO_CONTENT)

#     raise HTTPException(status_code=404, detail=f"Character {name_query} not found")

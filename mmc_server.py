import os
<<<<<<< HEAD
from typing import Dict, List
=======
import sys
import warnings
from typing import Optional, Dict, List
>>>>>>> master
import torch

from fastapi import FastAPI
from pydantic import BaseModel

from tomcat_speech.models.multimodal_models import MultitaskModel # fixme
<<<<<<< HEAD
from tomcat_speech.data_prep.data_prep_helpers import (
=======
sys.path.append("../multimodal_data_preprocessing")
from utils.data_prep_helpers import (
>>>>>>> master
    make_glove_dict,
    Glove,
)
from tomcat_speech.training_and_evaluation_functions.asist_analysis_functions import predict_with_model

# the current parameters file is saved as testing_parameters/config.py
import tomcat_speech.parameters.testing_parameters.config as params

# Get model and glove paths
MODEL_PATH = os.path.dirname(__file__) + "/data/EmoPers_trained_model.pt"
GLOVE_PATH = os.path.dirname(__file__) + "/data/glove.subset.300d.txt"

# Set device, checking CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# IMPORT GLOVE + MAKE GLOVE OBJECT
glove_dict = make_glove_dict(GLOVE_PATH)
glove = Glove(glove_dict)
pretrained_embeddings = glove.data
num_embeddings = pretrained_embeddings.size()[0]

# Create predictor object
PREDICTOR = MultitaskModel(
        params=params.model_params,
        num_embeddings=num_embeddings,
        pretrained_embeddings=pretrained_embeddings,
        multi_dataset=False,
    )

# get saved parameters
PREDICTOR.load_state_dict(torch.load(MODEL_PATH))
PREDICTOR.to(device)

<<<<<<< HEAD
=======
# class Emotions(BaseModel):


>>>>>>> master
class DialogAgentMessage(BaseModel):
    """Data model for incoming message from UAZ Dialog Agent"""

    data: Dict
    msg: Dict
    # todo, add the relevant things and rename


# Define data model for incoming message
class ClassificationMessage(BaseModel):
    """Data model for outgoing message from UAZ Dialog Agent"""
    speaker: str
    emotions: Dict
    traits: Dict
    penultimate_emotions: List[float]
    penultimate_traits: List[float]


# Create the FastAPI instance
app = FastAPI(title="UAZ Multimodal Participant State Model", version="0.0.2")


def build_result_message(data):
    result_message = {
        'speaker': data['speaker'],
        'emotions': data['emotions'],
        'traits': data['traits'],
        'penultimate_emotions': data['penultimate_layer_emotions'],
        'penultimate_traits': data['penultimate_layer_traits'],
    }
    return result_message


@app.get("/encode", response_model=Dict)
def classify_utterance(message: DialogAgentMessage):
    # provide format
    data = {'data': message.data, 'msg': message.msg}
    # Make the prediction.
    results = predict_with_model([data], PREDICTOR, glove, device, params)
    # Since we are sending a single utterance, there is only one element in the results list
    data = results[0]['data']
    # Convert the predictions output to the json format needed for fastapi
    result_message = build_result_message(data)

    return result_message

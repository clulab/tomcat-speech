import os
from typing import Optional, Dict, List
import torch

# Mmmm.... smells like code smell! Feel free to make this work the right way, I tried.
import sys
sys.path.insert(0, "../")

from fastapi import FastAPI
from pydantic import BaseModel

from tomcat_speech.models.input_models import MultitaskModel # fixme
from tomcat_speech.data_prep.data_prep_helpers import (
    make_glove_dict,
    Glove,
)
from tomcat_speech.train_and_test_models.asist_analysis_functions import predict_with_model

# the current parameters file is saved as testing_parameters/config.py
import tomcat_speech.train_and_test_models.testing_parameters.config as params

# Get model and glove paths
MODEL_PATH = os.path.dirname(__file__) + "/data/MC_GOLD_classwts_nogender_25to75perc_avg_IS13.pth"
GLOVE_PATH = os.path.dirname(__file__) + "/data/glove.short.300d.punct.txt"

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

# class Emotions(BaseModel):


# todo: email Adarsh about not using this, just using raw json with
# get_json_output_of_speech_analysis

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
app = FastAPI(title="UAZ Multimodal Participant State Model", version="0.0.1")
# todo: are we outputing emotions? or other things?


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

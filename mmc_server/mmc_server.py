import os
from typing import Optional, Dict, List
from dataclasses import dataclass

from fastapi import FastAPI
from pydantic import BaseModel

from tomcat_speech.models.bimodal_models import UttLevelBimodalCNN # fixme

# Get model path
MODEL_PATH = os.path.dirname(__file__) + "/data/baseline_model_speaker.pt" # fixme

# Create predictor object
PREDICTOR = UttLevelBimodalCNN(model_path=MODEL_PATH, history_len=7) # fixme, real params


class DialogAgentMessage(BaseModel):
    """Data model for incoming message from UAZ Dialog Agent"""

    participant_id: str
    text: str
    extractions: List[Dict]
    # todo, add the relevant things and rename


# Define data model for incoming message
class ClassificationMessage(BaseModel):
    """Data model for outgoing message from TAMU Dialog Act Classifier"""

    classification: str
    # todo: update with what we are outputting


# Create the FastAPI instance
app = FastAPI(title="UAZ Multimodal Participant State Model", version="0.0.1")
# todo: are we outputing emotions? or other things?


@app.get("/reset-model")
def reset_model():
    # The model should reset before and after each mission.
    PREDICTOR.reset_model()

# todo: what should we name this?
@app.get("/encode", response_model=str)
def classify_utterance(message: DialogAgentMessage):
    # todo: update with what is needed by the model
    # todo: update with the right method (forward? another?)
    results = PREDICTOR.predict(
        f"{message.participant_id}:{message.text}"
    )
    return results

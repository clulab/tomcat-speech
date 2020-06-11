# asist-speech

This repository contains python code to prepare multimodal (currently audio,text,speaker) data for input into models and to instantiate and run these models.

All code was created for the following system requirements:

Python = 3.7
torch = 1.4.0
pandas = 1.0.1
numpy = 1.18.1
sklearn = 0.22.2


data_prep contains:
- audio_extraction.py : code to extract necessary features from audio + corresponding transcriptions
- data_prep.py : classes to prepare data for input into the models; mostly used for health outcomes data currently
- meld_input_formatting.py : Formats MELD dataset for input into models
- subset_glove.py : code to create a subset of GloVe for faster loading at test time


models contains:
- parameters/ : directory of parameters files used by the models
- baselines.py : model classes to be used for baselines
- bimodal_models.py : model classes to be used with bimodal data
- input_models.py : model classes used for preparing input representations (not yet implemented)
- plot_training.py : initial plotting of training and dev loss/accuracy curves
- train_and_test_models.py : code for training + evaluation of models


models_test contains:
- glove_test.py : test usage of subset_glove.py
- input_manipulation_test.py : test usage of audio_extraction.py
- meld_test.py : test usage of running a model with MELD
- model_test.py : the main file for running models with health outcomes data

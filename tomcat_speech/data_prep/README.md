# multimodal-data-preprocessing
Data preprocessing code for multiple multimodal datasets

This code currently includes preprocessing for text and audio modalities. 
Text may be encoded as GloVe, RoBERTa, DistilBERT, or BERT embeddings. 
Text may also be preserved to be fed directly into huggingface models. 

## Included datasets

* MELD (https://github.com/declare-lab/MELD)
* MUStARD (https://github.com/soujanyaporia/MUStARD)
* First Impressions V2 (http://chalearnlap.cvc.uab.es/dataset/24/description/)
* RAVDESS (https://zenodo.org/record/1188976)
* Columbia Deception Corpus (https://catalog.ldc.upenn.edu/LDC2013S09)
* CMU MOSI (http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/)

## Preprocessing datasets
To make use of the data preprocessing code, download the raw data for each dataset of interest. 
Use `preprocessing_scripts/extract_acoustic_features.py` to extract acoustic features and organize raw data, 
then run `preprocessing_scripts/save_partitioned_data.py` to save partitioned data as pickle files.
Alter the associated config files rather than these scripts themselves.

Saved data may be used as input into custom-made machine learning models in PyTorch (https://pytorch.org/) or models this repo.

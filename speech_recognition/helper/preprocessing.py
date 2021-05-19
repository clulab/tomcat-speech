import os
import re

import soundfile as sf

from collections import defaultdict

from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from sklearn.model_selection import train_test_split

import json

def transcript_to_dict(tsv_file, header):
    '''
    Create a dictionary that maps filename and transcription (transcription will be massaged here)
    params
    @tsv_file: tsv file that has two columns (1) file name, and (2) transcription
    return: dict with {filename: transcription}
    TODO: How to deal with numbers? 
    '''

    char_to_ignore_regex = '[\,\?\.\!\-\;\:\"\+\/\*\%\(\)\[\]’@_%&$#=]'

    transcription_dict = {}
    starting_index = 0

    if header is True:
        starting_index = 1

    with open(tsv_file, "r") as f:
        data = f.readlines()

    for i in range(starting_index, len(data)):
        # print(data[i])
        try: 
            filename, transcription = data[i].rstrip().split("\t")
            transcription = re.sub(char_to_ignore_regex, '', transcription).upper()
            transcription = transcription.replace("…", "")
            transcription = transcription.replace("%", "")
            transcription_dict[filename] = transcription
        except:
            next

    return transcription_dict

def speech_to_dict(filepath):
    '''
    Create the dictionary that maps filename and full file path
    params
    @filepath: path that contains wav files
    @tsv_file: tsv file that has two columns (1) file name, and (2) transcription
    return: None. This function generates a csv file with two columns (1) full path to audio files, and (2) transcription
    '''

    # Grep wav files in a folder
    filenames = [x for x in os.listdir(filepath) if x.endswith(".wav")]

    # Create dictionary with {filename: full file path}
    speech_dict = {}
    for filename in filenames:
        filename_without_wav = filename.replace(".wav", "")
        file_fullpath = os.path.join(filepath, filename)
        speech_dict[filename_without_wav] = file_fullpath

    return speech_dict

def check_number(sentence):
    return bool(re.search(r'\d', sentence))

def generate_dict(filepath, tsv_file, output_name, header=False):
    '''
    Generate dictionary and save it to csv file
    params
    @filepath: path that contains wav files
    return: dict with {filename: {speech_data: Array, sampling_rate: Int}
    '''
    speech_dict = speech_to_dict(filepath)
    transcript_dict = transcript_to_dict(tsv_file, header=header)
    filenames = list(transcript_dict.keys())

    with open(output_name, "w") as out:
        header = "file,text\n"
        out.write(header)
        for filename in filenames:
            if filename in transcript_dict:
                if not check_number(transcript_dict[filename]):
                    item = "%s,%s\n" % (speech_dict[filename], transcript_dict[filename])
                    out.write(item)
                else:
                    next



# def creat_feature_arrays(speech_dict, sr, transcript_dict):
#     tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-large-960hv")
#     feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=sr, padding_value=0.0, do_normalize=True, return_attention_mask=True)
#     processor = Wav2Vect2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

#     wav_names = list(transcript_dict.keys())

#     idx = []
#     speech_data = []
#     texts = []

#     for wav_name in wav_names:
#         idx.append(wav_name)
#         speech = speech_dict[wav_name]['data']
#         sampling_rate = speech_dict[wav_name]['sampling_rate']
#         transcription = transcript_dict[wav_name]

#         speech_value = processor(speech, sampling_rate=sampling_rate).input_values
#         text_tokens = processor(transcription).input_ids

        



#  class SpeechToDict(speech_dict, transcript_dict)):
#     def __init__(self, speech_dict, transcript_dict):
#         tokenizer = Wav2Vec2CTCTokenizer()
#         feature_extractor = Wav2Vec2FeatureExtractor()
#         processor = Wav2Vect2Processor(feature_extractor = feature_extractor, tokenizer = tokenizer)

#         self.speech_data = speech_dict
#         self.transcript_data = transcript_dict
#         self.wav_names = list(transcript_dict.keys())

#     def __len__(self):
#         return len(self.wav_names)

#     def __getitem__(self, idx): 
#         try:
#             wav_name = self.wav_names[idx]
#             speech = self.speech_data[wav_name]["data"]
#             sampling_rate = self.speech_data[wav_name]["sampling_rate"]
#             transcription = self.transcript_data[wav_name]

#             item = {
#                 'id': wav_name,
#                 'speech': speech, 
#                 'speech_value': processor(speech, sampling_rate=sampling_rate).input_values
#                 'sr': sampling_rate,
#                 'text': transcription, 
#                 'text_tokens': processor(transcription).input_ids
#             }

#             return item
        
#         except KeyError:
#             next

# class DataPrep:
#     '''
#     A class to prepare train/test data
#     '''

#     def __init__(
#         self, 
#         wav_path, 
#         tsv_file, 

#     )

#     self.speech_dict = speech_to_dict(wav_path)
#     self.transcript_dict = transcript_to_dict(tsv_file, header=False)
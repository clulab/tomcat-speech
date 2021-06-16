import os
import re

from collections import defaultdict

from sklearn.model_selection import train_test_split

import json

def transcript_to_dict(tsv_file, header):
    '''
    Create a dictionary that maps filename and transcription (transcription will be massaged here)
    params
    @tsv_file: tsv file that has two columns (1) file name, and (2) transcription
    return: dict with {filename: transcription}
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
    return: dictionary {file_id: file_fullpath}
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
    '''
    ASR system shows an error when there is a digit in the thranscription. 
    This function is used to check the existence of digits in the transcription. 
    '''
    return bool(re.search(r'\d', sentence))

def generate_dict(filepath, tsv_file, output_name, header=False):
    '''
    Generate dictionary and save it to csv file
    params
    @filepath: path that contains wav files
    @tsv_file: path to tsv file contains file id and transcription (e.g., "1_123  this is transcription")
    @output_name: filename of the output csv
    return: dict with {filename: {speech_data: Array, sampling_rate: Int}
    '''

    # Load dictionaries
    speech_dict = speech_to_dict(filepath)
    transcript_dict = transcript_to_dict(tsv_file, header=header)
    filenames = list(transcript_dict.keys())

    # Create csv file
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

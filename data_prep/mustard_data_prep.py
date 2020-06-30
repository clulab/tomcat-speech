# prepare the data from MUStARD dataset

import os
import json
from data_prep.audio_extraction import convert_mp4_to_wav


def organize_labels_from_json(jsonfile, savepath, save_name):
    """
    Take the jsonfile containing the text,
    """
    with open(jsonfile, 'r') as jfile:
        json_data = json.load(jfile)

    # create holder for relevant information
    data_holder = [["clip_id", "utterance", "speaker", "sarcasm"]]

    for clip in json_data.keys():
        clip_id = clip
        utt = json_data[clip]["utterance"]
        spk = json_data[clip]["speaker"]
        sarc = str(1 if json_data[clip]["sarcasm"] else 0)

        data_holder.append([clip_id, utt, spk, sarc])

    with open(os.path.join(savepath, save_name), 'w') as savefile:
        for item in data_holder:
            savefile.write("\t".join(item))
            savefile.write("\n")


######USAGE
# organize_labels_from_json("/Users/jculnan/Downloads/MUStARD/sarcasm_data.json",
#                           "/Users/jculnan/Downloads/MUStARD", "mustard_utts.tsv")

# path_to_files = "/Users/jculnan/Downloads/MUStARD/utterances_final"
#
# for clip in os.listdir(path_to_files):
#     convert_mp4_to_wav(os.path.join(path_to_files, clip))

# test audio data extraction and formatting code from audio_extraction.py

from tomcat_speech import data_prep as extraction

import pandas as pd
import os

# set paths
# path to transcription files
trs_path = "/Volumes/LIvES/323_files/audio"
# trs_path = "/Volumes/LIvES/multimodal_data/failed_items"
# path to audio files
audio_path = trs_path
# path to save output
save_path = "/Volumes/LIvES/multimodal_data_updated"
# set of OpenSMILE features to extract
feature_set = "IS12"

failed_list = []
for item in os.listdir(trs_path):
    if item.endswith(".trs"):
        try:
            # extract the information from transcription files
            item_name = item.split(".")[0]
            print(item_name)

            # uncomment the following two lines first time using
            # trans_convert = extraction.TRSToCSV(trs_path, item_name)
            # trans_convert.convert_trs(save_path) # convert and save trs file

            # extract audio features with OpenSMILE
            acoustic_savename = "{0}_{1}".format(item_name, feature_set)
            audio_extract = extraction.ExtractAudio(
                audio_path, "{0}.wav".format(item_name), save_path
            )
            audio_extract.save_acoustic_csv(
                feature_set, "{0}.csv".format(acoustic_savename)
            )

            # load csv of features + csv of transcription information
            audio_df = extraction.load_feature_csv(
                "{0}/{1}".format(save_path, "{0}.csv".format(acoustic_savename))
            )

            # uncomment the first time using
            # extraction.expand_words("{0}/{1}.tsv".format(save_path, item_name),
            #                         "{0}/{1}-expanded.csv".format(save_path, item_name))

            # read in the saved csv
            expanded_wds_df = pd.read_csv(
                "{0}/{1}-expanded.csv".format(save_path, item_name), sep="\t"
            )

            # combine the files
            combined = pd.merge(audio_df, expanded_wds_df, on="frameTime")

            # average across words and save as new csv
            wd_avgd = extraction.avg_feats_across_words(combined)
            wd_avgd.to_csv("{0}/{1}_avgd.csv".format(save_path, acoustic_savename))

        except:
            failed_list.append(item)

with open("/Volumes/LIvES/multimodal_data_updated/failed_items.txt", "w") as tfile:
    for item in failed_list:
        tfile.write(item)
        tfile.write("\n")

# prepare the asist-produced audio and transcription data for neural classifiers
import sys

import data_prep.audio_extraction as audio_extraction

import os
import pandas as pd
import pprint
import ast


class JSONtoTSV:
    """
    Takes an AWS-generated transcript and gets its words and timestamps
    lines of tsv output are timestart, timeend, word, utt_num, word_num where:
        timestart = start of word
        timeend = time of word end
        word = the word predicted
        utt_num = utterance number within the file
        word_num = word number within the file
    NOTE: Transcripts currently have bad json format, so use AST to fix
    """
    def __init__(self, path, jsonfile, use_txt=True):
        self.path = path
        self.jname = jsonfile
        if use_txt:
            self.jfile = "{0}/{1}.txt".format(path, jsonfile)
        else:
            self.jfile = "{0}/{1}.json".format(path, jsonfile)

    def convert_json(self, savepath):
        jarray = [['speaker', 'timestart', 'timeend', 'word', 'utt_num', 'word_num']]

        # read in the json
        with open(self.jfile, 'r') as djfile:
            jf = djfile.read()
            fixed = ast.literal_eval(jf)

        # get only the words
        all_words = fixed['results']['items']

        # set utterance and word counters
        utt = 0
        wd_num = 0

        for item in all_words:

            if item['type'] == "punctuation":
                utt += 1

            elif item['type'] == "pronunciation":
                speaker = 1  # todo: this assumes only 1 speaker per file for now; change if needed
                # get start and end times
                timestart = item['start_time']
                timeend = item['end_time']

                # get predicted word
                word = item['alternatives'][0]['content'].lower()  # todo: will there always be only 1 alternative?
                word_num = wd_num

                # get the utterance and word number
                utt_num = utt

                # update wd counter
                wd_num += 1

                jarray.append([speaker, timestart, timeend, word, utt_num, word_num])

        with open("{0}/{1}.tsv".format(savepath, self.jname), 'w') as tsvfile:
            for item in jarray:
                tsvfile.write("\t".join(item))
                tsvfile.write("\n")


class ASISTInput:
    def __init__(self, asist_path, save_path, smilepath="~/opensmile-2.3.0",
                 acoustic_feature_set="IS10", missions=None):
        self.path = asist_path
        self.save_path = save_path
        self.smilepath = smilepath  # the path to your openSMILE installation

        # set of acoustic features to extract
        # options are "IS09", "IS10", "IS11", "IS12", "IS13"
        self.acoustic_feature_set = acoustic_feature_set

        if missions:
            self.missions = missions  # list of names of missions of interest (e.g.['mission_2'])
        else:
            self.missions = (['mission_2'])

    def extract_audio_and_text_data(self):
        """
        get the audio data; feed it through processes in audio_extraction.py
        :param missions : a list of names of the mission(s) whose data is of interest
        """
        for item in os.listdir(self.path):
            item_path = "{0}/{1}".format(self.path, item)
            if os.path.isdir(item_path):
                print(item)
                for mission in self.missions:
                    if "{0}_transcript_full.txt".format(mission) in os.listdir(item_path):
                        print(mission)
                        # print(self.save_path)
                        participant_id = item

                        # read transcript and extract words and times to a clean csv file
                        transcript_convert = JSONtoTSV(item_path, "{0}_transcript_full".format(mission),
                                                       use_txt=True)
                        transcript_convert.convert_json(self.save_path)

                        # open corresponding audio and send through extraction; return csv file
                        # ID audio file
                        audio_path = "{0}/{1}".format(item_path, mission)

                        # set the name for saving csvs
                        acoustic_savename = "{0}_{1}".format(participant_id, mission)

                        # extract audio features and save csv
                        audio_extract = audio_extraction.ExtractAudio(audio_path, "player_audio.wav", self.save_path,
                                                                      self.smilepath)
                        audio_extract.save_acoustic_csv(self.acoustic_feature_set, "{0}.csv".format(acoustic_savename))

                        # load csv of features + csv of transcription information
                        audio_df = audio_extraction.load_feature_csv(
                            "{0}/{1}".format(self.save_path, "{0}.csv".format(acoustic_savename)))

                        # TODO: CAN ONLY COMPLETE THIS ONCE THE TEXT IS ALTERED
                        audio_extraction.expand_words("{0}/{1}".format(self.save_path, "{0}.tsv".format(acoustic_savename)),
                                                      "{0}/{1}-expanded.csv".format(self.save_path,
                                                                                    acoustic_savename))

                        # read in the saved csv
                        expanded_wds_df = pd.read_csv("{0}/{1}-expanded.csv".format(self.save_path,
                                                                                    acoustic_savename))

                        # combine the files
                        combined = pd.merge(audio_df, expanded_wds_df, on='frameTime')

                        # average across words and save as new csv
                        wd_avgd = audio_extraction.avg_feats_across_words(combined)
                        wd_avgd.to_csv("{0}/{1}_avgd.csv".format(self.save_path, acoustic_savename))

                        print("MADE IT THROUGH ONCE!")


if __name__ == "__main__":
    # # define variables
    data_path = "../../Downloads/real_search_data"
    asist_path = "../../"
    save_path = "output/asist_audio"
    missions = ['mission_1', 'mission_2', 'mission_0']
    acoustic_feature_set = "IS10"
    smile_path = "~/opensmile-2.3.0"

    # try out the audio
    # print(missions)
    asist = ASISTInput(data_path, save_path, smile_path, missions=missions)
    asist.extract_audio_and_text_data()

    # # path to json
    # # jpath = "../../Downloads/real_search_data/2eef2943-32c4-45fb-ba0a-d64170510a1f"
    # jpath = "../../Downloads/real_search_data/139e6d55-ce96-4605-8c77-6847e0392d19"
    # # name of json file
    # jname = "mission_2_transcript_full"
    #
    # v = JSONtoCSV(jpath, jname)
    # v.convert_json(save_path)
    # print
    "DONE!"

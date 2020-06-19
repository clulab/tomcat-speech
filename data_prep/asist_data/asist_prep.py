# prepare the asist-produced audio and transcription data for neural classifiers

import data_prep.audio_extraction as audio_extraction

import os
import sys
import pandas as pd
import pprint
import ast
import random


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
    def __init__(self, path, jsonfile, save_name, use_txt=True):
        self.path = path
        self.jname = jsonfile
        self.save_name = save_name

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

                jarray.append([str(speaker), timestart, timeend, word, str(utt_num), str(word_num)])

        with open("{0}/{1}.tsv".format(savepath, self.save_name), 'w') as tsvfile:
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

    def extract_asist_audio_data(self, create_gold_labels=True, nested=False):
        """
        Get the audio data; convert it to .wav and feed it through the processes in audio_extraction.py
        todo: currently only deals with flat structure
        """
        # create fake gold labels to see how this works in the system
        if create_gold_labels:
            gold_labels = [["sid", "overall"]]
            all_participants = []
        if not nested:
            for item in os.listdir(self.path):
                if item.endswith("_video.mp4"):
                    # get the participant number
                    experiment_id = item.split("_")[4]
                    participant_id = item.split("_")[7]
                    itempath = self.path + "/" + item
                    print(self.path)
                    print(item)
                    print(itempath)

                    # create gold label, participant id pair
                    # todo: remove this once we have gold labels
                    if create_gold_labels:
                        gold = random.randint(0, 1)
                        if participant_id not in all_participants:
                            gold_labels.append([item, str(gold)])

                        # add participant to list of participants so it doesn't get repeated
                        all_participants.append(participant_id)

                    # convert mp4 files to wav
                    audio_path = convert_mp4_to_wav(itempath)
                    print(audio_path)
                    audio_name = audio_path.split("/")[-1] # because we don't want the full path
                    print(audio_name)

                    # set the name for saving csvs
                    acoustic_savename = "{0}_{1}".format(experiment_id, participant_id)

                    print("Extracting openSMILE features...")

                    # extract audio features and save csv
                    audio_extract = audio_extraction.ExtractAudio(self.path, audio_name, self.save_path,
                                                                  self.smilepath)
                    audio_extract.save_acoustic_csv(self.acoustic_feature_set,
                                                    "{0}_feats.csv".format(acoustic_savename))

        if create_gold_labels:
            ys_path = "{0}/asist_ys".format(self.save_path)
            os.system('if [ ! -d "{0}" ]; then mkdir -p {0}; fi'.format(ys_path))
            with open("{0}/all_ys.csv".format(ys_path), 'w') as goldfile:
                for item in gold_labels:
                    goldfile.write(",".join(item))
                    goldfile.write("\n")

    def extract_asist_text_data(self, nested=False):
        """
        Convert Zoom transcriptions into usable csv transcription files
        """


    def extract_tomcat_audio_and_text_data(self):
        """
        get the audio data; feed it through processes in audio_extraction.py
        :param missions : a list of names of the mission(s) whose data is of interest
        fixme: this will contain a gold-label creation mechanism; remove it once asist has real gold labels
        """
        gold_labels = [["sid", "overall"]]
        all_participants = []

        for item in os.listdir(self.path):
            item_path = "{0}/{1}".format(self.path, item)
            if os.path.isdir(item_path):
                print(item)
                for mission in self.missions:
                    if "{0}_transcript_full.txt".format(mission) in os.listdir(item_path) and \
                            check_transcript("{0}/{1}_transcript_full.txt".format(item_path, mission)):
                        print(mission)
                        # print(self.save_path)
                        participant_id = item

                        # create gold label, participant id pair
                        # todo: remove this once we have gold labels
                        gold = random.randint(0, 1)
                        if participant_id not in all_participants:
                            gold_labels.append([item, str(gold)])

                        # add participant to list of participants so it doesn't get repeated
                        all_participants.append(participant_id)

                        # set the name for saving csvs
                        acoustic_savename = "{0}_{1}".format(participant_id, mission)

                        # open corresponding audio and send through extraction; return csv file
                        # ID audio file
                        audio_path = "{0}/{1}".format(item_path, mission)

                        print("Extracting openSMILE features...")

                        # extract audio features and save csv
                        audio_extract = audio_extraction.ExtractAudio(audio_path, "player_audio.wav", self.save_path,
                                                                      self.smilepath)
                        audio_extract.save_acoustic_csv(self.acoustic_feature_set,
                                                        "{0}_feats.csv".format(acoustic_savename))

                        # load csv of features + csv of transcription information
                        audio_df = audio_extraction.load_feature_csv(
                            "{0}/{1}".format(self.save_path, "{0}_feats.csv".format(acoustic_savename)))

                        print("Extracting words from transcripts...")

                        # read transcript and extract words and times to a clean csv file
                        transcript_convert = JSONtoTSV(item_path, "{0}_transcript_full".format(mission),
                                                       save_name=acoustic_savename, use_txt=True)
                        transcript_convert.convert_json(self.save_path)

                        print("Aligning audio and text data...")

                        audio_extraction.expand_words("{0}/{1}".format(self.save_path, "{0}.tsv".format(acoustic_savename)),
                                                      "{0}/{1}-expanded.csv".format(self.save_path,
                                                                                    acoustic_savename))

                        # read in the saved csv
                        expanded_wds_df = pd.read_csv("{0}/{1}-expanded.csv".format(self.save_path,
                                                                                    acoustic_savename), sep="\t")

                        # combine the files
                        combined = pd.merge(audio_df, expanded_wds_df, on='frameTime')

                        # average across words and save as new csv
                        wd_avgd = audio_extraction.avg_feats_across_words(combined)
                        wd_avgd.to_csv("{0}/{1}_avgd.csv".format(self.save_path, acoustic_savename))

        print("Audio and text extracted and word-level alignment completed")
        # todo: remove the following once we have gold labels
        ys_path = "{0}/asist_ys".format(self.save_path)
        os.system('if [ ! -d "{0}" ]; then mkdir -p {0}; fi'.format(ys_path))
        with open("{0}/all_ys.csv".format(ys_path), 'w') as goldfile:
            for item in gold_labels:
                goldfile.write(",".join(item))
                goldfile.write("\n")


def check_transcript(name_and_path):
    """
    Check a transcript to make sure it contains data
    This is necessary since some files contain transcripts but no data
    """
    # read in the json
    with open(name_and_path, 'r') as djfile:
        jf = djfile.read()
        fixed = ast.literal_eval(jf)

    # get only the words
    all_words = fixed['results']['items']
    print(len(all_words))
    if len(all_words) > 2:
        contains_data = True
    else:
        contains_data = False
    print(contains_data)
    # sys.exit(1)

    return contains_data


def convert_mp4_to_wav(mp4_file):
    # if the audio is in an mp4 file, convert to wav
    # file is saved to the location where the mp4 was found
    # returns the name of the file and its path
    # print(mp4_file)
    file_name = mp4_file.split(".mp4")[0]
    # print(file_name)
    wav_name = "{}.wav".format(file_name)
    # print(wav_name)
    # sys.exit(1)
    os.system("ffmpeg -i {0} {1}".format(mp4_file, wav_name))
    return wav_name


if __name__ == "__main__":
    if len(sys.argv) <= 2:
        # # define variables
        # data_path = "../../Downloads/real_search_data"
        data_path = "../../Downloads/data_flatstructure"
        save_path = "output/asist_audio"
        missions = ['mission_1', 'mission_2', 'mission_0']
        acoustic_feature_set = "IS10"
        smile_path = "~/opensmile-2.3.0"

        # try out the audio
        # print(missions)
        asist = ASISTInput(data_path, save_path, smile_path, missions=missions,
                           acoustic_feature_set=acoustic_feature_set)
        if len(sys.argv) == 1:
            asist.extract_tomcat_audio_and_text_data()
        elif len(sys.argv) == 2 and sys.argv[1] == "mp4_data":
            print("Going to extract asist audio data from mp4 files")
            asist.extract_asist_audio_data()

    elif len(sys.argv) == 6:
        # variables may be entered manually
        data_path = sys.argv[1]
        save_path = sys.argv[2]
        missions = [sys.argv[3]]
        acoustic_feature_set = sys.argv[4]
        smile_path = sys.argv[5]


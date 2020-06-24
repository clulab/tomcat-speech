# prepare the asist-produced audio and transcription data for neural classifiers

import data_prep.audio_extraction as audio_extraction

import os
import sys
import pandas as pd
import pprint
import ast
import random
import re
import time


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
        utt_enders = ['.', '?', '!']

        for item in all_words:

            if item['type'] == "punctuation" and item['alternatives'][0]['content'] in utt_enders:
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


class ZoomTranscriptToTSV:
    """
    Takes a zoom-generated transcript and gets words, speakers, and timestamps
    lines of tsv are speaker, utt_timestart, utt_timeend, utt, utt_num, where:
        speaker = name of speaker
        utt_timestart = time of utterance start
        utt_timeend = time of utterance end
        utt = words in the utterance
        utt_num = utterance number within the conversation
    """
    def __init__(self, path, txtfile, save_name):
        self.path = path
        self.text_file = path + "/" + txtfile
        self.save_name = save_name

    def convert_transcript(self, savepath):
        # convert the transcript to a tsv
        # savepath should be the full path to saving, including the file name and ext

        # set holders for turn number, times, speaker, and utterances
        speakers = ["speaker"]
        utt_timestarts = ["timestart"]
        utt_timeends = ["timeend"]
        utts = ["utt"]
        utt_nums = ["utt_num"]

        with open(self.text_file, 'r') as zoomfile:
            # skip the first line
            zoomfile.readline()
            # set a count to loop through file
            c = 0
            for line in zoomfile:
                line = line.strip()
                # increment the count
                c += 1
                # if it is a blank line
                if c % 4 == 1:
                    continue
                # if the line contains the line number
                elif c % 4 == 2:
                    utt_nums.append(line)
                # if the line contains start and end times
                elif c % 4 == 3:
                    # find the timestamps
                    utt_timestart = re.search(r'(\d\d:\d\d:\d\d\.\d\d\d) -->', line).group(1)
                    utt_timeend = re.search(r'--> (\d\d:\d\d:\d\d\.\d\d\d)', line).group(1)

                    utt_timestarts.append(utt_timestart)
                    utt_timeends.append(utt_timeend)
                # if the line contains the speaker and utterances
                else:
                    # find the speaker
                    split_line = line.split(":")
                    if len(split_line) > 1:
                        speaker = split_line[0]
                        # find the utterances, and be careful
                        # in case there are colons in transcription
                        utt = ":".join(split_line[1:])
                    else:
                        speaker = ""  # first line of one file has an utt but no speaker
                        utt = split_line[0]

                    speakers.append(speaker)
                    utts.append(utt)

        # create a tsvfile from the data
        with open(savepath + "/" + self.save_name + ".tsv", 'w') as savefile:
            for i in range(len(utt_nums)):
                savefile.write(speakers[i] + "\t" + utt_timestarts[i] + "\t" + utt_timeends[i] + "\t" +
                               utts[i] + "\t" + utt_nums[i] + "\n")


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

        # if we are using the flat structure
        if not nested:
            # iterate through items in the dir, look for the videos
            for item in os.listdir(self.path):
                if item.endswith("_video.mp4"):
                    # get the participant and experiment ids
                    experiment_id = item.split("_")[4]
                    participant_id = item.split("_")[7]

                    # set the path to the file
                    itempath = self.path + "/" + item

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
                    audio_name = audio_path.split("/")[-1] # because we don't want the full path

                    # set the name for saving csvs
                    acoustic_savename = "{0}_{1}".format(experiment_id, participant_id)

                    print("Extracting openSMILE features...")

                    # extract audio features and save csv
                    audio_extract = audio_extraction.ExtractAudio(self.path, audio_name, self.save_path,
                                                                  self.smilepath)
                    audio_extract.save_acoustic_csv(self.acoustic_feature_set,
                                                    "{0}_feats.csv".format(acoustic_savename))
        # add the gold labels to a file for later use
        if create_gold_labels:
            ys_path = "{0}/asist_ys".format(self.save_path)
            os.system('if [ ! -d "{0}" ]; then mkdir -p {0}; fi'.format(ys_path))
            with open("{0}/all_ys.csv".format(ys_path), 'w') as goldfile:
                for item in gold_labels:
                    goldfile.write(",".join(item))
                    goldfile.write("\n")

    def extract_audio_data(self, audio_path, audio_file, mp4=True):
        """
        Extract acoustic features from a given file
        """
        # get participant and experiment ids
        experiment_id = audio_file.split("_")[4]
        participant_id = audio_file.split("_")[7]

        audio_path_and_file = audio_path + "/" + audio_file

        # convert mp4 files to wav
        if mp4:
            audio_path = convert_mp4_to_wav(audio_path_and_file + ".mp4")
            audio_name = audio_path.split("/")[-1]  # because we don't want the full path
            audio_path = "/".join(audio_path.split("/")[:-1])
        else:
            audio_name = "player_audio.wav"

        # set the name for saving csvs
        acoustic_savename = "{0}_{1}".format(experiment_id, participant_id)

        # open corresponding audio and send through extraction; return csv file
        print("Extracting openSMILE features...")

        # extract audio features and save csv
        audio_extract = audio_extraction.ExtractAudio(audio_path, audio_name, self.save_path,
                                                      self.smilepath)
        audio_extract.save_acoustic_csv(self.acoustic_feature_set,
                                        "{0}_feats.csv".format(acoustic_savename))

        # return name of output file
        return "{0}_feats.csv".format(acoustic_savename)

    def extract_zoom_text_data(self, nested=False):
        """
        Convert Zoom transcriptions into usable csv transcription files
        todo: reorganize this
        """
        # if using the flat directory structure
        if not nested:
            # look for transcript items
            for item in os.listdir(self.path):
                if item.endswith("_transcript.txt"):
                    # get participant and experiment ids
                    experiment_id = item.split("_")[4]
                    participant_id = item.split("_")[7]

                    # set the path to the item
                    text_path = self.path + "/" + item

                    # set the name for saving csvs
                    text_savename = "{0}_{1}".format(experiment_id, participant_id)

                    # reorganize the transcription into a csv
                    transcript_convert = ZoomTranscriptToTSV(self.path, item, text_savename)
                    transcript_convert.convert_transcript(self.save_path)

    def align_text_and_audio_word_level(self, path_to_files, expanded_wds_file, audio_feats_file):
        """
        Align and combine text and audio data at the word level
        Expanded_wds_file : the name of the file containing words and timestamps
        audio_feats_file : the name of the file containing extracted acoustic features
        """
        # read in the saved acoustic features file
        audio_df = audio_extraction.load_feature_csv(
            "{0}/{1}".format(path_to_files, audio_feats_file))

        # read in the saved csv
        expanded_wds_df = pd.read_csv("{0}/{1}".format(path_to_files, expanded_wds_file), sep="\t")

        # combine the files
        combined = pd.merge(audio_df, expanded_wds_df, on='frameTime')

        # average across words and save as new csv
        save_name = "_".join(audio_feats_file.split("_")[:2])
        wd_avgd = audio_extraction.avg_feats_across_words(combined)
        wd_avgd.to_csv("{0}/{1}_avgd.csv".format(self.save_path, save_name), index=False)

    def extract_aws_text_data(self, aws_transcription_file, expand_data=False):
        """
        Convert AWS transcriptions into usable csv transcription files
        """
        # get participant and experiment ids
        experiment_id = aws_transcription_file.split("_")[4]
        participant_id = aws_transcription_file.split("_")[7]

        # set the name for saving csvs
        text_savename = "{0}_{1}".format(experiment_id, participant_id)

        # reorganize the transcription into a csv
        transcript_convert = JSONtoTSV(self.path, aws_transcription_file.split(".")[0],
                                       save_name=text_savename, use_txt=True)

        transcript_convert.convert_json(self.save_path)
        transcript_save_name = text_savename

        if expand_data:
            audio_extraction.expand_words("{0}/{1}".format(self.save_path, "{0}.tsv".format(text_savename)),
                                          "{0}/{1}-expanded.tsv".format(self.save_path, text_savename))
            transcript_save_name = "{}-expanded.tsv".format(text_savename)

        # return name of saved transcript file
        return transcript_save_name

    def extract_audio_and_aws_text(self, file_path, mp4=True):
        """
        A basic method to extract audio and text data
        Assumes that audio and transcriptions are in the same path
        """
        for item in os.listdir(file_path):
            if item.endswith("_transcript_full.txt"):
                # get the name of the file without _transcript_full.txt
                audio_name = "_".join(item.split("_")[:9])

                # create acoustic features for this file
                acoustic_feats_name = self.extract_audio_data(file_path, audio_name, mp4)

                # create preprocessed transcript of this file
                transcript_save_name = self.extract_aws_text_data(item, expand_data=True)

                # combine and word-align acoustic and text
                self.align_text_and_audio_word_level(self.save_path, transcript_save_name, acoustic_feats_name)

    def extract_tomcat_audio_and_text_data(self):
        """
        get the audio data; feed it through processes in audio_extraction.py
        :param missions : a list of names of the mission(s) whose data is of interest
        fixme: this contains a gold-label creation mechanism; remove it once asist has real gold labels
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

    def align_tomcat_text_and_acoustic_data(self):
        """
        To average acoustic features at the utterance level
        So hackathon data is formatted to fit in basic CNN
        """
        print("Alignment has begun")

        for item in os.listdir(self.save_path):
            # check to make sure it's a file of acoustic features
            if item.endswith("_feats.csv"):

                print(item + " found")
                # get holder for averaged acoustic items
                all_acoustic_items = []

                # add the feature file to a dataframe
                acoustic_df = pd.read_csv("{0}/{1}".format(self.save_path, item), sep=";")
                acoustic_df = acoustic_df.drop(columns=['name'])

                # add column names to holder
                col_names = acoustic_df.columns.tolist()
                col_names.append('timestart')  # so that we can join dataframes later

                # get experiment and participant IDs
                experiment_id = item.split("_")[0]
                participant_id = item.split("_")[1]

                # add the corresponding dataframe of utterance info
                utt_df = pd.read_csv("{0}/{1}_{2}.tsv".format(self.save_path, experiment_id,
                                                              participant_id), sep="\t")

                # ID all rows id df between start and end of an utterace
                for row in utt_df.itertuples():

                    # print(row)
                    # get the goal start and end time
                    start_str = row.timestart
                    end_str = row.timeend

                    start_time = split_zoom_time(start_str)
                    end_time = split_zoom_time(end_str)



                    # get the portion of the dataframe that is between the start and end times
                    this_utterance = acoustic_df[acoustic_df['frameTime'].between(start_time, end_time)]

                    # get the mean values of all columns
                    this_utt_avgd = this_utterance.mean().tolist()
                    this_utt_avgd.append(start_str)  # add timestart so dataframes can be joined

                    # add means to list
                    all_acoustic_items.append(this_utt_avgd)

                # convert all_acoustic_items to pd dataframe
                acoustic = pd.DataFrame(all_acoustic_items, columns=col_names)

                # join the dataframes
                df = pd.merge(utt_df, acoustic, on="timestart")

                # save the joined df as a new csv
                df.to_csv("{0}/{1}_{2}_avgd.csv".format(self.save_path, experiment_id, participant_id))


def split_zoom_time(timestamp):
    """
    split the hh:mm:ss.sss zoom timestamps to seconds + ms
    used to calculate start and end of acoustic features
    """
    h, m, s = timestamp.split(":")
    return (float(h) * 60 + float(m)) * 60 + float(s)


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

    # only say it contains data if it contains more than 2 words
    if len(all_words) > 2:
        contains_data = True
    else:
        contains_data = False

    return contains_data


def convert_mp4_to_wav(mp4_file):
    # if the audio is in an mp4 file, convert to wav
    # file is saved to the location where the mp4 was found
    # returns the name of the file and its path
    file_name = mp4_file.split(".mp4")[0]
    wav_name = "{}.wav".format(file_name)
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
            # extract audio from mp4 files
            asist.extract_asist_audio_data()
            # extract text from zoom transcripts
            asist.extract_asist_text_data()
            # combine the audio and text files
            asist.align_tomcat_text_and_acoustic_data()
        elif len(sys.argv) == 2 and sys.argv[1] == "prep_for_sentiment_analyzer":
            asist.extract_audio_and_aws_text(asist.path)


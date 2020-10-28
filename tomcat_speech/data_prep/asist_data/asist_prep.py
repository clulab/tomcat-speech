# prepare the asist-produced audio and transcription data for neural classifiers

import argparse
import ast
import os
import random
import re
import sys
import subprocess as sp
import tomcat_speech.data_prep.asist_data.sentiment_score_prep as sent_prep
import tomcat_speech.data_prep.audio_extraction as audio_extraction
import pandas as pd


################################################################################
############               TRANSCRIPT-ALTERING CLASS                ############
################################################################################


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

    def __init__(
            self,
            path,
            jsonfile,
            savename,
            use_txt=True,
    ):
        # def __init__(self, path, jsonfile, savename, use_txt=True):
        self.path = path
        self.jname = jsonfile
        self.savename = savename
        if use_txt:
            self.jfile = f"{path}/{jsonfile}.txt"
        else:
            self.jfile = f"{path}/{jsonfile}.json"

    def convert_json(self, savepath):
        # print(self, savepath)
        jarray = [
            ["speaker", "timestart", "timeend", "word", "utt_num", "word_num"]
        ]

        # read in the json
        with open(self.jfile, "r") as djfile:
            jf = djfile.read()
            fixed = ast.literal_eval(jf)

        # get only the words
        all_words = fixed["results"]["items"]
        # print(all_words[0])
        # set utterance and word counters
        utt = 0
        wd_num = 0
        utt_enders = [".", "?", "!"]

        for item in all_words:

            if (
                    item["type"] == "punctuation"
                    and item["alternatives"][0]["content"] in utt_enders
            ):
                utt += 1

            elif item["type"] == "pronunciation":
                speaker = 1  # todo: this assumes only 1 speaker per file for now; change if needed
                # get start and end times
                timestart = item["start_time"]
                timeend = item["end_time"]

                # get predicted word
                word = item["alternatives"][0][
                    "content"
                ].lower()  # todo: will there always be only 1 alternative?
                word_num = wd_num

                # get the utterance and word number
                utt_num = utt

                # update wd counter
                wd_num += 1

                jarray.append(
                    [
                        str(speaker),
                        timestart,
                        timeend,
                        word,
                        str(utt_num),
                        str(word_num),
                    ]
                )
        with open(f"{savepath}/{self.savename}.tsv", "w") as tsvfile:
            for item in jarray:
                tsvfile.write("\t".join(item) + "\n")


class ZoomTranscriptToTSV:
    """
    Takes a Zoom-generated transcript and gets words, speakers, and timestamps
    lines of tsv are speaker, utt_timestart, utt_timeend, utt, utt_num, where:
        speaker = name of speaker
        utt_timestart = time of utterance start
        utt_timeend = time of utterance end
        utt = words in the utterance
        utt_num = utterance number within the conversation
    """

    def __init__(self, path, txtfile, savename):
        self.path = path
        self.text_file = path + "/" + txtfile
        self.savename = savename

    def convert_transcript(self, savepath):
        # convert the transcript to a tsv
        # savepath should be the full path to saving, including the file name and ext
        # set holders for turn number, times, speaker, and utterances
        speakers = ["speaker"]
        utt_timestarts = ["timestart"]
        utt_timeends = ["timeend"]
        utts = ["utt"]
        utt_nums = ["utt_num"]

        with open(self.text_file, "r") as zoomfile:
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
                    utt_timestart = re.search(
                        r"(\d\d:\d\d:\d\d\.\d\d\d) -->", line
                    ).group(1)
                    utt_timeend = re.search(
                        r"--> (\d\d:\d\d:\d\d\.\d\d\d)", line
                    ).group(1)

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
        with open(savepath + "/" + self.savename + ".tsv", "w") as savefile:
            for i in range(len(utt_nums)):
                savefile.write(
                    "\t".join(
                        (
                            speakers[i],
                            utt_timestarts[i],
                            utt_timeends[i],
                            utts[i],
                            utt_nums[i],
                        )
                    )
                    + "\n"
                )


################################################################################
############                 ASIST DATA INPUT CLASS                 ############
################################################################################


class ASISTInput:
    def __init__(
            self,
            asist_path,
            save_path,
            smilepath="opensmile-2.3.0",
            acoustic_feature_set="IS10",
            missions=None,
    ):
        self.path = asist_path
        self.save_path = save_path
        self.smilepath = smilepath  # the path to your openSMILE installation

        # set of acoustic features to extract
        # options are "IS09", "IS10", "IS11", "IS12", "IS13"
        self.acoustic_feature_set = acoustic_feature_set

        if missions:
            self.missions = missions  # list of names of missions of interest (e.g.['mission_2'])
        else:
            self.missions = ["mission_2"]

    def extract_audio_data(
            self, audio_path, audio_file, mp4=False, use_missions=False, m4a=True
    ):
        """
        Extract acoustic features from a given file
        """
        # internal files with missions had a different naming convention
        if not use_missions:
            # get participant and experiment ids
            experiment_id = audio_file.split("_")[0]
            participant_id = audio_file.split("_")[7]
            # set the name for saving csvs
            acoustic_savename = f"{experiment_id}_{participant_id}"
            # print("acoustic_savename", acoustic_savename)
        else:
            participant_id = audio_file.split("_")[0]
            mission = audio_file.split("_")[-1]
            acoustic_savename = f"{participant_id}_mission_{mission}"
            # print(acoustic_savename)

        # convert mp4 files to wav if needed
        if mp4:
            audio_path_and_file = audio_path + "/" + audio_file
            audio_path = audio_extraction.convert_mp4_to_wav(
                audio_path_and_file + ".mp4"
            )
            audio_name = audio_path.split("/")[
                -1
            ]  # because we don't want the full path
            audio_path = "/".join(audio_path.split("/")[:-1])
        if m4a:
            print("m4a files detected")
            audio_path_and_file = audio_path + "/" + audio_file
            audio_path = audio_extraction.convert_m4a_to_wav(
                audio_path_and_file + ".m4a"
            )
            audio_name = audio_path.split("/")[
                -1
            ]  # because we don't want the full path
            audio_path = "/".join(audio_path.split("/")[:-1])
        else:
            audio_name = "player_audio.wav"

        # open corresponding audio and send through extraction; return csv file
        print("Extracting openSMILE features...")

        # extract audio features and save csv if not already extracted
        if os.path.exists(
                self.save_path + "/" + acoustic_savename + "_feats.csv"
        ):
            print(
                f"Acoustic features already extracted for file {acoustic_savename}"
            )
        else:
            audio_extract = audio_extraction.ExtractAudio(
                audio_path, audio_name, self.save_path, self.smilepath
            )
            audio_extract.save_acoustic_csv(
                self.acoustic_feature_set,
                f"{acoustic_savename}_feats.csv",
            )

        # return name of output file
        return f"{acoustic_savename}_feats.csv"

    def extract_zoom_text_data(self):
        """
        Convert Zoom transcriptions into usable csv transcription files
        """
        # look for transcript items
        for item in os.listdir(self.path):
            if item.endswith("_transcript.txt"):
                # get participant and experiment ids
                experiment_id = item.split("_")[4]
                participant_id = item.split("_")[7]

                # set the path to the item
                text_path = self.path + "/" + item

                # set the name for saving csv files
                text_savename = f"{experiment_id}_{participant_id}"

                # reorganize the transcription into a csv
                transcript_convert = ZoomTranscriptToTSV(
                    self.path, item, text_savename
                )
                transcript_convert.convert_transcript(self.save_path)

    def align_text_and_audio_word_level(
            self, path_to_files, expanded_wds_file, audio_feats_file
    ):
        """
        Align and combine text and audio data at the word level
        Expanded_wds_file : the name of the file containing words and timestamps
        audio_feats_file : the name of the file containing extracted acoustic features
        """
        # read in the saved acoustic features file

        audio_df = audio_extraction.load_feature_csv(
            f"{path_to_files}/{audio_feats_file}"
        )
        # read in the saved csv
        expanded_wds_df = pd.read_csv(
            f"{path_to_files}/{expanded_wds_file}", sep="\t"
        )
        # add break point

        # combine the files
        combined = pd.merge(audio_df, expanded_wds_df, on="frameTime")
        # find out if there are speaker labels

        # average across words and save as new csv
        savename = "_".join(audio_feats_file.split("_")[:2])
        wd_avgd = audio_extraction.avg_feats_across_words(combined)
        wd_avgd.to_csv(f"{self.save_path}/{savename}_avgd.csv", index=False)

    def extract_aws_text_data(
            self, aws_transcription_file, expand_data=False, use_missions=False
    ):
        """
        Convert AWS transcriptions into usable csv transcription files
        """
        if not use_missions:
            # get participant and experiment ids
            experiment_id = aws_transcription_file.split("_")[
                1
            ]  # edit according to use case
            participant_id = aws_transcription_file.split("_")[
                8
            ]  # edit according to use case
            # set the name for saving csv files
            text_savename = f"{experiment_id}_{participant_id}"
            # create instance of JSON to TSV class
            transcript_convert = JSONtoTSV(
                self.path,
                aws_transcription_file.split(".json")[
                    0
                ],  # edit according to use case
                savename=text_savename,
                use_txt=False,
            )
        else:
            # set the name for saving csv files
            text_savename = aws_transcription_file
            # set the path to the item--participant_id is the directory name
            item_path = (
                    self.path + "/" + text_savename.split("_")[0]
            )  # edit according to use case
            # set the name of the mission--transcript names contain the mission
            mission = (
                    "mission_" + text_savename.split("_")[-1]
            )  # edit according to use case
            # create instance of JSON to TSV class
            transcript_convert = JSONtoTSV(
                item_path,
                f"{mission}_transcript_full",
                savename=text_savename,
                use_txt=False,
            )

        # reorganize the transcription into a csv
        transcript_convert.convert_json(self.save_path)
        transcript_savename = text_savename

        if expand_data:
            audio_extraction.expand_words(
                f"{self.save_path}/{text_savename}.tsv",
                f"{self.save_path}/{text_savename}-expanded.tsv",
            )
            transcript_savename = f"{text_savename}-expanded.tsv"

        # return name of saved transcript file
        return transcript_savename

    # use the previously-defined functions to extract audio and text for different conditions
    def extract_audio_and_aws_text(self, file_path, mp4=False, m4a=True):
        """
        A basic method to extract audio and text data
        Assumes that audio and transcriptions are in the same path
        """
        for item in os.listdir(file_path):
            if item.endswith("_transcript_full.txt"):
                # get the name of the file without _transcript_full.txt
                audio_name = "_".join(
                    item.split("_")[:9]
                )  # edit according to use case

                # create acoustic features for this file
                acoustic_feats_name = self.extract_audio_data(
                    file_path, audio_name, mp4
                )

                # create preprocessed transcript of this file
                transcript_savename = self.extract_aws_text_data(
                    item, expand_data=True
                )

                # combine and word-align acoustic and text
                self.align_text_and_audio_word_level(
                    self.save_path, transcript_savename, acoustic_feats_name
                )
            elif item.startswith("transcript") and item.endswith(".json"):
                # get the name of the file without transcript and .json
                audio_name1 = "_".join(
                    item.split("_")[1:]
                )  # edit according to use case
                audio_name = audio_name1.split(".json")[
                    0
                ]  # edit according to use case
                # create acoustic features for this file
                acoustic_feats_name = self.extract_audio_data(
                    file_path, audio_name, m4a
                )
                # create preprocessed transcript of this file
                transcript_savename = self.extract_aws_text_data(
                    item, expand_data=True
                )

                # combine and word-align acoustic and text
                self.align_text_and_audio_word_level(
                    self.save_path, transcript_savename, acoustic_feats_name
                )

    def extract_audio_and_aws_text_with_missions(self, mp4=False):
        """
        Extract audio and aws transcripts from internal data
        Assumes multiple possible missions to distinguish between
        """
        for item in os.listdir(self.path):
            item_path = f"{self.path}/{item}"
            if os.path.isdir(item_path):
                for mission in self.missions:
                    if f"{mission}_transcript_full.txt" in os.listdir(
                            item_path
                    ) and check_transcript(
                        f"{item_path}/{mission}_transcript_full.txt"
                    ):
                        participant_id = item
                        name_and_mission = participant_id + "_" + mission

                        # set the name for saving csvs
                        acoustic_savename = f"{participant_id}_{mission}"

                        # open corresponding audio and send through extraction; return csv file
                        audio_path = f"{item_path}/{mission}"

                        # create acoustic features for this file
                        acoustic_feats_name = self.extract_audio_data(
                            audio_path,
                            acoustic_savename,
                            mp4,
                            use_missions=True,
                        )

                        # create preprocessed transcript of this file
                        transcript_savename = self.extract_aws_text_data(
                            name_and_mission,
                            expand_data=True,
                            use_missions=True,
                        )

                        # align features and transcripts
                        self.align_text_and_audio_word_level(
                            self.save_path,
                            transcript_savename,
                            acoustic_feats_name,
                        )

    def extract_audio_and_zoom_text(self, file_path, mp4=True, m4a=False):
        """
        Extract the audio and zoom-generated transcripts; keep them separate
        """
        # extract audio
        for item in os.listdir(file_path):
            # if item.endswith("_video.mp4"):
            if item.endswith(".mp4"):
                # get the name of the audio file without .mp4
                audio_name = item.split(".mp4")[0]

                # create acoustic features for this file
                _ = self.extract_audio_data(file_path, audio_name, mp4)
            elif item.endswith(".m4a"):
                # get the name of the audio file without .m4a
                audio_name = item.split(".m4a")[0]

                # create acoustic features for this file
                _ = self.extract_audio_data(file_path, audio_name, m4a)
        # extract transcripts
        self.extract_zoom_text_data()

        # align at utterance level
        self.align_tomcat_text_and_acoustic_data()

    def align_tomcat_text_and_acoustic_data(self):
        """
        To average acoustic features at the utterance level
        So hackathon data is formatted to fit in basic CNN
        Assumes that the files to be manipulated are all found in the save_path
        """
        print("Alignment has begun")

        for item in os.listdir(self.save_path):
            # check to make sure it's a file of acoustic features
            if item.endswith("_feats.csv") and "mission" not in item.split(
                    "_"
            ):

                print(item + " found")
                # get holder for averaged acoustic items
                all_acoustic_items = []

                # add the feature file to a dataframe
                acoustic_df = pd.read_csv(f"{self.save_path}/{item}", sep=";")
                acoustic_df = acoustic_df.drop(columns=["name"])

                # add column names to holder
                col_names = acoustic_df.columns.tolist()
                col_names.append(
                    "timestart"
                )  # so that we can join dataframes later

                # get experiment and participant IDs
                experiment_id = item.split("_")[
                    0
                ]  # edit according to use case
                participant_id = item.split("_")[
                    1
                ]  # edit according to use case

                # add the corresponding dataframe of utterance info
                utt_df = pd.read_table(
                    f"{self.save_path}/{experiment_id}_{participant_id}.tsv"
                )

                # ID all rows id df between start and end of an utterace
                for row in utt_df.itertuples():
                    # get the goal start and end time
                    start_str = row.timestart
                    end_str = row.timeend

                    start_time = split_zoom_time(start_str)
                    end_time = split_zoom_time(end_str)

                    # get the portion of the dataframe that is between the start and end times
                    this_utterance = acoustic_df[
                        acoustic_df["frameTime"].between(start_time, end_time)
                    ]

                    # use this_utterance as input for gender_classifier.
                    # get the mean values of all columns

                    this_utt_avgd = this_utterance.mean().tolist()
                    this_utt_avgd.append(
                        start_str
                    )  # add timestart so dataframes can be joined

                    # add means to list
                    all_acoustic_items.append(this_utt_avgd)

                # convert all_acoustic_items to pd dataframe
                acoustic = pd.DataFrame(all_acoustic_items, columns=col_names)

                # join the dataframes
                df = pd.merge(utt_df, acoustic, on="timestart")

                # save the joined df as a new csv
                df.to_csv(
                    f"{self.save_path}/{experiment_id}_{participant_id}_avgd.csv"
                )


################################################################################
############              ASIST PREP HELPER FUNCTIONS               ############
################################################################################


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
    with open(name_and_path, "r") as djfile:
        jf = djfile.read()
        fixed = ast.literal_eval(jf)

    # get only the words
    all_words = fixed["results"]["items"]

    # only say it contains data if it contains more than 2 words
    if len(all_words) > 2:
        contains_data = True
    else:
        contains_data = False

    return contains_data


def create_random_gold_labels(data_path):
    """
    Create gold labels before real ones have been made
    """
    # create holder for gold labels
    gold_labels = [["sid", "overall"]]
    # create set for participants
    all_participants = set()

    # add all participants
    [
        all_participants.add(item.split("_")[1])  # edit according to use case
        for item in os.listdir(data_path)
        if item.endswith("_avgd.csv")
    ]

    # add participant_ids and gold labels to holder
    [
        gold_labels.append(
            (participant, str(random.randint(0, 1)))
        )  # edit according to use case
        for participant in all_participants
    ]

    # save holder to file
    ys_path = f"{data_path}/asist_ys"
    os.makedirs(ys_path, exist_ok=True)
    with open(f"{ys_path}/all_ys.csv", "w") as goldfile:
        for item in gold_labels:
            goldfile.write(",".join(item) + "\n")


def run_sentiment_analysis_pipeline(asist, sentiment_text_path):
    """
    Run the full text-based sentiment analysis portion of the pipeline
    asist : an ASISTInput object

    """
    # prepare audio and text data
    asist.extract_audio_and_aws_text(asist.path)

    # prepare utterances for input into analyzer
    transcription_path = asist.path

    # get instance of TranscriptPrepper class
    transcript_prepper = sent_prep.TranscriptPrepper(
        transcription_path, sentiment_text_path
    )

    # prepare transcripts
    transcript_prepper.split_transcripts_by_utterance()

    # holder for all output file names
    out_names = []

    # put utterances through analyzer
    for f in os.listdir(sentiment_text_path):
        # find files produced by megh function
        if f.endswith("_transcript_split.txt"):
            # prepare name for output files
            out_name = "_".join(f.split("_")[:-3]) + "_sentiment_out.txt"
            out_names.append(out_name)
            # run shell script
            sp.run(
                [
                    "./get_asist_sentiment_analysis.sh",
                    f"{sentiment_text_path}/{f}",
                    f"{sentiment_text_path}/{out_name}",
                ]
            )

    # return the names of all score files created
    return out_names


################################################################################
############                         USAGE                          ############
################################################################################
p = "~"
home_dir = os.path.expanduser(p)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        help="Directory in which the data resides",
        default=str(home_dir) + "/Downloads/data_flatstructure",
    )
    parser.add_argument(
        "--save_path",
        help="Directory to which the output should be written",
        default="output/asist_audio",
    )
    parser.add_argument(
        "--opensmile_path",
        help="Path to OpenSMILE",
        default=str(home_dir) + "/opensmile-2.3.0",
    )
    parser.add_argument(
        "--sentiment_text_path",
        help="Path to text-based sentiment analysis outputs",
        default="output/",
    )
    if len(sys.argv) <= 2:
        # define variables
        # data_path = "../../Downloads/real_search_data"
        data_path = str(home_dir) + "/Downloads/data_flatstructure"
        save_path = "output/asist_audio"
        sentiment_text_path = "output/"
        missions = ["mission_1", "mission_2", "mission_0"]
        acoustic_feature_set = "IS10"
        smile_path = str(home_dir) + "/opensmile-2.3.0"

        # create instance of input class
        asist = ASISTInput(
            data_path,
            save_path,
            smile_path,
            missions=missions,
            acoustic_feature_set=acoustic_feature_set,
        )
        if len(sys.argv) == 1:
            # asist.extract_tomcat_audio_and_text_data()
            asist.extract_audio_and_aws_text_with_missions()
        elif len(sys.argv) == 2 and sys.argv[1] == "mp4_data":
            # extract audio + zoom text, use utterance averaging of features for alignment
            asist.extract_audio_and_zoom_text(asist.path)
        elif len(sys.argv) == 2 and sys.argv[1] == "m4a_data":
            # extract audio + zoom text, use utterance averaging of features for alignment
            asist.extract_audio_and_aws_text(asist.path)
        elif (
                len(sys.argv) == 2 and sys.argv[1] == "prep_for_sentiment_analyzer"
        ):
            run_sentiment_analysis_pipeline(asist, sentiment_text_path)

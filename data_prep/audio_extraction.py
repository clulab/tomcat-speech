# extract information from audio
# this information is modified from the code in:
# https://github.com/jmculnan/audio_feature_extraction (private repo)

# required packages
import os, sys
import json, re
import pprint

import pandas as pd


class TRSToCSV:
    """
    Takes a trs file and converts it to a csv
    lines of csvfile are speaker,timestart,timeend, word, utt_num where:
        speaker   = caller or patient
        timestart = time of start of word
        timeend   = time of end of word
        word      = the word predicted
        utt_num   = the utterance number within the conversation
        word_num  = the word number (useful for averaging over words)
    """

    def __init__(self, path, trsfile):
        self.path = path
        self.tname = trsfile
        self.tfile = "{0}/{1}.trs".format(path, trsfile)

    def convert_trs(self, savepath):
        trs_arr = [["speaker", "timestart", "timeend", "word", "utt_num", "word_num"]]
        with open(self.tfile, "r") as trs:
            print(self.tfile)
            # for line in trs:
            # try:
            utt = 0
            spkr = 0
            wd_num = 0
            coach_participant = {}
            for line in trs:
                if "<Speaker id" in line:
                    participant_num = re.search(r'"spkr(\d)', line).group(1)
                    participant = re.search(r'name="(\S+)"', line).group(1)
                    coach_participant[participant_num] = participant
                if "<Turn " in line:
                    # print(line)
                    timestart = re.search(r'startTime="(\d+.\d+)"', line).group(1)
                    timeend = re.search(r'endTime="(\d+.\d+)"', line).group(1)
                    speaker = re.search(r'spkr(\d)">', line).group(1)
                    word = re.search(r"/>(\S+)</Turn>", line).group(1)

                    if (
                        coach_participant[speaker] == "coach"
                        or coach_participant[speaker] == "Coach"
                    ):
                        # print("Coach found")
                        real_speaker = 2
                    elif (
                        coach_participant[speaker] == "participant"
                        or coach_participant == "Participant"
                    ):
                        # print("Participant found")
                        real_speaker = 1
                    else:
                        print("There was some problem here")
                        sys.exit(1)

                    wd_num += 1
                    if spkr != real_speaker:
                        spkr = real_speaker
                        utt += 1

                    trs_arr.append(
                        [real_speaker, timestart, timeend, word, utt, wd_num]
                    )
        with open("{0}/{1}.tsv".format(savepath, self.tname), "w") as cfile:
            for item in trs_arr:
                cfile.write(
                    str(item[0])
                    + "\t"
                    + str(item[1])
                    + "\t"
                    + str(item[2])
                    + "\t"
                    + str(item[3])
                    + "\t"
                    + str(item[4])
                    + "\t"
                    + str(item[5])
                    + "\n"
                )


class ExtractAudio:
    """
    Takes audio and extracts features from it using openSMILE
    """

    def __init__(self, path, audiofile, savedir, smilepath="~/opensmile-2.3.0"):
        self.path = path
        self.afile = path + "/" + audiofile
        self.savedir = savedir
        self.smile = smilepath

    def save_acoustic_csv(self, feature_set, savename):
        """
        Get the CSV for set of acoustic features for a .wav file
        feature_set : the feature set to be used
        savename : the name of the saved CSV
        Saves the CSV file
        """
        # todo: can all of these take -lldcsvoutput ?
        if feature_set == "IS09":
            fconf = "IS09_emotion.conf"
        elif feature_set == "IS10":
            fconf = "IS10_paraling.conf"
        elif feature_set == "IS12":
            fconf = "IS12_speaker_trait.conf"
        elif feature_set == "IS13":
            fconf = "IS13_ComParE.conf"
        else:
            fconf = "IS09_emotion.conf"
            # fconf = "emobase.conf"

        # check to see if save path exists; if not, make it
        os.system('if [ ! -d "{0}" ]; then mkdir -p {0}; fi'.format(self.savedir))

        # run openSMILE
        os.system(
            "{0}/SMILExtract -C {0}/config/{1} -I {2} -lldcsvoutput {3}/{4}".format(
                self.smile, fconf, self.afile, self.savedir, savename
            )
        )


class AudioSplit:
    """Takes audio, can split and join using ffmpeg"""

    def __init__(self, path, pathext, audio_name, diarized_csv):
        self.path = path
        self.aname = audio_name
        self.cname = diarized_csv
        self.afile = "{0}/{1}".format(path, audio_name)
        self.cfile = "{0}/{1}".format(path, diarized_csv)
        self.ext = pathext
        self.fullp = "{0}/{1}".format(path, pathext)

    def split_audio(self):
        """
        Splits audio based on an input csvfile.
        csvfile is assumed to start with the following format:
          speaker,timestart,timeend where
          speaker   = caller or patient
          timestart = time of start of turn
          timeend   = time of turn end
        """
        n = 0

        if not os.path.exists(self.fullp):
            os.mkdir(self.fullp)

        with open(self.cfile, "r") as csvfile:
            for line in csvfile:
                speaker, timestart, timeend = line.strip().split(",")[:3]
                if not os.path.exists("{0}/{1}".format(self.fullp, speaker)):
                    os.mkdir("{0}/{1}".format(self.fullp, speaker))
                os.system(
                    "ffmpeg -i {0} -ss {1} -to {2} {3}/{4}/{5}.wav -loglevel quiet".format(
                        self.afile, timestart, timeend, self.fullp, speaker, n
                    )
                )
                if n % 1000 == 0:
                    print("Completed {0} lines".format(n + 1))
                n += 1

    def make_textfile(self, audiodir, speaker):
        """
        Make a .txt file containing the names of all audio in the directory
        Used for ffmpeg concatenation
        """
        txtfilename = "{0}-{1}.txt".format(self.ext, speaker)
        txtfilepath = "{0}/{1}/{2}".format(self.fullp, speaker, txtfilename)
        with open(txtfilepath, "w") as txtfile:
            for item in os.listdir(audiodir):
                if item[-4:] == ".wav":
                    txtfile.write("file '{0}'\n".format(item))

    def join_audio(self, txtfile, speaker):
        """
        Joins audio in an input directory using a textfile with path info
        """
        if not os.path.exists("{0}/output".format(self.path)):
            os.mkdir("{0}/output".format(self.path))

        outputname = "{0}-{1}.wav".format(self.ext, speaker)
        print(
            "ffmpeg -f concat -safe 0 -i {0}/{1}/{2} -c copy {3}/output/{4}".format(
                self.fullp, speaker, txtfile, self.path, outputname
            )
        )

        os.system(
            "ffmpeg -f concat -safe 0 -i {0}/{1}/{2} -c copy {3}/output/{4} -loglevel quiet".format(
                self.fullp, speaker, txtfile, self.path, outputname
            )
        )
        print("Concatenation completed for {0}".format(self.fullp))


def transform_audio(txtfile):
    """
    Used for taking audio and transforming it in the way initially envisioned
    for LIvES project.
    txtfile = the path to a file containing rows of:
        path : a path to the audio data
        trsfile : the name of the transcription file (without .trs)
        callwav : the name of wav file being transformed
    todo: does this need to be changed to be relevant for this input?
    """
    with open(txtfile, "r") as tfile:
        # print("textfile opened!")
        for line in tfile:
            # print("Line is: " + line)
            path, trsfile, callwav = line.strip().split(",")

            csvfile = "{0}.csv".format(trsfile)
            extension = callwav.split(".")[0]
            speakers = ["1", "2"]

            # diarized_input = DiarizedToCSV(path, jsonfile)
            diarized_input = TRSToCSV(path, trsfile)
            # print("diarized_input created")
            diarized_input.convert_trs()
            # print("conversion to json happened")

            audio_input = AudioSplit(path, extension, callwav, csvfile)
            audio_input.split_audio()

            for speaker in speakers:
                audio_input.make_textfile(
                    "{0}/{1}/{2}".format(path, extension, speaker), speaker
                )
                audio_input.join_audio(
                    "{0}-{1}.txt".format(extension, speaker), speaker
                )

            os.system("rm -r {0}/{1}".format(path, extension))


def load_feature_csv(audio_csv):
    """
    Load audio features from an existing csv
    audio_csv = the path to and name of the csv file
    """
    # todo: should we add ability to remove columns here, or somewhere else?
    return pd.read_csv(audio_csv, sep=";")


def drop_cols(self, dataframe, to_drop):
    """
    to drop columns from pandas dataframe
    used in get_features_dict
    """
    return dataframe.drop(to_drop, axis=1).to_numpy().tolist()


def expand_words(trscsv, file_to_save):
    """
    Expands transcription file to include values at every 10ms
    Used to combine word, speaker, utt information with features
    extracted from OpenSMILE
    :param trscsv: the transcription tsv
    :param file_to_save:
    :return:
    """
    saver = [["frameTime", "speaker", "word", "utt_num", "word_num"]]
    with open(trscsv, "r") as tcsv:
        tcsv.readline()
        for line in tcsv:
            speaker, timestart, timeend, word, utt_num, wd_num = line.strip().split(
                "\t"
            )
            saver.append([timestart, speaker, word, utt_num, wd_num])
            newtime = float(timestart) + 0.01
            while newtime < float(timeend):
                newtime += 0.01
                saver.append([str(newtime), speaker, word, utt_num, wd_num])
            saver.append([timeend, speaker, word, utt_num, wd_num])
    with open(file_to_save, "w") as wfile:
        for item in saver:
            wfile.write("\t".join(item))
            wfile.write("\n")


def avg_feats_across_words(feature_df):
    """
    Takes a pandas df of acoustic feats and collapses it into one
    with feats avg'd across words
    :param feature_df: pandas dataframe of features in 24msec intervals
    :return: a new pandas df
    """
    # summarize + avg like dplyr in R
    feature_df = feature_df.groupby(
        ["word", "speaker", "utt_num", "word_num"], sort=False
    ).mean()
    feature_df = feature_df.reset_index()
    return feature_df


class GetFeatures:
    """
    Takes input files and gets acoustic features
    Organizes features as required for this project
    Combines data from acoustic csv + transcription csv
    """

    def __init__(self, path, acoustic_csv, trscsv):
        self.path = path
        self.acoustic_csv = acoustic_csv
        self.trscsv = trscsv

    def get_features_dict(self, dropped_cols=None):
        """
        Get the set of phonological/phonetic features
        """
        # create a holder for features
        feature_set = {}

        # iterate through csv files created by openSMILE
        for csvfile in os.listdir(self.savepath):
            if csvfile.endswith(".csv"):
                csv_name = csvfile.split(".")[0]
                # get data from these files
                csv_data = pd.read_csv(
                    "{0}/{1}".format(self.savepath, csvfile), sep=";"
                )
                # drop name and time frame, as these aren't useful
                if dropped_cols:
                    csv_data = self.drop_cols(csv_data, dropped_cols)
                else:
                    csv_data = (
                        csv_data.drop(["name", "frameTime"], axis=1).to_numpy().tolist()
                    )
                if "nan" in csv_data or "NaN" in csv_data or "inf" in csv_data:
                    pprint.pprint(csv_data)
                    print("Data contains problematic data points")
                    sys.exit(1)

                # add it to the set of features
                feature_set[csv_name] = csv_data

        return feature_set


def convert_mp4_to_wav(mp4_file):
    # if the audio is in an mp4 file, convert to wav
    # file is saved to the location where the mp4 was found
    # returns the name of the file and its path
    file_name = mp4_file.split(".mp4")[0]
    wav_name = "{}.wav".format(file_name)
    # check if the file already exists
    if not os.path.exists(wav_name):
        os.system("ffmpeg -i {0} -ac 1 {1}".format(mp4_file, wav_name))
    # otherwise, print that it exists
    else:
        print("{} already exists".format(wav_name))

    return wav_name
    
def convert_m4a_to_wav(m4a_file):
    # if the audio is in an mp4 file, convert to wav
    # file is saved to the location where the mp4 was found
    # returns the name of the file and its path
    file_name = m4a_file.split(".m4a")[0]
    wav_name = "{}.wav".format(file_name)
    # check if the file already exists
    if not os.path.exists(wav_name):
        os.system("ffmpeg -i {0} -ac 1 {1}".format(m4a_file, wav_name))
    # otherwise, print that it exists
    else:
        print("{} already exists".format(wav_name))

    return wav_name

def extract_portions_of_mp4_or_wav(
    path_to_sound_file,
    sound_file,
    start_time,
    end_time,
    save_path=None,
    short_file_name=None,
):
    """
    Extracts only necessary portions of a sound file
    sound_file : the name of the full file to be adjusted
    start_time : the time at which the extracted segment should start
    end_time : the time at which the extracted segment should end
    short_file_name : the name of the saved short sound file
    """
    # set full path to file
    full_sound_path = os.path.join(path_to_sound_file, sound_file)

    # check sound file extension
    if sound_file.endswith(".mp4"):
        # convert to wav first
        full_sound_path = convert_mp4_to_wav(full_sound_path)

    if sound_file.endswith(".m4a"):
        # convert m4a to wav first
        full_sound_path = convert_m4a_to_wav(full_sound_path)

    if not short_file_name:
        print("short file name not found")
        short_file_name = "{0}_{1}_{2}.wav".format(
            sound_file.split(".")[0], start_time, end_time
        )

    if save_path is not None:
        save_name = save_path + "/" + short_file_name
    else:
        save_name = path_to_sound_file + "/" + short_file_name

    # get shortened version of file
    os.system(
        "ffmpeg -i {0} -ss {1} -to {2} {3}".format(
            full_sound_path, start_time, end_time, save_name
        )
    )

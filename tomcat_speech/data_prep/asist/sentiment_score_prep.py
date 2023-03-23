# prepare sentiment scores output by a text-based analyzer
# organize them for input into a neural network classifier
#   or for use with a hybrid sentiment analyzer

import glob
import os
import re
import sys

import pandas as pd
import warnings


class TranscriptPrepper:
    """
    Prepares transcripts to be entered into the sentiment analyzer
    Identifies transcripts and splits them at utterance boundary
    """

    def __init__(self, path_to_transcripts, save_path):
        self.transcript_path = path_to_transcripts
        self.save_path = save_path
        os.system('if [ ! -d "{0}" ]; then mkdir -p {0}; fi'.format(save_path))

        # get a list of transcript files
        self.transcript_files = self.get_list_of_transcripts()

    def get_list_of_transcripts(self):
        """
        find all transcript files in a given directory
        location : the path to directory containing transcripts
        """
        fold = self.transcript_path + "/*video_transcript.txt"
        files = glob.glob(fold)

        if len(files) == 0:
            sys.exit("No relevant transcript files")

        return files

    def split_transcripts_by_utterance(self):
        """
        uses [., ?, !] to split transcripts at sentence boundaries
        saves the split transcripts into files
        used as input to text-based sentiment analysis
        """
        for item in self.transcript_files:
            # get just the file name
            item_name = item.split("/")[-1].split(".txt")[0]

            # create name for the output file
            output_name = self.save_path + "/" + item_name + "_split.txt"

            # find utterances
            with open(item) as f:
                my_list = list(f)
                split_utts = re.split("\.\s|\?\s|\!\s", my_list[0])

            # save the split transcript to a file
            with open(output_name, "w") as file_handler:
                for utt in split_utts:
                    file_handler.write("%s\n" % utt)


class SentimentScores:
    def __init__(self, path_to_scores, score_text_names):
        """
        path_to_scores : the path to sentiment score files
        Score_texts : a list of file names where
        sentiment scores are contained
        """
        self.score_text_names = score_text_names
        self.path_to_scores = path_to_scores
        # self.scores_paths = [os.path.join(self.path_to_scores, name)
        #                      for name in score_text_names]

    def prepare_scores(self, counts=True, probabilities=True):
        """
        Prepare score texts for input into the pipeline
        Select counts, probabilities, or both
        """
        # prepare holder for score texts
        score_text_dict = {}

        # prepare to extract only the necessary columns
        cols, colnames = set_cols_and_colnames(
            counts, probabilities, include_utts=False
        )

        for name in self.score_text_names:
            # get the path for a given score text
            score_path = os.path.join(self.path_to_scores, name)

            # load the score text as a pandas df
            score_text = pd.read_csv(
                score_path, delim_whitespace=True, usecols=cols, names=colnames
            )

            # save score text to dict
            score_text_dict[name] = score_text

        return score_text_dict

    def join_words_with_predictions(
        self, input_path, utterance_input_names, counts=True, probabilities=True,
    ):
        """
        Connects scores with the utterances they were predicted on
        input_path : the path to the files containing input utterances
        utterance_input_names : a list of the names for input files
        """
        # check to ensure input and output lists are same size
        if len(utterance_input_names) != len(self.score_text_names):
            sys.exit("different numbers of input and score files")

        # create holder for merged
        utterances_with_preds = {}

        # prepare to extract only the necessary columns
        cols, colnames = set_cols_and_colnames(counts, probabilities, include_utts=True)

        # for each item in the output list
        for input_name in utterance_input_names:
            # ensure the input name does not contain the full path
            input_name = input_name.split("/")[-1]

            # get the study and subject IDs
            study_id = input_name.split("_")[4]
            participant_id = input_name.split("_")[7]

            # get the path
            utterance_text_path = os.path.join(input_path, input_name)

            # ID the equivalent output item
            output_name = [
                name
                for name in self.score_text_names
                if study_id in name and participant_id in name
            ]
            if len(output_name) == 1:
                output_path = os.path.join(self.path_to_scores, output_name[0])
            else:
                sys.exit("Multiple output files for the same study and participant")

            # extract the columns needed into pandas DFs
            # we want one column per line, so sep should be something not found
            utterance_text = pd.read_csv(
                utterance_text_path, sep="\t\t", engine="python", names=["utt"]
            )

            scores_text = pd.read_csv(
                output_path, delim_whitespace=True, names=colnames
            )

            # merge dataframes
            utts_with_scores = pd.concat([utterance_text, scores_text], axis=1)

            # add to holder
            utterances_with_preds[
                "{0}_{1}".format(study_id, participant_id)
            ] = utts_with_scores

        return utterances_with_preds


# helper functions
def set_cols_and_colnames(counts=True, probabilities=True, include_utts=True):
    """
    Set the indices of columns to extract and prepare the column names
    """
    # prepare to extract only the necessary columns
    if counts and probabilities:
        cols = [0, 1, 2, 3]
        colnames = ["pos_count", "neg_count", "pos_prob", "neg_prob"]
    elif counts:
        cols = [0, 1]
        colnames = ["pos_count", "neg_count"]
    elif probabilities:
        cols = [2, 3]
        colnames = ["pos_prob", "neg_prob"]
    else:
        warnings.warn("no sentiment scores selected; defaulting to probabilities")
        cols = [2, 3]
        colnames = ["pos_prob", "neg_prob"]

    return cols, colnames

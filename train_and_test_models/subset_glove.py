# test glove subsetting

import data_prep.glove_subsetting as glove
import pandas as pd
import numpy as np

# set paths
# vocab_path = "/Volumes/LIvES/multimodal_data/"
glove_path = "../../glove.42B.300d.txt"
save_path = "../../glove.short.300d.punct.txt"
short_path = save_path
vec_length = 300

vocab_file_path = "../../datasets/multimodal_datasets/MUStARD/mustard_utts.tsv"
vocab_file = pd.read_csv(vocab_file_path, sep="\t", names=['id', 'utterance'])
vocab_file.drop(columns=['id'], inplace=True)

vocab_file_path8 = "../../datasets/multimodal_datasets/MUStARD/mustard_16000_transcription.txt"
vocab_file8 = pd.read_csv(vocab_file_path8, sep="\t", names=['id', 'utterance'])
vocab_file8.drop(columns=['id'], inplace=True)

vocab_file_path2 = "../../datasets/multimodal_datasets/MELD_formatted/train/train_sent_emo.csv"
vocab_file2 = pd.read_csv(vocab_file_path2, sep=",", usecols=["Utterance"])
vocab_file2.rename(columns={"Utterance": "utterance"}, inplace=True)

vocab_file_path3 = "../../datasets/multimodal_datasets/MELD_formatted/dev/dev_sent_emo.csv"
vocab_file3 = pd.read_csv(vocab_file_path3, sep=",", usecols=["Utterance"])
vocab_file3.rename(columns={"Utterance": "utterance"}, inplace=True)

vocab_file_path4 = "../../datasets/multimodal_datasets/MELD_formatted/test/test_sent_emo.csv"
vocab_file4 = pd.read_csv(vocab_file_path4, sep=",", usecols=["Utterance"])
vocab_file4.rename(columns={"Utterance": "utterance"}, inplace=True)

vocab_file_path5 = "../../datasets/multimodal_datasets/Chalearn/train/gold_and_utts.tsv"
vocab_file5 = pd.read_csv(vocab_file_path5, sep="\t", usecols=["utterance"])

vocab_file_path6 = "../../datasets/multimodal_datasets/Chalearn/val/gold_and_utts.tsv"
vocab_file6 = pd.read_csv(vocab_file_path6, sep="\t", usecols=["utterance"])

vocab_file_path7 = "../../datasets/multimodal_datasets/Chalearn/test/gold_and_utts.tsv"
vocab_file7 = pd.read_csv(vocab_file_path7, sep="\t", usecols=["utterance"])

all_vocab = pd.concat([vocab_file, vocab_file2, vocab_file3, vocab_file4, vocab_file5, vocab_file6,
                       vocab_file7, vocab_file8], axis=0)
# all_vocab = vocab_file
all_vocab = all_vocab.replace(np.nan, "", regex=True)

glove_file = glove.read_glove(save_path)
glove_set = set(glove_file.keys())

# glove_file = pd.read_csv(save_path, sep=" ", usecols=[0], names=['word'], engine='python')

# get data from transcribed df
vset = glove.get_all_vocab_from_transcribed_df(all_vocab)
# find only those words not already in the glove file
small_vset = glove.compare_vocab_with_existing_data(vset, glove_set)
# get subset of glove using this
subset = glove.subset_glove(glove_path, small_vset, vec_len=vec_length)

print(len(small_vset))
print(len(subset))


# appnd to file
glove.append_subset(subset, save_path)


# get the set of all vocab from files
# vset = glove.get_all_vocab(vocab_path)
# get subset of glove using this
# subset = glove.subset_glove(glove_path, vset, vec_len=vec_length)
# save this subset
# glove.save_subset(subset, save_path)

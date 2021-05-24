# test glove subsetting

import tomcat_speech.data_prep.glove_subsetting as glove
import pandas as pd
import numpy as np

# set paths
# vocab_path = "/Volumes/LIvES/multimodal_data/"
glove_path = "../../glove.42B.300d.txt"
save_path = "../../glove.short.300d.punct.chkldi.txt"
short_path = save_path
vec_length = 300

# vocab_file_path = "../../datasets/multimodal_datasets/MUStARD/mustard_utts.tsv"
# vocab_file = pd.read_csv(vocab_file_path, sep="\t", names=['id', 'utterance'])
# vocab_file.drop(columns=['id'], inplace=True)
#
# vocab_file_path8 = "../../datasets/multimodal_datasets/MUStARD/mustard_16000_transcription.txt"
# vocab_file8 = pd.read_csv(vocab_file_path8, sep="\t", names=['id', 'utterance'])
# vocab_file8.drop(columns=['id'], inplace=True)
#
# vocab_file_path9 = "../../datasets/multimodal_datasets/MUStARD/mustard_sphinx.txt"
# vocab_file9 = pd.read_csv(vocab_file_path9, sep="\t", names=['id', 'utterance'])
# vocab_file9.drop(columns=['id'], inplace=True)
#
# vocab_file_path10 = "../../datasets/multimodal_datasets/MUStARD/mustard_google.tsv"
# vocab_file10 = pd.read_csv(vocab_file_path10, sep="\t", usecols=["utterance"])

# all_vocab = pd.concat([vocab_file, vocab_file8, vocab_file9, vocab_file10], axis=0)

# vocab_file_path2 = "../../datasets/multimodal_datasets/MELD_formatted/train/train_sent_emo.csv"
# vocab_file2 = pd.read_csv(vocab_file_path2, sep=",", usecols=["Utterance"])
# vocab_file2.rename(columns={"Utterance": "utterance"}, inplace=True)

# vocab_file_path11 = "../../datasets/multimodal_datasets/MELD_formatted/train/meld_google.tsv"
# vocab_file11 = pd.read_csv(vocab_file_path11, sep="\t", usecols=["Utterance"])
# vocab_file11.rename(columns={"Utterance": "utterance"}, inplace=True)
#
# vocab_file_path12 = "../../datasets/multimodal_datasets/MELD_formatted/train/meld_train_sphinx_16000.txt"
# vocab_file12 = pd.read_csv(vocab_file_path12, sep="\t", names=["id", "utterance"])
# vocab_file12.drop(columns=["id"], inplace=True)
#
# vocab_file_path13 = "../../datasets/multimodal_datasets/MELD_formatted/dev/meld_google.tsv"
# vocab_file13 = pd.read_csv(vocab_file_path13, sep="\t", usecols=["Utterance"])
# vocab_file13.rename(columns={"Utterance": "utterance"}, inplace=True)
#
# vocab_file_path14 = "../../datasets/multimodal_datasets/MELD_formatted/dev/meld_dev_sphinx_16000.txt"
# vocab_file14 = pd.read_csv(vocab_file_path14, sep="\t", names=["id", "utterance"])
# vocab_file14.drop(columns=["id"], inplace=True)
#
# vocab_file_path15 = "../../datasets/multimodal_datasets/MELD_formatted/test/meld_google.tsv"
# vocab_file15 = pd.read_csv(vocab_file_path15, sep="\t", usecols=["Utterance"])
# vocab_file15.rename(columns={"Utterance": "utterance"}, inplace=True)
#
# vocab_file_path16 = "../../datasets/multimodal_datasets/MELD_formatted/test/meld_test_sphinx_16000.txt"
# vocab_file16 = pd.read_csv(vocab_file_path16, sep="\t", names=["id", "utterance"])
# vocab_file16.drop(columns=["id"], inplace=True)
#
# vocab_file_path17 = "../../datasets/multimodal_datasets/MELD_formatted/train/meld_16000_train_transcription.txt"
# vocab_file17 = pd.read_csv(vocab_file_path17, sep="\t", names=["id", "utterance"])
# vocab_file17.drop(columns=["id"], inplace=True)
#
# vocab_file_path18 = "../../datasets/multimodal_datasets/MELD_formatted/dev/meld_16000_dev_transcription.txt"
# vocab_file18 = pd.read_csv(vocab_file_path18, sep="\t", names=["id", "utterance"])
# vocab_file18.drop(columns=["id"], inplace=True)
#
# vocab_file_path19 = "../../datasets/multimodal_datasets/MELD_formatted/test/meld_16000_test_transcription.txt"
# vocab_file19 = pd.read_csv(vocab_file_path19, sep="\t", names=["id", "utterance"])
# vocab_file19.drop(columns=["id"], inplace=True)
#
# all_vocab = pd.concat([vocab_file11, vocab_file12, vocab_file13, vocab_file14, vocab_file15, vocab_file16,
#                        vocab_file17, vocab_file18, vocab_file19], axis=0)
#
# vocab_file_path3 = "../../datasets/multimodal_datasets/MELD_formatted/dev/dev_sent_emo.csv"
# vocab_file3 = pd.read_csv(vocab_file_path3, sep=",", usecols=["Utterance"])
# vocab_file3.rename(columns={"Utterance": "utterance"}, inplace=True)
#
# vocab_file_path4 = "../../datasets/multimodal_datasets/MELD_formatted/test/test_sent_emo.csv"
# vocab_file4 = pd.read_csv(vocab_file_path4, sep=",", usecols=["Utterance"])
# vocab_file4.rename(columns={"Utterance": "utterance"}, inplace=True)
#
# vocab_file_path5 = "../../datasets/multimodal_datasets/Chalearn/train/gold_and_utts.tsv"
# vocab_file5 = pd.read_csv(vocab_file_path5, sep="\t", usecols=["utterance"])
#
# vocab_file_path6 = "../../datasets/multimodal_datasets/Chalearn/val/gold_and_utts.tsv"
# vocab_file6 = pd.read_csv(vocab_file_path6, sep="\t", usecols=["utterance"])
#
# vocab_file_path7 = "../../datasets/multimodal_datasets/Chalearn/test/gold_and_utts.tsv"
# vocab_file7 = pd.read_csv(vocab_file_path7, sep="\t", usecols=["utterance"])
#
# vocab_file_path20 = "../../datasets/multimodal_datasets/Chalearn/train/chalearn_google.tsv"
# vocab_file20 = pd.read_csv(vocab_file_path20, sep="\t", usecols=["utterance"])
#
# vocab_file_path21 = "../../datasets/multimodal_datasets/Chalearn/val/chalearn_google.tsv"
# vocab_file21 = pd.read_csv(vocab_file_path21, sep="\t", usecols=["utterance"])
#
# vocab_file_path22 = "../../datasets/multimodal_datasets/Chalearn/test/chalearn_google.tsv"
# vocab_file22 = pd.read_csv(vocab_file_path22, sep="\t", usecols=["utterance"])
#
# vocab_file_path23 = "../../datasets/multimodal_datasets/Chalearn/train/chalearn_sphinx.tsv"
# vocab_file23 = pd.read_csv(vocab_file_path23, sep="\t", usecols=["utterance"])
#
# vocab_file_path24 = "../../datasets/multimodal_datasets/Chalearn/val/chalearn_sphinx.tsv"
# vocab_file24 = pd.read_csv(vocab_file_path24, sep="\t", usecols=["utterance"])
#
# vocab_file_path25 = "../../datasets/multimodal_datasets/Chalearn/test/chalearn_sphinx.tsv"
# vocab_file25 = pd.read_csv(vocab_file_path25, sep="\t", usecols=["utterance"])

vocab_file_path26 = "../../datasets/multimodal_datasets/Chalearn/train/chalearn_kaldi.tsv"
vocab_file26 = pd.read_csv(vocab_file_path26, sep="\t", usecols=["utterance"])

vocab_file_path27 = "../../datasets/multimodal_datasets/Chalearn/val/chalearn_kaldi.tsv"
vocab_file27 = pd.read_csv(vocab_file_path27, sep="\t", usecols=["utterance"])

vocab_file_path28 = "../../datasets/multimodal_datasets/Chalearn/test/chalearn_kaldi.tsv"
vocab_file28 = pd.read_csv(vocab_file_path28, sep="\t", usecols=["utterance"])

all_vocab = pd.concat([vocab_file26, vocab_file27, vocab_file28], axis=0)

# all_vocab = pd.concat([vocab_file20, vocab_file21, vocab_file22, vocab_file23, vocab_file24, vocab_file25],
#                       axis=0)

# all_vocab = pd.concat([vocab_file, vocab_file2, vocab_file3, vocab_file4, vocab_file5, vocab_file6,
#                        vocab_file7, vocab_file8], axis=0)
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
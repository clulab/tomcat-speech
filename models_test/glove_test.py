# test glove subsetting

import os
import h5py
import pandas as pd

import data_prep.subset_glove as glove

# set paths
vocab_path = "/Volumes/LIvES/multimodal_data/"
glove_path = "../../glove.twitter.27B/glove.twitter.27B.50d.txt"
save_path = "../../glove.shorter.50d.txt"
short_path = save_path
vec_length = 50

# get the set of all vocab from files
vset = glove.get_all_vocab(vocab_path)
# get subset of glove using this
subset = glove.subset_glove(glove_path, vset, vec_len=vec_length)
# save this subset
glove.save_subset(subset, save_path)

#
# # copied from data_prep
# def get_idx(token2idx, token):
#     if token2idx[token]:
#         return token2idx[token]
#     else:
#         return token2idx["<UNK>"]
#
#
# def get_indices(token2idx, tk_series):
#     # print(token2idx)
#     tks = []
#     for tk in tk_series:
#         try:
#             if token2idx[tk]:
#                 tks.append(token2idx[tk])
#             else:
#                 tks.append(token2idx["<UNK>"])
#         except KeyError:
#             tks.append(token2idx["<UNK>"])
#     # print(tks)
#     return tks
#
#
# # get counts of the vocabulary
# words = []
# wd2idx = {}
# vecs = []
# with open(short_path) as shortfile:
#     c = 0
#     for line in shortfile:
#         line = line.strip().split()
#         words.append(line[0])
#         vecs.append(line)
#         wd2idx[line[0]] = c
#         c += 1
#
# print("vecs extracted")
#
# idx2vecs = {}
# for item in vecs:
#     idx2vecs[wd2idx[item[0]]] = item[1:]
#
# print("idx2vecs completed")
#
# # # first attempt. it...works, but isn't ideal
# # # 37 mins for 323 files
# # for dfile in os.listdir(vocab_path):
# #     if dfile.endswith("_IS09_avgd.csv"):
# #         print(dfile)
# #         dname = dfile.split(".")[0]
# #         file_df = pd.read_csv(vocab_path + "/" + dfile)
# #         file_df['wd_idx'] = get_indices(wd2idx, file_df['word'])
# #         file_df['sid'] = dfile.split("_")[0]
# #         file_df['vec'] = get_indices(idx2vecs, file_df['wd_idx'])
# #         file_df.to_csv(vocab_path + "/" + dname + "-final.csv")
#
# # second attempt
#
# for dfile in os.listdir(vocab_path):
#     if dfile.endswith("_IS09_avgd.csv"):
#         all_lines = []
#         dname = dfile.split(".")[0]
#         with open(vocab_path + "/" + dfile) as d:
#             d.readline()
#             for line in d:
#                 line = line.strip().split(",")
#                 wd = line[1].lower()
#                 try:
#                     vec = idx2vecs[wd2idx[line[1]]]
#                 except KeyError:
#                     vec = idx2vecs[wd2idx["<UNK>"]]
#                 intermediate = []
#                 intermediate.extend(line[5:])
#                 intermediate.extend(vec)
#                 intermediate.extend(line[2])
#                 all_lines.append(intermediate)
#         with open(vocab_path + "/" + dname + "-final.csv", 'w') as sfile:
#             for item in all_lines:
#                 sfile.write(",".join(item))
#                 sfile.write("\n")

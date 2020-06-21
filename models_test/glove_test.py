# test glove subsetting

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
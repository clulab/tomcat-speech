# set the paths to your datasets
base_path = "/media/jculnan/One Touch/jculnan/datasets"
# set the path to each potential dataset
# if you don't have one or more of these datasets, that's fine
# just make sure that your 'datasets' list (below) ONLY contains
# datasets that you DO have access to
cdc_path = f"{base_path}/Columbia_deception_corpus"
mosi_path = f"{base_path}/CMU_MOSI"
firstimpr_path = f"{base_path}/FirstImpr"
meld_path = f"{base_path}/MELD_formatted"
mustard_path = f"{base_path}/MUStARD"
ravdess_path = f"{base_path}/RAVDESS_Speech"
asist_path = f"{base_path}/MultiCAT"

# set the path to where you would like your data saved
save_path = "/media/jculnan/One Touch/jculnan/datasets/field_separated_data"

# set the path to your GloVe data
glove_path = "/media/jculnan/One Touch/jculnan/datasets/glove/glove.subset.300d.txt"

# ID the set of acoustic features you wish to include
# options: IS09, IS10, IS11, IS12, IS13
# generally IS13 outperforms the others, and is what we use in our models
# this feature set needs to have been previously extracted
# using 'extract_acoustic_features.py'
feature_set = "IS13"

# set path to pickle file(s) containing lists of ids
# dict of {dataset: string path to pickle files with ids separated by partition}
# this is set to None if you do not have such a file
# this is only needed when a dataset does not come prepartitioned
# and you wish to have specific items in each partition
# currently implemented for RAVDESS
selected_ids_paths = None

# set the transcription type
# this is generally 'gold' unless you have created other transcriptions
# using another ASR system and have files saved using this other asr type
transcription_type = "gold"

# set the embedding type
# this may be 'bert', 'distilbert', 'roberta', 'glove', or 'text'
# if 'text', the text data will NOT be converted to embeddings
# and can be fed directly into a huggingface transformer model
emb_type = "text"  # include text, to be used with bert in the model itself

# select which dataset(s) you want to preprocess
# can be one or more of:
# 'asist', 'cdc', 'firstimpr', 'meld', 'mosi', 'mustard', 'ravdess'
datasets = ["asist"]

# if using First Impressions or MOSI, select the prediction type
# for mosi, can be 'regression' to preserve continuous numbers
# 'classification' for 7-class classification setup usually used
# or 'ternary' for 3-class classification setup
# 'classification' is standard in our models, which can convert
# gold labels to ternary as part of the model itself.
mosi_pred_type = "classification"
# for first impressions, use one of 'max_class' (dominant trait),
# 'binary' (high-low for each trait), or 'ternary' (high-med-low)
# for each trait
# max_class is currently expected by most of our models
firstimpr_pred_type = "max_class"

# set the path to a custom acoustic feature file, if previously generated
# dict of {dataset: string path to files with custom acoustic features}
# otherwise set this to None to use openSMILE features
custom_feats_file = None

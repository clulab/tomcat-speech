
from argparse import Namespace
import os

DEBUG = False  # if true, no saving of files; output in the terminal

# do you want to load pre-saved dataset files?
load_dataset = True

EXPERIMENT_ID = 1

# during training: enter a brief description that will make the experiment easy to identify
# during testing: this is the name of the parent directory for different random seed models saved from an experiment
# EXPERIMENT_DESCRIPTION = "MC_customfeats_distilbert_ClassWeights_batchNormEps1e-2_"
EXPERIMENT_DESCRIPTION = "DELETE_ME_"
# indicate whether this code is being run locally or on the server
USE_SERVER = False

# get this file's path to save a copy
CONFIG_FILE = os.path.abspath(__file__)

# path from which to load pickled data files
load_path = "../../datasets/pickled_data/distilbert_custom_feats"

num_tasks = 5

# set parameters for data prep
glove_path = "../../datasets/glove/glove.subset.300d.txt"

# set dir to save full experiments
exp_save_path = "output/multitask"

# set the acoustic feature set
feature_set = "combined_features_distilbert_dict"

num_feats = 130
if feature_set.lower() == "is13":
    num_feats = 130
elif "combined_features" in feature_set.lower() or "custom" in feature_set.lower():
    num_feats = 10

model_params = Namespace(
    # consistency parameters
    seed=88,
    # trying text only model or not
    text_only=False,
    audio_only=False,
    # overall model parameters
    model="Multitask-baseline",
    num_epochs=5,
    batch_size=100,  # 128,  # 32
    early_stopping_criterion=2,
    num_gru_layers=1,  # 1,  # 3,  # 1,  # 4, 2,
    bidirectional=False,
    use_distilbert=True,  # false if using GloVe, else true
    # input dimension parameters
    text_dim=768,  # text vector length
    short_emb_dim=30,  # length of trainable embeddings vec
    audio_dim=num_feats,
    # text NN
    kernel_1_size=3,
    kernel_2_size=4,
    kernel_3_size=5,
    out_channels=20,
    text_cnn_hidden_dim=100,
    # text_output_dim=30,   # 100,   # 50, 300,
    text_gru_hidden_dim=100,  # 30,  # 50,  # 20
    # acoustic NN
    avgd_acoustic=False,  # set true to use avgd acoustic feat vectors without RNN
    add_avging=True,  # set to true if you want to avg acoustic feature vecs upon input
    acoustic_gru_hidden_dim=100,
    # speaker embeddings
    use_speaker=False,
    num_speakers=13,  # check this number
    speaker_emb_dim=3,
    # gender embeddings
    use_gender=False,
    gender_emb_dim=4,
    # outputs
    output_dim=30,  # output dimensions from last layer of base model
    output_0_dim=2,  # 2 classes in CDC
    output_1_dim=7,  # 7 classes in CMU MOSI (-3 to + 3)
    output_2_dim=5,  # 5 classes in firstimpr
    output_3_dim=7,  # 7 classes in MELD
    output_4_dim=2,  # 2 classes in RAVDESS intensities
    # FC layer parameters
    fc_hidden_dim=30,  # 20,  must match output_dim if final fc layer removed from base model
    final_hidden_dim=50,  # the out size of dset-specific fc1 and input of fc2
    dropout=0.3,  # 0.2, 0.3
    # optimizer parameters
    lr=1e-3,
    beta_1=0.9,
    beta_2=0.999,
    weight_decay=0.0001,
)

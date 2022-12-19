# incorporates elements from https://github.com/marcovzla/discobert/blob/master/config.py

from argparse import Namespace
import os  # incorporates elements from https://github.com/marcovzla/discobert/blob/master/config.py

from argparse import Namespace
import os

DEBUG = False # no saving of files; output in the terminal; first random seed from the list

# do you want to save dataset files?
save_dataset = False

# do you want to load pre-saved dataset files?
load_dataset = True

# set the task
task = "firstimpr"

EXPERIMENT_ID = 2
# during training: enter a brief description that will make the experiment easy to identify
# during testing: this is the name of the parent directory for different random seed models saved from an experiment
# EXPERIMENT_DESCRIPTION = "MMC_25perc-cutoff_15secMax_noClassWeights_IS1010_GaussianNoise_"
# EXPERIMENT_DESCRIPTION = "CHALEARN_KALDI_TEXTONLY_VALF1CHECKED_25perc-cutoff_15secMax_noClassWeights_IS1076_AcHid50_"
EXPERIMENT_DESCRIPTION = f"{task}_ReworkedCodeTest_"
# indicate whether this code is being run locally or on the server
USE_SERVER = True

# get this file's path to save a copy
CONFIG_FILE = os.path.abspath(__file__)

num_tasks = 5

# set parameters for data prep
glove_path = "/home/tomcat/multimodal_data/glove.subset.300d.txt"
#glove_path = "../../datasets/glove/glove.subset.300d.txt"
datasets = ["meld"] #"toymeld"
datasets_num = {"meld":7, "firstimpr":5, "ravdess":2, "mosi":7, "cdc":2, "toymeld": 7}

if USE_SERVER:
    load_path = "/home/tomcat/multimodal_data"
else:
    # path from which to load pickled data files
    #load_path = "../../datasets/pickled_data/distilbert_custom_feats"
    None

# if you are loading a pretrained model, put the path to it here
pretrained_model = None

# set dir to save full experiments
exp_save_path = f"output/single_task/{task}"

# set the acoustic feature set
#feature_set = "combined_features_distilbert_dict"
#feature_set = "IS13_glove_dict"
feature_set = "IS13_distilbert_dict" #revised on 11/30/22

num_feats = 130
if feature_set.lower() == "is13":
    num_feats = 130
elif "combined_features" in feature_set.lower() or "custom" in feature_set.lower():
    num_feats = 10

model_params = Namespace(
        # use gradnorm for loss normalization
        use_gradnorm=False,
        # decide whether to use early, intermediate, or late fusion
        fusion_type="early",  # int, late, early
        # consistency parameters
        seed=88,  # 1007
        # trying text only model or not
        text_only=False,
        audio_only=False,
        # overall model parameters
        #model=f"Single_task_{task}", # - => _ 10/24/22
        model=f"text_only", #added 10/24/22
        num_epochs=2, #revised from 200 to 1 for testing 12/15/22
        batch_size=100,  # 128,  # 32
        early_stopping_criterion=50,
        num_gru_layers=1,  # 1,  # 3,  # 1,  # 4, 2, #2=>1 for testing 12/15/22
        bidirectional=False,
        use_distilbert=True,
        # input dimension parameters
        text_dim=768,  # text vector length # 768 for bert/distilbert, 300 for glove
        short_emb_dim=30,  # length of trainable embeddings vec
        audio_dim=num_feats,  # audio vector length
        # text NN
        kernel_1_size=3,
        kernel_2_size=4,
        kernel_3_size=5,
        out_channels=20,
        text_cnn_hidden_dim=100,
        # text_output_dim=30,   # 100,   # 50, 300,
        text_gru_hidden_dim=20,  # 30,  # 50,  # 20 #300=>20 for testing 12/15/22
        # acoustic NN
        avgd_acoustic=False,  # set true to use avgd acoustic feat vectors without RNN
        add_avging=True,  # set to true if you want to avg acoustic feature vecs upon input
        acoustic_gru_hidden_dim=100,
        # use classweights
        use_clsswts = False, #if true, "meld" distribution, not assist distribution
        # speaker embeddings
        use_speaker=False,
        num_speakers=13,  # check this number
        speaker_emb_dim=3,
        # gender embeddings
        use_gender=False,
        gender_emb_dim=4,
        # spectrogram features
        use_spec=False,
        spec_only=False, # not text, audio
        # set whether to have a single loss function
        single_loss = False,
        # whether to use loss multiplier by dataset size
        loss_multiplier = False, # output
        output_dim=10,  # output dimensions from last layer of base model #100=>10 for testing 12/15/22
        output_0_dim=datasets_num[datasets[0]], # set to 7 because ['meld'] has 7 classes
        #print(output_0_dim), # added 12/05/22
        #output_0_dim=5,  # output vec for first task
        output_1_dim=0,  # output vec for second task
        output_2_dim=0,  # output vec for third task
        output_3_dim=0,
        output_4_dim=0,
        # FC layer parameters
        num_fc_layers=1,  # 1,  # 2,
        fc_hidden_dim=10,  # 20,  must match output_dim if final fc layer removed from base model #100=>10 for testing 12/15/22
        final_hidden_dim=20,  # the out size of dset-specific fc1 and input of fc2 #200=>20 for testing 12/15/22
        dropout=0.4,  # 0.2, 0.3
        # optimizer parameters
        lr=1e-6,
        beta_1=0.9,
        beta_2=0.999,
        weight_decay=0.0001,
)

# incorporates elements from https://github.com/marcovzla/discobert/blob/master/config.py

from argparse import Namespace
import os


DEBUG = True # no saving of files; output in the terminal; first random seed from the list
EXPERIMENT_ID = 1
# during training: enter a brief description that will make the experiment easy to identify
# during testing: this is the name of the parent directory for different random seed models saved from an experiment
EXPERIMENT_DESCRIPTION = "meld-plus-mustard_initial"
# get this file's path to save a copy
CONFIG_FILE = os.path.abspath(__file__)

model_type = "MULTITASK_TEST"

# set parameters for data prep
# glove_file = "/work/bsharp/glove.short.300d.punct.txt"
# glove_file = "/data/nlp/corpora/glove/glove.840B.300d.no_proc_header.txt"
glove_file = "../../glove.short.300d.punct.txt"
# glove_file = "../../glove.42B.300d.txt"

mustard_path = "../../datasets/multimodal_datasets/MUStARD"
meld_path = "../../datasets/multimodal_datasets/MELD_formatted"
# meld_path = "../../datasets/multimodal_datasets/MELD_five_dialogues"
ravdess_path = "../../datasets/multimodal_datasets/RAVDESS_speech"

# set dir to save full experiments
exp_save_path = "output/multitask/experiments"

data_type = "multitask"
fusion_type = "early"

acoustic_columns = ['pcm_loudness_sma', 'F0finEnv_sma', 'voicingFinalUnclipped_sma', 'jitterLocal_sma',
                              'shimmerLocal_sma', 'pcm_loudness_sma_de', 'F0finEnv_sma_de',
                              'voicingFinalUnclipped_sma_de', 'jitterLocal_sma_de', 'shimmerLocal_sma_de']

model_params = Namespace(
    # consistency parameters
    seed=88,  # 1007
    # trying text only model or not
    text_only=False,
    # overall model parameters
    model="Multitask-mustard",
    num_epochs=2,
    batch_size=10,  # 128,  # 32
    early_stopping_criteria=2,
    num_gru_layers=2,  # 1,   # 3,  # 1,  # 4, 2,
    bidirectional=False,
    # input dimension parameters
    text_dim=300,  # text vector length
    short_emb_dim=30,  # length of trainable embeddings vec
    audio_dim=10,  # 78,  # 76,  # 79,  # 10 # audio vector length
    # audio_dim=10,
    # text NN
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
    use_gender=True,
    gender_emb_dim=4,
    # outputs
    output_dim=20,  # output dimensions from last layer of base model
    output_0_dim=1,
    output_1_dim=7,  # 7,  # length of output vector
    output_2_dim=0,  # 3,    # length of second task output vec
    output_3_dim=0,
    # FC layer parameters
    num_fc_layers=1,  # 1,  # 2,
    fc_hidden_dim=100,  # 20,
    dropout=0.4,  # 0.2
    # optimizer parameters
    lrs=[1e-2, 1e-3, 1e-4],
    beta_1=0.9,
    beta_2=0.999,
    weight_decay=[0.0001, 0.00001, 0],
)
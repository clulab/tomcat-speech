# incorporates elements from https://github.com/marcovzla/discobert/blob/master/config.py

from argparse import Namespace
import os

DEBUG = (
    False  # no saving of files; output in the terminal; first random seed from the list
)

# do you want to save dataset files?
save_dataset = False

# do you want to load pre-saved dataset files?
load_dataset = True


EXPERIMENT_ID = 1
# during training: enter a brief description that will make the experiment easy to identify
# during testing: this is the name of the parent directory for different random seed models saved from an experiment
EXPERIMENT_DESCRIPTION = "MC_Testing_gridsearch_code_"
# indicate whether this code is being run locally or on the server
USE_SERVER = True  # 01/10/23

# whether you are starting with a pretrained model that you update during training
trained_model = None  # or path to model file

# get this file's path to save a copy
CONFIG_FILE = os.path.abspath(__file__)

num_tasks = 5

# set parameters for data prep
glove_path = "/home/tomcat/multimodal_data/glove.subset.300d.txt"  # 01/10/23

if USE_SERVER:
    load_path = "/home/tomcat/multimodal_data"  # 01/10/23
else:
    # path from which to load pickled data files
    # load_path = "../../datasets/pickled_data/IS13_glove_GOLD"
    load_path = None

# set dir to save full experiments
exp_save_path = "output/multitask"

# set type of predictions to make for chalearn
chalearn_predtype = "max_class"

# set the acoustic feature set
feature_set = "IS13_distilbert_dict"  # 01/10/23

datasets = ["meld"]  # added 01/10/23

saved_model = None

num_feats = 130
if feature_set.lower() == "is13":
    num_feats = 130
elif "combined_features" in feature_set.lower() or "custom" in feature_set.lower():
    num_feats = 10

model_params = Namespace(
    # use gradnorm for loss normalization
    use_gradnorm=False,
    # whether to use data sampler
    use_sampler=False,
    # whether to use class weights 01/10/23
    use_clsswts=False,
    # decide whether to use early, intermediate, or late fusion
    fusion_type="int",  # int, late, early
    # consistency parameters
    seed=88,  # 1007
    # trying text only model or not
    text_only=False,
    audio_only=False,
    use_spec=False,
    spec_only=False,
    # overall model parameters
    model="Multitask",
    num_epochs=200,
    batch_size=[100],  # 128,  # 32
    early_stopping_criterion=5,
    num_gru_layers=2,  # 1,  # 3,  # 1,  # 4, 2,
    bidirectional=True,  # 01/10/23
    use_distilbert=False,
    # set whether to have a single loss function
    single_loss=False,
    # whether to use loss multiplier by dataset size 01/10/23
    loss_multiplier=False,
    # input dimension parameters
    text_dim=300,  # text vector length # 768 for bert/distilbert, 300 for glove
    short_emb_dim=30,  # length of trainable embeddings vec
    audio_dim=num_feats,  # audio vector length
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
    output_dim=[100],  # output dimensions from last layer of base model
    output_0_dim=7,  # output vec for first task 2 7 5 7 2
    output_1_dim=0,  # output vec for second task, 01/10/23
    output_2_dim=0,  # output vec for third task
    output_3_dim=0,
    output_4_dim=0,
    # FC layer parameters
    num_fc_layers=1,  # 1,  # 2,
    fc_hidden_dim=100,  # 20,  must match output_dim if final fc layer removed from base model
    final_hidden_dim=[50, 20],  # the out size of dset-specific fc1 and input of fc2
    dropout=[0.2, 0.3],  # 0.2, 0.3
    # optimizer parameters
    lr=[1e-3, 1e-4],
    beta_1=0.9,
    beta_2=0.999,
    weight_decay=0.0001,
)

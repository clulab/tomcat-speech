# incorporates elements from https://github.com/marcovzla/discobert/blob/master/config.py

from argparse import Namespace
import os

DEBUG = (
    False  # no saving of files; output in the terminal; first random seed from the list
)

# what number experiment is this?
# can leave it at 1 or increment if you have multiple
#   experiments with the same description from the same date
EXPERIMENT_ID = 1
# during training: enter a brief description that will make the experiment easy to identify
# during testing: this is the name of the parent directory for different random seed models saved from an experiment
EXPERIMENT_DESCRIPTION = "Test_finetuning_"

# indicate whether this code is being run locally or on the server
USE_SERVER = False

# get this file's path to save a copy
# this does not need to be changed
CONFIG_FILE = os.path.abspath(__file__)

# how many tasks are you running over?
# it's not critical to change this number unless you're running
#   a single dataset over multiple tasks (e.g. asist data)
num_tasks = 3

# set parameters for data prep
# where is your GloVe file located?
glove_path = "/media/jculnan/backup/jculnan/datasets/glove/glove.subset.300d.txt"

# where is the preprocessed pickle data saved?
if USE_SERVER:
    load_path = "/data/nlp/corpora/MM/pickled_data/distilbert_custom_feats"
else:
    load_path = "/media/jculnan/backup/jculnan/datasets/pickled_data"

# set directory to save full experiments
exp_save_path = "output/multitask"

# set type of predictions to make for chalearn
# this is currently not implemented as we only use
#   dominant trait prediction as the personality task
# todo: reincorporate this implementation?
chalearn_predtype = "max_class"

# set the acoustic and text feature sets
# the first item should be the acoustic feature set
# the second item should be the text embedding type (distilbert, bert, glove)
# the third item is whether to use data in list or dict form
# currently, list form is being phased out, so use dict
# if these items are not set correctly,
# the data may not be loaded properly
feature_set = "IS13_glove_dict"

# give a list of the datasets to be used
# todo: link datasets to output classes
# datasets = ["mosi", "ravdess"]
datasets = ["asist"]

# the number of classes per dataset
dset2classes = {"meld": 7, "mosi": 3, "ravdess": 2, "firstimpr": 5}

# whether to fine-tune on a saved model
# if fine-tuning a saved model, set this to the string path of the model
# else, set it to None
saved_model = "output/multitask/1_Testing_gridsearch_2022-08-16/LR0.001_BATCH100_NUMLYR2_SHORTEMB30_INT-OUTPUT100_DROPOUT0.2_FC-FINALDIM20/Testing_gridsearch_code_.pt"

# the number of acoustic features to use
# 130 is the number of features in IS13 set
# 10 is the number of features in the custom set
num_feats = 130
if "is13" in feature_set.lower():
    num_feats = 130
elif "combined_features" in feature_set.lower() or "custom" in feature_set.lower():
    num_feats = 10

# a namespace object containing the parameters that you might need to alter for training
model_params = Namespace(
    # these parameters are separated into two sections
    # in the first section are parameters that are currently used
    # in the second section are parameters that are not currently used
    #   the latter may either be reincorporated into the network(s)
    #   in the future or may be removed from this Namespace
    # --------------------------------------------------
    # set the random seed; this seed is used by torch and random functions
    seed=88,  # 1007
    # overall model selection
    # --------------------------------------------------
    # 'model' is used to select an overall model during model selection
    # this is in select_model within train_and_test_utils.py
    # options: acoustic_shared, text_shared, duplicate_input, text_only, multitask
    # other options may be added in the future for more model types
    model="Multitask",
    # data preparation parameters
    # --------------------------------------------------
    # whether we are using bert-based dense embeddings
    # if this is true, we use distilbert or bert embeddings
    # if this is false, we use GloVe embeddings
    use_distilbert=False,
    # whether to use data sampler
    # currently we use oversampling only
    # when set to true, we use an instance of class oversampler
    # from tomcatspeech.data_prep.samplers
    use_sampler=False,
    # loss function parameters
    # --------------------------------------------------
    # whether to use class weights
    # if true, these class weights are added to the loss function
    #   at its instantiation (in training scripts, e.g. train_multitask.py)
    use_clsswts=False,
    # set whether to have a single loss function
    # if true, we have a single loss function operating over all tasks
    # if false, we have a separate loss function for each task
    # generally speaking, it is better to set this to false
    single_loss=False,
    # whether to use loss multiplier by dataset size
    # if we set this to true, the loss of each task is multiplied by
    #   a number representing the proportion of overall data for that task
    # if we set it to false, the losses are added without adjusting for
    #   the number of data points per task
    # this is completed when loading the data in the train/finetune scripts
    loss_multiplier=False,
    # optimizer parameters
    # --------------------------------------------------
    # learning rate
    # with multiple tasks, these learning rates tend to be pretty low
    # (usually 1e-3 -- 1e-5)
    lr=1e-4,
    # hyperparameters for adam optimizer -- usually not needed to alter these
    beta_1=0.9,
    beta_2=0.999,
    weight_decay=0.0001,
    # parameters for model base selection
    # --------------------------------------------------
    # decide whether to use early, intermediate, or late fusion
    # this allows the multitask model to select one of three different bases
    # when the model is instantiated (used by model code in multimodal_models.py)
    fusion_type="int",  # int, late, early
    # whether we are including the spectrogram modality of audio here
    # if true, we use spectrograms in addition to other modalities present
    # if false, we do not use spectrograms
    use_spec=False,
    # if one of these is true, the text-only or audio-only model base should be selected
    # these flags are used directly by the model (e.g. in multimodal_models.py)
    #   during instantiation
    # if text only, the model should only include text modality
    text_only=False,
    # if audio only, the model should only include audio modality
    audio_only=False,
    # whether we have an audio spectrogram model only
    # if true, audio spectrograms are the only modality we use
    # if false, we do not use only spectrogram data
    spec_only=False,
    # whether to average acoustic features after reading them in
    # if true, the acoustic features are averaged and an audio FFNN is used
    # if false, the acoustic features are not averaged and an audio LSTM is used
    # this defaults to true, as it's much faster but performance is comparable
    add_avging=True,  # set to true if you want to avg acoustic feature vecs upon input
    # parameters for model training
    # --------------------------------------------------
    # the maximum number of epochs that a model can run
    num_epochs=100,
    # the minibatch size
    batch_size=100,  # 128,  # 32
    # how many epochs the model will run after updating
    early_stopping_criterion=20,
    # parameters for model architecture
    # --------------------------------------------------
    # number of classes for each of the tasks of interest
    # each one is either the number of classes, or 0 if there is no task
    # todo: automate this based on datasets
    output_0_dim=5,  # number of classes in the first task
    output_1_dim=7,  # number of classes in the second task
    output_2_dim=3,  # number of classes in the third task
    output_3_dim=0,  # number of classes in the fourth task
    output_4_dim=0,  # number of classes in the fifth task
    # number of layers in the recurrent portion of our model
    # this has actually changed from gru to lstm
    num_gru_layers=2,  # 1,  # 3,  # 1,  # 4, 2,
    # whether the recurrent portion of the model is bidirectional
    bidirectional=True,
    # input dimension parameters
    text_dim=300,  # text vector length # 768 for bert/distilbert, 300 for glove
    short_emb_dim=30,  # length of trainable embeddings vec
    # how long is each audio input -- set by the number of acoustic features above
    audio_dim=num_feats,  # audio vector length
    # hyperparameter for text LSTM
    # the size of the hidden dimension between LSTM layers
    text_gru_hidden_dim=100,  # 30,  # 50,  # 20
    # output dimensions for model
    output_dim=100,  # output dimensions from last layer of base model
    # number of fully connected layers after concatenation of modalities
    # this number must either be 1 or 2
    num_fc_layers=1,  # 1,  # 2,
    # the output dimension of the fully connected layer(s)
    fc_hidden_dim=100,  # 20,  must match output_dim if final fc layer removed from base model
    final_hidden_dim=20,  # the out size of dset-specific fc1 and input of fc2
    # the dropout applied to layers of the NN model
    # portions of this model have a separate dropout specified
    # it may be beneficial to add multiple dropout parameters here
    # so that each may be tuned
    dropout=0.2,  # 0.2, 0.3
    # parameters that are only used with specific architecture
    # --------------------------------------------------
    # hyperparameters for a text CNN
    # not used unless you are using a CNN text base
    # most of the time, these parameters aren't needed
    kernel_1_size=3,  # first kernel size with 3 convolutional filters
    kernel_2_size=4,  # second kernel size with 3 convolutional filters
    kernel_3_size=5,  # third kernel size with 3 convolutional filters
    out_channels=20,  # number of output channels for text CNN
    text_cnn_hidden_dim=100,  # hidden dimension for text CNN
    # the hidden dimension size for acoustic RNN layers
    # if add_avging is true, this number isn't used,
    #   so it's usually unused
    acoustic_gru_hidden_dim=100,
    # currently unused parameters
    # --------------------------------------------------
    # whether to use gradnorm for loss normalization
    # this has been removed from the code, so don't worry about this flag
    # if you want to reimplement this, use the function:
    #   multitask_train_and_predict_with_gradnorm from training_and_evaluation_functions.py
    use_gradnorm=False,
    # whether the input features are already averaged when loaded
    # if true, the features read in are already averaged (vector per input item)
    # if false, the features read in are not averaged (tensor per input item)
    avgd_acoustic=False,  # set true to use avgd acoustic feat vectors without RNN
    # speaker embeddings
    # if use_speaker is true, speaker-specific embeddings are used
    # if use_speaker is false, no speaker-specific embeddings are used
    # this hasn't been used for most of the history of this project
    # if we want to reincorporate it into the networks, they will need updating
    use_speaker=False,
    # this is the total number of speakers; if speaker_vectors are used,
    #   this feature should be removed and calculated automatically
    num_speakers=13,
    # the length of the trainable speaker embedding, used if use_speaker == True
    speaker_emb_dim=3,
    # whether to include speaker gender embeddings
    # can only be usefully used if speaker gender is known and included with
    #   saved data that is loaded -- this is not included in the asist data
    #   so it is not currently important to include
    # if use_gender is True, we have a trainable speaker gender embedding
    # if  use_gender is False, we do not include this trainable embedding
    use_gender=False,
    # the length of the trainable gender embedding, if use_speaker == True
    gender_emb_dim=4,
)

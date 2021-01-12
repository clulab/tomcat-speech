# incorporates elements from https://github.com/marcovzla/discobert/blob/master/config.py

from argparse import Namespace
import os

# DEBUG = True # no saving of files; output in the terminal; first random seed from the list

# do you want to save dataset files?
save_dataset = False

# do you want to load pre-saved dataset files?
load_dataset = False


EXPERIMENT_ID = 1
# during training: enter a brief description that will make the experiment easy to identify
# during testing: this is the name of the parent directory for different random seed models saved from an experiment
# EXPERIMENT_DESCRIPTION = "meld-mustard-chalearn_singleOptimizer_IS10-76feats_finalFC-dropout-removed_2lyr-in-dset-specific-output_"
# EXPERIMENT_DESCRIPTION = "GRADNORM_MMC_25perc-cutoff_15secMax_noClassWeights_IS1010_"
# EXPERIMENT_DESCRIPTION = "MMC_25perc-cutoff_15secMax_noClassWeights_IS1010_GaussianNoise_"
EXPERIMENT_DESCRIPTION = "MMC_SPHINX_25perc-cutoff_15secMax_noClassWeights_IS1076_"
# indicate whether this code is being run locally or on the server
USE_SERVER = False

# get this file's path to save a copy
CONFIG_FILE = os.path.abspath(__file__)

model_type = "MULTITASK"

# set parameters for data prep
# glove_file = "/work/bsharp/glove.short.300d.punct.txt"
# glove_file = "/data/nlp/corpora/glove/glove.840B.300d.no_proc_header.txt"
glove_file = "../../glove.short.300d.punct.txt"
# glove_file = "../../glove.42B.300d.txt"

if USE_SERVER:
    mustard_path = "/data/nlp/corpora/MM/MUStARD"
    meld_path = "/data/nlp/corpora/MM/MELD_formatted"
    chalearn_path = "/data/nlp/corpora/MM/Chalearn"
else:
    mustard_path = "../../datasets/multimodal_datasets/MUStARD"
    meld_path = "../../datasets/multimodal_datasets/MELD_formatted"
    # meld_path = "../../datasets/multimodal_datasets/MELD_five_dialogues"
    chalearn_path = "../../datasets/multimodal_datasets/Chalearn"

    # meld_path = "../../datasets/multimodal_datasets/MELD_five_dialogues"
    # meld_path = "../../datasets/multimodal_datasets/MELD_five_utterances"
    # ravdess_path = "../../datasets/multimodal_datasets/RAVDESS_speech"

# set dir to save full experiments
exp_save_path = "output/multitask/sphinx_comparisons"

data_type = "multitask"
fusion_type = "early"

#set type of predictions to make for chalearn
chalearn_predtype = "max_class"

# small set
# acoustic_columns = ['pcm_loudness_sma', 'F0finEnv_sma', 'voicingFinalUnclipped_sma', 'jitterLocal_sma',
#                               'shimmerLocal_sma', 'pcm_loudness_sma_de', 'F0finEnv_sma_de',
#                               'voicingFinalUnclipped_sma_de', 'jitterLocal_sma_de', 'shimmerLocal_sma_de']
# large set
acoustic_columns = ['pcm_loudness_sma', 'mfcc_sma[0]', 'mfcc_sma[1]', 'mfcc_sma[2]', 'mfcc_sma[3]',
                    'mfcc_sma[4]', 'mfcc_sma[5]', 'mfcc_sma[6]', 'mfcc_sma[7]', 'mfcc_sma[8]',
                    'mfcc_sma[9]', 'mfcc_sma[10]', 'mfcc_sma[11]', 'mfcc_sma[12]', 'mfcc_sma[13]',
                    'mfcc_sma[14]', 'logMelFreqBand_sma[0]', 'logMelFreqBand_sma[1]',
                    'logMelFreqBand_sma[2]', 'logMelFreqBand_sma[3]', 'logMelFreqBand_sma[4]',
                    'logMelFreqBand_sma[5]', 'logMelFreqBand_sma[6]', 'logMelFreqBand_sma[7]',
                    'lspFreq_sma[0]', 'lspFreq_sma[1]', 'lspFreq_sma[2]', 'lspFreq_sma[3]',
                    'lspFreq_sma[4]', 'lspFreq_sma[5]', 'lspFreq_sma[6]', 'lspFreq_sma[7]',
                    'F0finEnv_sma', 'voicingFinalUnclipped_sma', 'F0final_sma', 'jitterLocal_sma',
                    'jitterDDP_sma', 'shimmerLocal_sma', 'pcm_loudness_sma_de', 'mfcc_sma_de[0]',
                    'mfcc_sma_de[1]', 'mfcc_sma_de[2]', 'mfcc_sma_de[3]', 'mfcc_sma_de[4]',
                    'mfcc_sma_de[5]', 'mfcc_sma_de[6]', 'mfcc_sma_de[7]', 'mfcc_sma_de[8]',
                    'mfcc_sma_de[9]', 'mfcc_sma_de[10]', 'mfcc_sma_de[11]', 'mfcc_sma_de[12]',
                    'mfcc_sma_de[13]', 'mfcc_sma_de[14]', 'logMelFreqBand_sma_de[0]',
                    'logMelFreqBand_sma_de[1]', 'logMelFreqBand_sma_de[2]', 'logMelFreqBand_sma_de[3]',
                    'logMelFreqBand_sma_de[4]', 'logMelFreqBand_sma_de[5]', 'logMelFreqBand_sma_de[6]',
                    'logMelFreqBand_sma_de[7]', 'lspFreq_sma_de[0]', 'lspFreq_sma_de[1]',
                    'lspFreq_sma_de[2]', 'lspFreq_sma_de[3]', 'lspFreq_sma_de[4]', 'lspFreq_sma_de[5]',
                    'lspFreq_sma_de[6]', 'lspFreq_sma_de[7]', 'F0finEnv_sma_de',
                    'voicingFinalUnclipped_sma_de', 'F0final_sma_de', 'jitterLocal_sma_de',
                    'jitterDDP_sma_de', 'shimmerLocal_sma_de']

model_params = Namespace(
    # use gradnorm for loss normalization
    use_gradnorm=False,
    # consistency parameters
    seed=88,  # 1007
    # trying text only model or not
    text_only=False,
    # overall model parameters
    model="Multitask-mustard",
    num_epochs=100,
    batch_size=[100],  # 128,  # 32
    early_stopping_criteria=100,
    num_gru_layers=[2],  # 1,  # 3,  # 1,  # 4, 2,
    bidirectional=False,
    # input dimension parameters
    text_dim=300,  # text vector length
    short_emb_dim=[30],  # length of trainable embeddings vec
    audio_dim=len(acoustic_columns),  # 78,  # 76,  # 79,  # 10 # audio vector length
    # audio_dim=10,
    # text NN
    kernel_1_size=3,
    kernel_2_size=4,
    kernel_3_size=5,
    out_channels=20,
    text_cnn_hidden_dim=100,
    # text_output_dim=30,   # 100,   # 50, 300,
    text_gru_hidden_dim=[100],  # 30,  # 50,  # 20
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
    output_dim=[100],  # output dimensions from last layer of base model
    output_0_dim=2,  # output vec for first task
    output_1_dim=7,  # 7,  # output vec for second task
    output_2_dim=5,  # 3,    # output vec for third task
    output_3_dim=0,
    # FC layer parameters
    num_fc_layers=1,  # 1,  # 2,
    fc_hidden_dim=100,  # 20,  must match output_dim if final fc layer removed from base model
    final_hidden_dim=50, # the out size of dset-specific fc1 and input of fc2
    dropout=[0.4],  # 0.2
    # optimizer parameters
    lrs=[1e-3, 1e-4],
    beta_1=0.9,
    beta_2=0.999,
    weight_decay=0.0001,
)

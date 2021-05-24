# incorporates elements from https://github.com/marcovzla/discobert/blob/master/config.py

from argparse import Namespace
import os

# DEBUG = True # no saving of files; output in the terminal; first random seed from the list

# do you want to save dataset files?
save_dataset = False

# do you want to load pre-saved dataset files?
load_dataset = False
load_path = "IS13_KALDI_HIGH_MED_LOW_EQUAL"

EXPERIMENT_ID = 3
# this is the name of the parent directory for different models saved from an experiment
# EXPERIMENT_DESCRIPTION = "Chalearn-high-low_25perc-cutoff_15secMax_noClassWeights_IS1376_"
EXPERIMENT_DESCRIPTION = "DELETE"
# indicate whether this code is being run locally or on the server
USE_SERVER = False

# get this file's path to save a copy
CONFIG_FILE = os.path.abspath(__file__)

model_type = "chalearn_high-low"

# set parameters for data prep
# glove_file = "/data/nlp/corpora/glove/glove.840B.300d.no_proc_header.txt"
glove_file = "../../glove.short.300d.punct.txt"
# glove_file = "../../glove.42B.300d.txt"

if USE_SERVER:
    chalearn_path = "/data/nlp/corpora/MM/Chalearn"
else:
    chalearn_path = "../../datasets/multimodal_datasets/Chalearn"

# set dir to save full experiments
exp_save_path = "output/multitask/chalearn-experiments"

data_type = "multitask"
fusion_type = "early"

#set type of predictions to make for chalearn
chalearn_predtype = "high-med-low" #"high-low" # high-med-low # max_class

# set the acoustic feature set
feature_set = "IS13"

# IS13 FULL SET
acoustic_columns = ['F0final_sma','voicingFinalUnclipped_sma','jitterLocal_sma','jitterDDP_sma','shimmerLocal_sma',
                    'logHNR_sma','audspec_lengthL1norm_sma','audspecRasta_lengthL1norm_sma','pcm_RMSenergy_sma',
                    'pcm_zcr_sma','audSpec_Rfilt_sma[0]','audSpec_Rfilt_sma[1]','audSpec_Rfilt_sma[2]',
                    'audSpec_Rfilt_sma[3]','audSpec_Rfilt_sma[4]','audSpec_Rfilt_sma[5]','audSpec_Rfilt_sma[6]',
                    'audSpec_Rfilt_sma[7]','audSpec_Rfilt_sma[8]','audSpec_Rfilt_sma[9]','audSpec_Rfilt_sma[10]',
                    'audSpec_Rfilt_sma[11]','audSpec_Rfilt_sma[12]','audSpec_Rfilt_sma[13]','audSpec_Rfilt_sma[14]',
                    'audSpec_Rfilt_sma[15]','audSpec_Rfilt_sma[16]','audSpec_Rfilt_sma[17]','audSpec_Rfilt_sma[18]',
                    'audSpec_Rfilt_sma[19]','audSpec_Rfilt_sma[20]','audSpec_Rfilt_sma[21]','audSpec_Rfilt_sma[22]',
                    'audSpec_Rfilt_sma[23]','audSpec_Rfilt_sma[24]','audSpec_Rfilt_sma[25]',
                    'pcm_fftMag_fband250-650_sma','pcm_fftMag_fband1000-4000_sma','pcm_fftMag_spectralRollOff25.0_sma',
                    'pcm_fftMag_spectralRollOff50.0_sma','pcm_fftMag_spectralRollOff75.0_sma',
                    'pcm_fftMag_spectralRollOff90.0_sma','pcm_fftMag_spectralFlux_sma',
                    'pcm_fftMag_spectralCentroid_sma','pcm_fftMag_spectralEntropy_sma',
                    'pcm_fftMag_spectralVariance_sma','pcm_fftMag_spectralSkewness_sma',
                    'pcm_fftMag_spectralKurtosis_sma','pcm_fftMag_spectralSlope_sma','pcm_fftMag_psySharpness_sma',
                    'pcm_fftMag_spectralHarmonicity_sma','mfcc_sma[1]','mfcc_sma[2]','mfcc_sma[3]','mfcc_sma[4]',
                    'mfcc_sma[5]','mfcc_sma[6]','mfcc_sma[7]','mfcc_sma[8]','mfcc_sma[9]','mfcc_sma[10]',
                    'mfcc_sma[11]','mfcc_sma[12]','mfcc_sma[13]','mfcc_sma[14]','F0final_sma_de',
                    'voicingFinalUnclipped_sma_de','jitterLocal_sma_de','jitterDDP_sma_de','shimmerLocal_sma_de',
                    'logHNR_sma_de','audspec_lengthL1norm_sma_de','audspecRasta_lengthL1norm_sma_de',
                    'pcm_RMSenergy_sma_de','pcm_zcr_sma_de','audSpec_Rfilt_sma_de[0]','audSpec_Rfilt_sma_de[1]',
                    'audSpec_Rfilt_sma_de[2]','audSpec_Rfilt_sma_de[3]','audSpec_Rfilt_sma_de[4]',
                    'audSpec_Rfilt_sma_de[5]','audSpec_Rfilt_sma_de[6]','audSpec_Rfilt_sma_de[7]',
                    'audSpec_Rfilt_sma_de[8]','audSpec_Rfilt_sma_de[9]','audSpec_Rfilt_sma_de[10]',
                    'audSpec_Rfilt_sma_de[11]','audSpec_Rfilt_sma_de[12]','audSpec_Rfilt_sma_de[13]',
                    'audSpec_Rfilt_sma_de[14]','audSpec_Rfilt_sma_de[15]','audSpec_Rfilt_sma_de[16]',
                    'audSpec_Rfilt_sma_de[17]','audSpec_Rfilt_sma_de[18]','audSpec_Rfilt_sma_de[19]',
                    'audSpec_Rfilt_sma_de[20]','audSpec_Rfilt_sma_de[21]','audSpec_Rfilt_sma_de[22]',
                    'audSpec_Rfilt_sma_de[23]','audSpec_Rfilt_sma_de[24]','audSpec_Rfilt_sma_de[25]',
                    'pcm_fftMag_fband250-650_sma_de','pcm_fftMag_fband1000-4000_sma_de',
                    'pcm_fftMag_spectralRollOff25.0_sma_de','pcm_fftMag_spectralRollOff50.0_sma_de',
                    'pcm_fftMag_spectralRollOff75.0_sma_de','pcm_fftMag_spectralRollOff90.0_sma_de',
                    'pcm_fftMag_spectralFlux_sma_de','pcm_fftMag_spectralCentroid_sma_de',
                    'pcm_fftMag_spectralEntropy_sma_de','pcm_fftMag_spectralVariance_sma_de',
                    'pcm_fftMag_spectralSkewness_sma_de','pcm_fftMag_spectralKurtosis_sma_de',
                    'pcm_fftMag_spectralSlope_sma_de','pcm_fftMag_psySharpness_sma_de',
                    'pcm_fftMag_spectralHarmonicity_sma_de','mfcc_sma_de[1]','mfcc_sma_de[2]','mfcc_sma_de[3]',
                    'mfcc_sma_de[4]','mfcc_sma_de[5]','mfcc_sma_de[6]','mfcc_sma_de[7]','mfcc_sma_de[8]',
                    'mfcc_sma_de[9]','mfcc_sma_de[10]','mfcc_sma_de[11]','mfcc_sma_de[12]','mfcc_sma_de[13]',
                    'mfcc_sma_de[14]']


model_params = Namespace(
    # use gradnorm for loss normalization
    use_gradnorm=True,
    # consistency parameters
    seed=88,  # 1007
    # trying text only model or not
    text_only=False,
    # overall model parameters
    model="chalearn",
    num_epochs=100,
    batch_size=[100],  # 128,  # 32
    early_stopping_criteria=10,
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
    text_gru_hidden_dim=[300],  # 30,  # 50,  # 20
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
    final_output_dim=3,  # output vec for each task
    # FC layer parameters
    num_fc_layers=1,  # 1,  # 2,
    fc_hidden_dim=100,  # 20,  must match output_dim if final fc layer removed from base model
    final_hidden_dim=50, # the out size of dset-specific fc1 and input of fc2
    dropout=[0.4],  # 0.2
    # optimizer parameters
    lrs=[1e-4],
    beta_1=0.9,
    beta_2=0.999,
    weight_decay=0.0001,
)

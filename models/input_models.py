# holds the models needed in order to prepare input data to the main models

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pandas as pd
import math
import pprint
import warnings

from sklearn.metrics import classification_report
from sklearn.utils import shuffle


class BaseConvolution(nn.Module):
    """
    The basic convolutional model to be used inside of the encoder
    Abstracted out since it may be used TWICE within the encoder
    (1x for text, 1x for audio)
    """
    def __init__(self, input_dim, out_channels, output_dim, num_conv_layers, kernel_size, stride,
                 num_fc_layers, dropout, dialogue_aware):
        super(BaseConvolution, self).__init__()

        # input text
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_conv_layers = num_conv_layers
        self.kernel_size = kernel_size
        self.stride = stride
        self.dropout = dropout
        self.dialogue_aware = dialogue_aware

        if dialogue_aware:
            self.conv1 = nn.Conv2d(input_dim, out_channels, kernel_size, stride, padding=1)
        else:
            self.conv1 = nn.Conv1d(input_dim, out_channels, kernel_size, stride, padding=1)

        # instantiate empty layers
        self.conv2 = None
        self.conv3 = None
        self.conv4 = None

        # add optional layers as required
        if num_conv_layers > 1:
            if dialogue_aware:
                self.conv2 = nn.Conv2d(out_channels, out_channels, self.kernel_size, stride)
            else:
                self.conv2 = nn.Conv1d(out_channels, out_channels, self.kernel_size, stride)
            if num_conv_layers > 2:
                if dialogue_aware:
                    self.conv3 = nn.Conv2d(out_channels, out_channels, self.kernel_size, stride)
                else:
                    self.conv3 = nn.Conv1d(out_channels, out_channels, self.kernel_size, stride)
                if num_conv_layers == 4:
                    if dialogue_aware:
                        self.conv4 = nn.Conv2d(out_channels, out_channels, self.kernel_size)
                    else:
                        self.conv4 = nn.Conv1d(out_channels, out_channels, self.kernel_size)

        # fc layers
        if num_fc_layers > 1:
            self.fc1 = nn.Linear(out_channels, out_channels)
            self.fc2 = nn.Linear(out_channels, output_dim)
        else:
            self.fc1 = nn.Linear(out_channels, output_dim)
            self.fc2 = None

    def forward(self, inputs):
        # todo: try the below for packing padded sequences
        # inputs = nn.utils.rnn.pack_padded_sequence(inputs, enforce_sorted=False)
        # print("Input shape is: {}".format(inputs.shape))

        if self.conv4 is not None:
            intermediate_1 = torch.relu(self.conv1(inputs))
            intermediate_2 = torch.relu(self.conv2(intermediate_1))
            intermediate_3 = torch.relu(self.conv3(intermediate_2))
            feats = torch.relu(self.conv4(intermediate_3))
        elif self.conv3 is not None:
            intermediate_1 = torch.relu(self.conv1(inputs))
            intermediate_2 = torch.relu(self.conv2(intermediate_1))
            feats = torch.relu(self.conv3(intermediate_2))
        elif self.conv2 is not None:
            intermediate_1 = F.leaky_relu(self.conv1(inputs))
            feats = F.leaky_relu(self.conv2(intermediate_1))
        else:
            feats = F.leaky_relu(self.conv1(inputs))

        if self.dialogue_aware:
            feats = torch.max(feats, dim=3)[0]
            feats = feats.permute(0, 2, 1)
        else:
            feats = torch.max(feats, dim=2)[0]

        # use pooled, squeezed feats as input into fc layers
        if self.fc2 is not None:
            fc1_out = torch.tanh(self.fc1((F.dropout(feats, self.dropout))))
            output = self.fc2(F.dropout(fc1_out, self.dropout))
        else:
            output = self.fc1(F.dropout(feats, self.dropout))

        # squeeze to 1 dimension for binary categorization
        if self.output_dim == 1:
            output = output.squeeze(1)

        # return the output
        # output is NOT fed through softmax or sigmoid layer here
        # assumption: output is intermediate layer of larger NN
        return output


class BasicEncoder(nn.Module):
    """
    An encoder to take a sequence of inputs and produce a sequence of intermediate representations
    Can include convolutions over text input and/or acoustic input--BUT NOT TOGETHER bc MELD isn't
    aligned at the word-level
    todo: for now, we assume utterance-level aggregation of acoustic features, so only conv option
        fully built is for text input; need to add acoustic conv--will involve reformatting input
        data for utterance.
    """
    def __init__(self, params, num_embeddings=None, pretrained_embeddings=None):
        super(BasicEncoder, self).__init__()
        # input text + acoustic + speaker
        self.text_dim = params.text_dim
        self.audio_dim = params.audio_dim
        self.num_embeddings = num_embeddings
        self.softmax = params.softmax

        # to determine if using convolutional networks
        self.text_network = params.text_network
        self.alignment = params.alignment

        # to determine if utterances are organized into dialogues
        self.dialogue_aware = params.dialogue_aware

        # if we need to use different modalities
        self.use_speaker = params.use_speaker
        # todo: implement these
        self.use_acoustic = params.use_acoustic
        self.use_text = params.use_text

        # if we feed text through additional layer(s)
        self.text_output_dim = params.text_output_dim
        if params.text_network:
            self.text_conv = BaseConvolution(params.text_dim, params.text_out_channels, params.text_output_dim,
                                             params.num_text_conv_layers, params.kernel_size, params.stride,
                                             params.num_text_fc_layers, params.dropout, params.dialogue_aware)

        # fixme: not yet fully implemented/tested
        if params.alignment is not "utt":
            self.audio_conv = BaseConvolution(params.audio_dim, params.audio_out_channels, params.audio_output_dim,
                                              params.num_audio_conv_layers, params.kernel_size, params.stride,
                                              params.num_audio_fc_layers, params.dropout, params.dialogue_aware)

        # set the size of the input into the fc layers
        if params.alignment is not "utt":
            self.audio_fc_input_dim = params.audio_output_dim
        else:
            self.audio_fc_input_dim = params.audio_dim

        if params.text_network:
            self.text_fc_input_dim = params.text_output_dim
        else:
            self.text_fc_input_dim = params.text_dim

        # initialize speaker embeddings
        if params.use_speaker:
            self.speaker_embeddings = nn.Embedding(params.num_speakers, params.spkr_emb_dim, max_norm=1.0)

        # set input dimensions for fc layers
        self.acoustic_fc = 0
        self.spkr_fc = 0
        self.text_fc = 0

        if params.use_speaker:
            self.spkr_fc = params.spkr_emb_dim
        if params.use_acoustic:
            self.acoustic_fc = self.audio_fc_input_dim
        if params.use_text:
            self.text_fc = self.text_fc_input_dim

        self.fc_input_dim = self.acoustic_fc + self.spkr_fc + self.text_fc

        # # set input dimensions for fc layer(s)
        # if params.use_speaker and params.use_acoustic and params.use_text:
        #     self.fc_input_dim = self.audio_fc_input_dim + self.text_fc_input_dim + params.spkr_emb_dim
        # elif params.use_acoustic and params.use_text:
        #     self.fc_input_dim = self.audio_fc_input_dim + self.text_fc_input_dim
        # elif params.use_acoustic and params.use_speaker:
        #     self.fc_input_dim = self.audio_fc_input_dim + params.spkr_emb_dim
        # elif params.use_text and params.use_speaker:
        #     self.fc_input_dim = self.text_fc_input_dim + params.spkr_emb_dim
        # elif params.use_text:
        #     self.fc_input_dim = self.text_fc_input_dim
        # elif params.use_acoustic:
        #     self.fc_input_dim = self.

        # set number of classes
        self.output_dim = params.output_dim

        # set number of layers and dropout
        self.num_fc_layers = params.num_fc_layers
        self.dropout = params.dropout

        # set convolutional kernel size and number of channels
        self.kernel_size = params.kernel_size
        self.out_channels = params.out_channels

        # initialize word embeddings
        if num_embeddings is not None:
            if pretrained_embeddings is None:
                self.embedding = nn.Embedding(num_embeddings, self.text_dim, params.padding_idx, max_norm=1.0)
                self.pretrained_embeddings = False
            else:
                self.embedding = nn.Embedding(num_embeddings, self.text_dim, params.padding_idx,
                                              _weight=pretrained_embeddings, max_norm=1.0)
                self.pretrained_embeddings = True

        # initialize fully connected layers
        if params.num_fc_layers == 1:
            self.fc1 = nn.Linear(self.fc_input_dim, params.output_dim)
            self.fc2 = None
        elif params.num_fc_layers == 2:
            self.fc1 = nn.Linear(self.fc_input_dim, params.fc_hidden_dim)
            self.fc2 = nn.Linear(params.fc_hidden_dim, params.output_dim)

    def forward(self, acoustic_input, text_input, speaker_input=None):
        # set number to use in calculations
        # using "dialogue aware" means every tensor will
        # have an additional dimension
        if self.dialogue_aware:
            dialogue_dim = 1
        else:
            dialogue_dim = 0

        # reshape speaker input if needed
        if self.use_speaker:
            if self.dialogue_aware:
                speaker_input = speaker_input.unsqueeze(2)
                acoustic_input = acoustic_input.permute(0, 3, 1, 2)  # todo: untested
            else:
                acoustic_input = acoustic_input.permute(0, 2, 1)

            # if not using utt_avgd acoustic feats, feed acoustic through CNN
            if self.alignment is not "utt":
                acoustic_input = self.audio_conv(acoustic_input)
            elif not self.dialogue_aware:
                acoustic_input = torch.mean(acoustic_input, dim=2)

        # if using pretrained, detach to not update weights
        if self.num_embeddings is not None and self.use_text:
            if self.pretrained_embeddings:
                embs = self.embedding(text_input).detach()
            else:
                embs = self.embedding(text_input)

            # permute the text embeddings and acoustic input
            if self.dialogue_aware:
                embs = embs.permute(0, 3, 1, 2)
            else:
                embs = embs.permute(0, 2, 1)

            # average embeddings OR feed through network
            if self.text_network:
                utt_embs = self.text_conv(embs)
            else:
                utt_embs = torch.mean(embs, dim=(2 + dialogue_dim))

        # combine modalities as required by architecture
        if self.use_speaker:
            spk_embs = self.speaker_embeddings(speaker_input).squeeze(dim=(2 + dialogue_dim))

            if self.num_embeddings is not None and self.use_text and self.use_acoustic:
                inputs = torch.cat((acoustic_input, utt_embs, spk_embs), 1 + dialogue_dim)
            elif self.use_acoustic:
                inputs = torch.cat((acoustic_input, spk_embs), 1 + dialogue_dim)
            else:
                inputs = spk_embs
            # print("inputs concatenated")
        elif self.use_acoustic:
            if self.num_embeddings is not None:
                inputs = torch.cat((acoustic_input, utt_embs), 1 + dialogue_dim)
            else:
                inputs = acoustic_input
        elif self.use_text and self.num_embeddings is not None:
            inputs = utt_embs

        # use pooled, squeezed feats as input into fc layers
        if self.fc2 is not None:
            fc1_out = torch.tanh(self.fc1((F.dropout(inputs, self.dropout))))
            output = self.fc2(F.dropout(fc1_out, self.dropout))
        else:
            output = self.fc1(F.dropout(inputs, self.dropout))
            # print("fc1 output computed")

        if self.output_dim == 1:
            output = torch.sigmoid(output)
            output = output.squeeze(1)
            # print("predictions computed")
        else:
            if self.softmax:
                output = F.softmax(output, dim=1)

        # return the output
        # squeeze to 1 dimension for binary categorization
        return output


class EmotionToSuccessFFNN(nn.Module):
    """
    A decoder for determining success in LIvES based on emotion prediction training
    Uses output of Basic Encoder
    OR previous layer
    """
    def __init__(self, params, num_utts, num_layers, hidden_dim, output_dim):
        super(EmotionToSuccessFFNN, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_utts = num_utts
        self.dropout = params.dropout
        self.softmax = params.softmax

        if num_layers == 2:
            self.fc1 = nn.Linear(num_utts, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
        elif num_layers == 1:
            self.fc1 = nn.Linear(num_utts, output_dim)

    def forward(self, inputs):
        # perform max-pool to get predicted emotion per utterance
        # print(inputs.shape)
        squeezed_size = inputs.size(dim=2)
        per_utt_preds = F.max_pool1d(inputs, squeezed_size).squeeze(2)

        # feed data through linear layer(s)
        if self.fc2:
            intermediate_1 = torch.tanh(self.fc1(F.dropout(per_utt_preds, self.dropout)))
            output = self.fc2(F.dropout(intermediate_1, self.dropout))
        else:
            output = self.fc1(F.dropout(per_utt_preds, self.dropout))

        # perform softmax or sigmoid
        if self.output_dim == 1:
            output = torch.sigmoid(output)
            output = output.squeeze(1)
        else:
            if self.softmax:
                output = F.softmax(output)

        return output
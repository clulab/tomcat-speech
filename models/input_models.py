# the models used for multimodal, multitask classification

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseGRU(nn.Module):
    """
    The basic GRU model to be used inside of the encoder
    Abstracted out since it may be used TWICE within the encoder
    (1x for text, 1x for audio)
    """
    def __init__(self, input_dim, hidden_size, output_dim, num_gru_layers,
                 num_fc_layers, dropout, bidirectional):
        super(BaseGRU, self).__init__()

        # input text
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_gru_layers = num_gru_layers
        self.dropout = dropout

        self.GRU = nn.GRU(input_dim, hidden_size, num_gru_layers, batch_first=True, dropout=dropout,
                          bidirectional=bidirectional)

        # fc layers
        if num_fc_layers > 1:
            self.fc1 = nn.Linear(hidden_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_dim)
        else:
            self.fc1 = nn.Linear(hidden_size, output_dim)
            self.fc2 = None

    def forward(self, inputs, input_lengths):
        inputs = nn.utils.rnn.pack_padded_sequence(inputs, input_lengths,
                                                   batch_first=True, enforce_sorted=False)

        rnn_feats, hidden = self.GRU(inputs)

        # reshape features for e
        feats = hidden.permute(1, 0, 2)

        # use pooled, squeezed feats as input into fc layers
        output = self.fc1(F.dropout(feats, self.dropout))

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

        # if we feed text through additional layer(s)
        self.text_output_dim = params.text_output_dim
        self.text_gru = BaseGRU(params.text_dim, params.text_gru_hidden_dim, params.text_output_dim,
                                params.num_gru_layers, params.num_fc_layers, params.dropout,
                                params.bidirectional)

        # set the size of the input into the fc layers
        self.fc_input_dim = params.text_output_dim + params.audio_dim

        # set number of classes
        self.output_dim = params.output_dim

        # set number of layers and dropout
        self.dropout = params.dropout

        # initialize word embeddings
        self.embedding = nn.Embedding(num_embeddings, self.text_dim,
                                      _weight=pretrained_embeddings, max_norm=1.0)

        # initialize fully connected layers
        self.fc1 = nn.Linear(self.fc_input_dim, params.output_dim)

    def forward(self, acoustic_input, text_input, length_input=None):
        # using pretrained embeddings, so detach to not update weights
        embs = self.embedding(text_input).detach()

        # feed embeddings through GRU
        utt_embs = self.text_gru(embs, length_input)
        utt_embs = utt_embs.permute(0, 2, 1)
        # take max (or avg, etc) to reduce dimensionality
        utt_embs = torch.mean(utt_embs, dim=2)

        # combine modalities as required by architecture
        inputs = torch.cat((acoustic_input, utt_embs), 1)

        # use pooled, squeezed feats as input into fc layers
        output = self.fc1(F.dropout(inputs, self.dropout))
        output = F.softmax(output, dim=1)

        # return the output
        return output
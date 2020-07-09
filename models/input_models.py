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

        self.GRU = nn.LSTM(input_dim, output_dim, num_gru_layers, batch_first=True, dropout=dropout,
                          bidirectional=bidirectional)
        # self.GRU = nn.GRU(input_dim, output_dim, num_gru_layers, batch_first=True, dropout=dropout,
        #                   bidirectional=bidirectional)

    def forward(self, inputs, input_lengths):
        inputs = nn.utils.rnn.pack_padded_sequence(inputs, input_lengths,
                                                   batch_first=True, enforce_sorted=False)

        # todo: look at this--make sure we're taking hidden from the right place
        packed_output, (hidden, cell) = self.GRU(inputs)
        # rnn_feats, hidden = self.GRU(inputs)

        output = hidden[:,-1,:]

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
        self.num_speakers = params.num_speakers

        # if we feed text through additional layer(s)
        # self.text_output_dim = params.text_output_dim
        self.text_rnn = nn.LSTM(
            input_size=params.text_dim + params.short_emb_dim,
            hidden_size=params.text_gru_hidden_dim,
            num_layers=params.num_gru_layers, 
            batch_first=True,
            bidirectional=False)

        self.acoustic_rnn = nn.LSTM(
            input_size=params.audio_dim,
            hidden_size=params.acoustic_gru_hidden_dim,
            num_layers=params.num_gru_layers,
            batch_first=True,
            bidirectional=False
        )

        # set the size of the input into the fc layers
        if params.avgd_acoustic or params.add_avging:
            self.fc_input_dim = params.text_gru_hidden_dim + params.audio_dim
        else:
            self.fc_input_dim = params.text_gru_hidden_dim + params.acoustic_gru_hidden_dim

        if params.use_speaker:
            self.fc_input_dim = self.fc_input_dim + params.speaker_emb_dim

        # self.fc_input_dim = params.text_output_dim

        # set number of classes
        self.output_dim = params.output_dim

        # set number of layers and dropout
        self.dropout = params.dropout

        # initialize word embeddings
        self.embedding = nn.Embedding(num_embeddings, self.text_dim,
                                      _weight=pretrained_embeddings)
        self.short_embedding = nn.Embedding(num_embeddings, params.short_emb_dim)
        # self.embedding = nn.Embedding(num_embeddings, self.text_dim)

        # initialize speaker embeddings
        self.speaker_embedding = nn.Embedding(params.num_speakers, params.speaker_emb_dim)

        # initialize fully connected layers
        self.fc1 = nn.Linear(self.fc_input_dim, params.fc_hidden_dim)
        self.fc2 = nn.Linear(params.fc_hidden_dim, params.output_dim)

    def forward(self, acoustic_input, text_input, speaker_input=None, length_input=None, acoustic_len_input=None):
        # using pretrained embeddings, so detach to not update weights
        # embs: (batch_size, seq_len, emb_dim)
        # embs = F.dropout(self.embedding(text_input), self.dropout).detach()
        embs = self.embedding(text_input).detach()

        # short_embs = F.dropout(self.short_embedding(text_input), self.dropout)
        short_embs = self.short_embedding(text_input)

        all_embs = torch.cat((embs, short_embs), dim=2)

        # get speaker embeddings, if needed
        if speaker_input is not None:
            speaker_embs = self.speaker_embedding(speaker_input).squeeze(dim=1)

        # packed = nn.utils.rnn.pack_padded_sequence(embs, length_input, batch_first=True, enforce_sorted=False)
        packed = nn.utils.rnn.pack_padded_sequence(all_embs, length_input, batch_first=True, enforce_sorted=False)

        # feed embeddings through GRU
        packed_output, (hidden, cell) = self.text_rnn(packed)
        # padded_output, lens = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        encoded_text = F.dropout(hidden[-1], self.dropout)

        if acoustic_len_input is not None:
            packed_acoustic = nn.utils.rnn.pack_padded_sequence(acoustic_input, acoustic_len_input, batch_first=True,
                                                                enforce_sorted=False)

            packed_acoustic_output, (acoustic_hidden, acoustic_cell) = self.acoustic_rnn(packed_acoustic)
            encoded_acoustic = F.dropout(acoustic_hidden[-1], self.dropout)

        else:
            encoded_acoustic = acoustic_input.squeeze()

        # inputs = encoded_text
        # combine modalities as required by architecture
        # inputs = torch.cat((acoustic_input, encoded_text), 1)
        if speaker_input is not None:
            inputs = torch.cat((encoded_acoustic, encoded_text, speaker_embs), 1)
        else:
            inputs = torch.cat((encoded_acoustic, encoded_text), 1)

        # use pooled, squeezed feats as input into fc layers
        output = torch.tanh(self.fc1(inputs))
        output = torch.relu(self.fc2(output))
        # output = F.softmax(output, dim=1)

        # return the output
        return output


class TextOnlyCNN(nn.Module):
    """
    A CNN with multiple input channels with different kernel size operating over input
    Used with only text modality.
    """
    def __init__(self, params, num_embeddings, pretrained_embeddings=None):
        super(TextOnlyCNN, self).__init__()
        # input dimensions
        self.text_dim = params.text_dim
        self.in_channels = params.text_dim

        # number of classes
        self.output_dim = params.output_dim

        # self.num_cnn_layers = params.num_cnn_layers
        self.dropout = params.dropout

        # kernels for each layer
        self.k1_size = params.kernel_1_size
        self.k2_size = params.kernel_2_size
        self.k3_size = params.kernel_3_size

        # number of output channels from conv layers
        self.out_channels = params.out_channels

        # word embeddings
        if pretrained_embeddings is None:
            self.embedding = nn.Embedding(num_embeddings, self.text_dim, padding_idx=0, max_norm=1.0)
            self.pretrained_embeddings = False
        else:
            self.embedding = nn.Embedding(num_embeddings, self.text_dim, padding_idx=0,
                                          _weight=pretrained_embeddings, max_norm=1.0)
            self.pretrained_embeddings = True

        self.conv1 = nn.Conv1d(self.in_channels, self.out_channels, self.k1_size)
        self.maxconv1 = nn.MaxPool1d(kernel_size=self.k1_size)
        self.conv2 = nn.Conv1d(self.in_channels, self.out_channels, self.k2_size)
        self.maxconv2 = nn.MaxPool1d(kernel_size=self.k2_size)
        self.conv3 = nn.Conv1d(self.in_channels, self.out_channels, self.k3_size)
        self.maxconv3 = nn.MaxPool1d(kernel_size=self.k3_size)

        # fully connected layers
        self.fc1 = nn.Linear(self.out_channels * 3, params.text_cnn_hidden_dim)
        self.fc2 = nn.Linear(params.text_cnn_hidden_dim, self.output_dim)

    def forward(self, acoustic_input, text_input, speaker_input=None, length_input=None):
        # get word embeddings
        if self.pretrained_embeddings:
            # detach to avoid training them if using pretrained
            inputs = self.embedding(text_input).detach()
        else:
            inputs = self.embedding(text_input)

        inputs = inputs.permute(0, 2, 1)

        # feed data into convolutional layers
        # dim=2 says it's an unexpected argument, but it is needed for this to work
        conv1_out = F.leaky_relu(self.conv1(inputs))
        feats1 = F.max_pool1d(conv1_out, conv1_out.size(dim=2)).squeeze(dim=2)
        conv2_out = F.leaky_relu(self.conv2(inputs))
        feats2 = F.max_pool1d(conv2_out, conv2_out.size(dim=2)).squeeze(dim=2)
        conv3_out = F.leaky_relu(self.conv3(inputs))
        feats3 = F.max_pool1d(conv3_out, conv3_out.size(dim=2)).squeeze(dim=2)

        # combine output of convolutional layers
        intermediate = torch.cat((feats1, feats2, feats3), 1)

        # feed this through fully connected layer
        fc1_out = torch.tanh(self.fc1((F.dropout(intermediate, self.dropout))))

        output = torch.relu(self.fc2(F.dropout(fc1_out, self.dropout)))
        # output = self.fc2(fc1_out)

        return output.squeeze(dim=1)


class PredictionLayer(nn.Module):
    """
    A final layer for predictions
    """
    def __init__(self, params, out_dim):
        super(PredictionLayer, self).__init__()
        self.input_dim = params.text_gru_hidden_dim

        # specify out_dim explicity so we can do multiple tasks at once
        self.output_dim = out_dim

        self.fc1 = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, combined_inputs):
        out = torch.relu(self.fc1(F.dropout(combined_inputs, self.dropout)))

        return out

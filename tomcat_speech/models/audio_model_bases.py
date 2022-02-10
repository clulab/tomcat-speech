
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseGRU(nn.Module):
    """
    The basic GRU model to be used inside of the encoder
    Abstracted out since it may be used TWICE within the encoder
    (1x for text, 1x for audio)
    """

    def __init__(
        self,
        input_dim,
        hidden_size,
        output_dim,
        num_gru_layers,
        num_fc_layers,
        dropout,
        bidirectional,
    ):
        super(BaseGRU, self).__init__()

        # input text
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_gru_layers = num_gru_layers
        self.dropout = dropout

        self.GRU = nn.LSTM(
            input_dim,
            output_dim,
            num_gru_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        # self.GRU = nn.GRU(input_dim, output_dim, num_gru_layers, batch_first=True, dropout=dropout,
        #                   bidirectional=bidirectional)

    def forward(self, inputs, input_lengths):
        inputs = nn.utils.rnn.pack_padded_sequence(
            inputs, input_lengths, batch_first=True, enforce_sorted=False
        )

        packed_output, (hidden, cell) = self.GRU(inputs)
        # rnn_feats, hidden = self.GRU(inputs)

        # todo: test this model somewhere
        #   other RNNs use hidden[-1] bc of batch_first=True
        output = hidden[:, -1, :]

        # output is NOT fed through softmax or sigmoid layer here
        # assumption: output is intermediate layer of larger NN
        return output


class AudioOnlyRNN(nn.Module):
    """
    An RNN used with RAVDESS, where primary information comes from audio
    Has capacity to include gender embeddings, todo: add anything?
    """

    def __init__(self, params):
        super(AudioOnlyRNN, self).__init__()

        # input dimensions
        self.audio_dim = params.audio_dim

        self.acoustic_rnn = nn.LSTM(
            input_size=params.audio_dim,
            hidden_size=params.acoustic_gru_hidden_dim,
            num_layers=params.num_gru_layers,
            batch_first=True,
            bidirectional=False,
        )

        # acoustic batch normalization
        self.acoustic_batch_norm = nn.BatchNorm1d(params.audio_dim)

        self.acoustic_fc_1 = nn.Linear(params.acoustic_gru_hidden_dim, 50)
        # self.acoustic_fc_2 = nn.Linear(100, 20)
        self.acoustic_fc_2 = nn.Linear(50, params.audio_dim)

        # dimension of input into final fc layers
        self.fc_input_dim = params.acoustic_gru_hidden_dim
        # self.fc_input_dim = params.audio_dim

        if params.use_speaker:
            self.fc_input_dim = self.fc_input_dim + params.speaker_emb_dim
        elif params.use_gender:
            self.fc_input_dim = self.fc_input_dim + params.gender_emb_dim

        # set number of classes
        self.output_dim = params.output_dim

        # set number of layers and dropout
        self.dropout = params.dropout

        self.gender_embedding = nn.Embedding(3, params.gender_emb_dim)

        # initialize fully connected layers
        self.fc1 = nn.Linear(self.fc_input_dim, params.fc_hidden_dim)

        self.fc2 = nn.Linear(params.fc_hidden_dim, params.output_dim)

    def forward(
        self,
        acoustic_input,
        acoustic_len_input,
        speaker_input=None,
        gender_input=None,
        text_input=None,
        length_input=None,
    ):
        # get speaker embeddings, if needed
        if speaker_input is not None:
            speaker_embs = self.speaker_embedding(speaker_input).squeeze(dim=1)
            # speaker_embs = self.speaker_batch_norm(speaker_embs)
        if gender_input is not None:
            gender_embs = self.gender_embedding(gender_input)

        # normalize
        # acoustic_input = self.acoustic_batch_norm(acoustic_input)
        # pack acoustic input
        packed = nn.utils.rnn.pack_padded_sequence(
            acoustic_input, length_input, batch_first=True, enforce_sorted=False
        )

        # feed embeddings through GRU
        packed_output, (hidden, cell) = self.acoustic_rnn(packed)

        encoded_acoustic = F.dropout(hidden[-1], 0.3)

        # encoded_acoustic = torch.tanh(F.dropout(self.acoustic_fc_1(encoded_acoustic), self.dropout))
        # encoded_acoustic = torch.tanh(F.dropout(self.acoustic_fc_2(encoded_acoustic), self.dropout))

        # combine modalities as required by architecture
        # inputs = torch.cat((acoustic_input, encoded_text), 1)
        if speaker_input is not None:
            inputs = torch.cat((encoded_acoustic, speaker_embs), 1)
        elif gender_input is not None:
            inputs = torch.cat((encoded_acoustic, gender_embs), 1)
        else:
            inputs = encoded_acoustic

        output = torch.tanh(F.dropout(self.fc1(inputs), 0.5))
        output = torch.relu(self.fc2(output))

        if self.output_dim == 1:
            output = torch.sigmoid(output)

        # return the output
        return output


class AcousticOnlyForMultitask(nn.Module):
    """
    A model using only acoustic features
    """

    def __init__(self, params, multi_dataset=True, use_rnn=False):
        super(AcousticOnlyForMultitask, self).__init__()

        # determine whether multiple datasets are used
        self.multi_dataset = multi_dataset
        self.use_rnn = use_rnn

        # set dropout
        self.dropout = params.dropout

        if use_rnn:
            # instantiate RNN if using
            self.acoustic_rnn = nn.LSTM(
                input_size=params.audio_dim,
                hidden_size=params.acoustic_gru_hidden_dim,
                num_layers=4,
                batch_first=True,
                bidirectional=True,
            )
        else:
            self.acoustic_fc1 = nn.Linear(
                params.audio_dim, params.acoustic_gru_hidden_dim
            )
            self.acoustic_fc2 = nn.Linear(
                params.acoustic_gru_hidden_dim, params.output_dim
            )

        # acoustic batch normalization
        self.acoustic_batch_norm = nn.BatchNorm1d(params.audio_dim)

    def forward(self, acoustic_input, length_input=None):
        # put acoustic features through batch normalization
        # acoustic_input = self.acoustic_batch_norm(acoustic_input)

        acoustic_input = torch.normal(acoustic_input)

        # noise = torch.ones(acoustic_input.shape)
        # noise = noise.normal_()
        #
        # noise = torch.normal(0, 0.2, size=acoustic_input.shape[1])

        # acoustic_input = torch.add(acoustic_input, noise[:, None])

        if self.use_rnn:
            # pack the data
            packed_feats = nn.utils.rnn.pack_padded_sequence(
                acoustic_input, length_input, batch_first=True, enforce_sorted=False
            )

            packed_output, (hidden, cell) = self.acoustic_rnn(packed_feats)

            encoded_acoustic = F.dropout(hidden[-1], self.dropout)
        else:
            feats = torch.relu(
                F.dropout(self.acoustic_fc1(acoustic_input), self.dropout)
            )
            encoded_acoustic = torch.relu(
                F.dropout(self.acoustic_fc2(feats), self.dropout)
            )

        return encoded_acoustic


class IntermediateFusionAcousticOnlyModel(nn.Module):
    """
    An encoder to take a sequence of inputs and produce a sequence of intermediate representations
    Can include convolutions over text input and/or acoustic input--BUT NOT TOGETHER bc MELD isn't
    aligned at the word-level
    """

    def __init__(self, params, num_embeddings=None, pretrained_embeddings=None):
        super(IntermediateFusionAcousticOnlyModel, self).__init__()
        # input text + acoustic + speaker
        self.text_dim = params.text_dim
        self.audio_dim = params.audio_dim
        self.num_embeddings = num_embeddings
        self.num_speakers = params.num_speakers
        self.text_gru_hidden_dim = params.text_gru_hidden_dim

        # get number of output dims
        self.out_dims = params.output_dim

        self.acoustic_rnn = nn.LSTM(
            input_size=params.audio_dim,
            hidden_size=params.acoustic_gru_hidden_dim,
            num_layers=4,
            batch_first=True,
            bidirectional=True,
        )

        # set the size of the input into the fc layers
        self.fc_input_dim = params.audio_dim

        if params.add_avging is False and params.avgd_acoustic is False:
            self.acoustic_fc_1 = nn.Linear(params.fc_hidden_dim, 100)
        else:
            self.acoustic_fc_1 = nn.Linear(params.audio_dim, 50)
        self.acoustic_fc_2 = nn.Linear(50, params.audio_dim)

        if params.use_speaker:
            self.fc_input_dim = self.fc_input_dim + params.speaker_emb_dim
        elif params.use_gender:
            self.fc_input_dim = self.fc_input_dim + params.gender_emb_dim

        # set number of classes
        self.output_dim = params.output_dim

        # set number of layers and dropout
        self.dropout = params.dropout

        # initialize speaker embeddings
        self.speaker_embedding = nn.Embedding(
            params.num_speakers, params.speaker_emb_dim
        )

        self.gender_embedding = nn.Embedding(3, params.gender_emb_dim)

        # initialize fully connected layers
        self.fc1 = nn.Linear(self.fc_input_dim, params.fc_hidden_dim)
        self.fc2 = nn.Linear(params.fc_hidden_dim, params.output_dim)

    def forward(
        self,
        acoustic_input,
        text_input,
        speaker_input=None,
        length_input=None,
        acoustic_len_input=None,
        gender_input=None,
    ):
        # get speaker embeddings, if needed
        if speaker_input is not None:
            speaker_embs = self.speaker_embedding(speaker_input).squeeze(dim=1)
        if gender_input is not None:
            gender_embs = self.gender_embedding(gender_input)

        if acoustic_len_input is not None:
            packed_acoustic = nn.utils.rnn.pack_padded_sequence(
                acoustic_input,
                acoustic_len_input.clamp(max=1500),
                batch_first=True,
                enforce_sorted=False,
            )
            (
                packed_acoustic_output,
                (acoustic_hidden, acoustic_cell),
            ) = self.acoustic_rnn(packed_acoustic)
            encoded_acoustic = F.dropout(acoustic_hidden[-1], self.dropout)

        else:
            if len(acoustic_input.shape) > 2:
                encoded_acoustic = acoustic_input.squeeze()
            else:
                encoded_acoustic = acoustic_input

        encoded_acoustic = torch.tanh(
            F.dropout(self.acoustic_fc_1(encoded_acoustic), self.dropout)
        )
        encoded_acoustic = torch.tanh(
            F.dropout(self.acoustic_fc_2(encoded_acoustic), self.dropout)
        )

        # combine modalities as required by architecture
        if speaker_input is not None:
            inputs = torch.cat((encoded_acoustic, speaker_embs), 1)
        elif gender_input is not None:
            inputs = torch.cat((encoded_acoustic, gender_embs), 1)
        else:
            inputs = encoded_acoustic

        # use pooled, squeezed feats as input into fc layers
        output = torch.tanh(F.dropout(self.fc1(inputs), 0.5))

        if self.out_dims == 1:
            output = torch.sigmoid(output)

        # return the output
        return output

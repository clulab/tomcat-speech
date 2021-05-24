# the models used for multimodal, multitask classification
import sys

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

        # todo: look at this--make sure we're taking hidden from the right place
        packed_output, (hidden, cell) = self.GRU(inputs)
        # rnn_feats, hidden = self.GRU(inputs)

        output = hidden[:, -1, :]

        # output is NOT fed through softmax or sigmoid layer here
        # assumption: output is intermediate layer of larger NN
        return output


class EarlyFusionMultimodalModel(nn.Module):
    """
    An encoder to take a sequence of inputs and produce a sequence of intermediate representations
    Can include convolutions over text input and/or acoustic input--BUT NOT TOGETHER bc MELD isn't
    aligned at the word-level
    """

    def __init__(self, params, num_embeddings=None, pretrained_embeddings=None):
        super(EarlyFusionMultimodalModel, self).__init__()
        # input text + acoustic + speaker
        self.text_dim = params.text_dim
        self.audio_dim = params.audio_dim
        self.num_embeddings = num_embeddings
        self.num_speakers = params.num_speakers
        self.text_gru_hidden_dim = params.text_gru_hidden_dim

        # get number of output dims
        self.out_dims = params.output_dim

        # if we feed text through additional layer(s)
        self.text_rnn = nn.LSTM(
            input_size=params.text_dim + params.short_emb_dim,
            hidden_size=params.text_gru_hidden_dim,
            num_layers=params.num_gru_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.acoustic_rnn = nn.LSTM(
            input_size=params.audio_dim,
            hidden_size=params.acoustic_gru_hidden_dim,
            num_layers=4,
            batch_first=True,
            bidirectional=True,
        )

        # set the size of the input into the fc layers
        # if params.avgd_acoustic or params.add_avging:
        self.fc_input_dim = params.text_gru_hidden_dim + params.audio_dim

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

        # initialize word embeddings
        self.embedding = nn.Embedding(
            num_embeddings, self.text_dim, _weight=pretrained_embeddings
        )
        self.short_embedding = nn.Embedding(num_embeddings, params.short_emb_dim)
        # self.text_batch_norm = nn.BatchNorm1d(self.text_dim + params.short_emb_dim)

        # initialize speaker embeddings
        self.speaker_embedding = nn.Embedding(
            params.num_speakers, params.speaker_emb_dim
        )

        # self.speaker_batch_norm = nn.BatchNorm1d(params.speaker_emb_dim)

        self.gender_embedding = nn.Embedding(3, params.gender_emb_dim)

        # acoustic batch normalization
        # self.acoustic_batch_norm = nn.BatchNorm1d(params.audio_dim)
        # self.acoustic_unk_norm = nn.BatchNorm1d(params.audio_dim)
        # self.acoustic_female_norm = nn.BatchNorm1d(params.audio_dim)
        # self.acoustic_male_norm = nn.BatchNorm1d(params.audio_dim)

        # initialize fully connected layers
        self.fc1 = nn.Linear(self.fc_input_dim, params.fc_hidden_dim)

        # self.interfc_batch_norm = nn.BatchNorm1d(params.fc_hidden_dim)

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
        # using pretrained embeddings, so detach to not update weights
        # embs: (batch_size, seq_len, emb_dim)
        embs = F.dropout(self.embedding(text_input), 0.1).detach()

        short_embs = F.dropout(self.short_embedding(text_input), 0.1)

        all_embs = torch.cat((embs, short_embs), dim=2)

        # get speaker embeddings, if needed
        if speaker_input is not None:
            speaker_embs = self.speaker_embedding(speaker_input).squeeze(dim=1)
        if gender_input is not None:
            gender_embs = self.gender_embedding(gender_input)

        packed = nn.utils.rnn.pack_padded_sequence(
            all_embs, length_input, batch_first=True, enforce_sorted=False
        )

        # feed embeddings through GRU
        packed_output, (hidden, cell) = self.text_rnn(packed)
        encoded_text = F.dropout(hidden[-1], 0.3)

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
            inputs = torch.cat((encoded_acoustic, encoded_text, speaker_embs), 1)
        elif gender_input is not None:
            inputs = torch.cat((encoded_acoustic, encoded_text, gender_embs), 1)
        else:
            inputs = torch.cat((encoded_acoustic, encoded_text), 1)

        # use pooled, squeezed feats as input into fc layers
        output = torch.tanh(F.dropout(self.fc1(inputs), 0.5))

        if self.out_dims == 1:
            output = torch.sigmoid(output)

        # return the output
        return output


class LateFusionMultimodalModel(nn.Module):
    """
    A late fusion model that combines modalities only at decision time
    """

    def __init__(self, params, num_embeddings=None, pretrained_embeddings=None):
        super(LateFusionMultimodalModel, self).__init__()
        self.text_dim = params.text_dim
        self.audio_dim = params.audio_dim
        self.num_embeddings = num_embeddings
        self.num_speakers = params.num_speakers
        self.text_gru_hidden_dim = params.text_gru_hidden_dim
        self.text_fc_input_dim = params.text_gru_hidden_dim + params.gender_emb_dim
        self.text_fc_hidden_dim = 100

        # get number of output dims
        self.out_dims = params.output_dim

        # if we feed text through additional layer(s)
        self.text_rnn = nn.LSTM(
            input_size=params.text_dim + params.short_emb_dim,
            hidden_size=params.text_gru_hidden_dim,
            num_layers=params.num_gru_layers,
            batch_first=True,
            bidirectional=True,
        )

        # initialize fully connected layers
        self.text_fc1 = nn.Linear(self.text_fc_input_dim, self.text_fc_hidden_dim)
        self.text_fc2 = nn.Linear(self.text_fc_hidden_dim, params.output_dim)

        # initialize acoustic portions of model
        self.acoustic_rnn = nn.LSTM(
            input_size=params.audio_dim,
            hidden_size=params.acoustic_gru_hidden_dim,
            num_layers=params.num_gru_layers,
            batch_first=True,
            bidirectional=False,
        )

        if params.avgd_acoustic or params.add_avging:
            self.acoustic_fc_1 = nn.Linear(
                params.audio_dim + params.gender_emb_dim, 100
            )
        else:
            self.acoustic_fc_1 = nn.Linear(params.acoustic_gru_hidden_dim, 100)
        self.acoustic_fc_2 = nn.Linear(100, params.output_dim)

        # set number of classes
        self.output_dim = params.output_dim

        # set number of layers and dropout
        self.dropout = params.dropout

        # initialize word embeddings
        self.embedding = nn.Embedding(
            num_embeddings, self.text_dim, _weight=pretrained_embeddings
        )
        self.short_embedding = nn.Embedding(num_embeddings, params.short_emb_dim)

        # initialize speaker embeddings
        self.speaker_embedding = nn.Embedding(
            params.num_speakers, params.speaker_emb_dim
        )
        self.gender_embedding = nn.Embedding(3, params.gender_emb_dim)

    def forward(
        self,
        acoustic_input,
        text_input,
        speaker_input=None,
        length_input=None,
        acoustic_len_input=None,
        gender_input=None,
    ):
        # using pretrained embeddings, so detach to not update weights
        # embs: (batch_size, seq_len, emb_dim)
        embs = F.dropout(self.embedding(text_input), 0.1).detach()
        short_embs = F.dropout(self.short_embedding(text_input), 0.1)

        all_embs = torch.cat((embs, short_embs), dim=2)

        # get speaker embeddings, if needed
        if speaker_input is not None:
            speaker_embs = self.speaker_embedding(speaker_input).squeeze(dim=1)
            # speaker_embs = self.speaker_batch_norm(speaker_embs)
        if gender_input is not None:
            gend_embs = self.gender_embedding(gender_input)

        # packed = nn.utils.rnn.pack_padded_sequence(embs, length_input, batch_first=True, enforce_sorted=False)
        packed = nn.utils.rnn.pack_padded_sequence(
            all_embs, length_input, batch_first=True, enforce_sorted=False
        )

        # feed embeddings through GRU
        packed_output, (hidden, cell) = self.text_rnn(packed)
        encoded_text = F.dropout(hidden[-1], 0.3)

        if gender_input is not None:
            encoded_text = torch.cat((encoded_text, gend_embs), dim=1)

        text_intermediate = torch.tanh(
            F.dropout(self.text_fc1(encoded_text), self.dropout)
        )
        text_predictions = torch.relu(self.text_fc2(text_intermediate))

        if acoustic_len_input is not None:
            packed_acoustic = nn.utils.rnn.pack_padded_sequence(
                acoustic_input,
                acoustic_len_input,
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

        if gender_input is not None:
            encoded_acoustic = torch.cat((encoded_acoustic, gend_embs), dim=1)

        encoded_acoustic = torch.tanh(
            F.dropout(self.acoustic_fc_1(encoded_acoustic), self.dropout)
        )
        acoustic_predictions = torch.tanh(
            F.dropout(self.acoustic_fc_2(encoded_acoustic), self.dropout)
        )

        # combine predictions to get results
        # text_predictions = torch.mul(text_predictions, 4)
        # acoustic_predictions = torch.mul(acoustic_predictions, 2)
        predictions = torch.add(text_predictions, acoustic_predictions)
        # predictions = torch.mul(text_predictions, acoustic_predictions)

        if self.out_dims == 1:
            predictions = torch.sigmoid(predictions)

        # return the output
        return predictions


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
            self.embedding = nn.Embedding(
                num_embeddings, self.text_dim, padding_idx=0, max_norm=1.0
            )
            self.pretrained_embeddings = False
        else:
            self.embedding = nn.Embedding(
                num_embeddings,
                self.text_dim,
                padding_idx=0,
                _weight=pretrained_embeddings,
                max_norm=1.0,
            )
            self.pretrained_embeddings = True

        self.conv1 = nn.Conv1d(self.in_channels, self.out_channels, self.k1_size)
        self.maxconv1 = nn.MaxPool1d(kernel_size=self.k1_size)
        self.conv2 = nn.Conv1d(self.in_channels, self.out_channels, self.k2_size)
        self.maxconv2 = nn.MaxPool1d(kernel_size=self.k2_size)
        self.conv3 = nn.Conv1d(self.in_channels, self.out_channels, self.k3_size)
        self.maxconv3 = nn.MaxPool1d(kernel_size=self.k3_size)

        # a second convolutional layer to run on top of the first ones
        # self.conv4 = nn.Conv1d(self.out_channels * 3, self.out_channels, self.k1_size)
        # self.conv5 = nn.Conv1d(self.out_channels * 3, self.out_channels, self.k2_size)
        #
        # # a third to go on the second
        # self.conv6 = nn.Conv1d(self.out_channels * 2, self.out_channels, self.k3_size)

        # fully connected layers
        self.fc1 = nn.Linear(self.out_channels * 3, params.text_cnn_hidden_dim)
        # self.fc1 = nn.Linear(self.out_channels, params.text_cnn_hidden_dim)
        self.fc2 = nn.Linear(params.text_cnn_hidden_dim, self.output_dim)

    def forward(
        self,
        acoustic_input,
        text_input,
        speaker_input=None,
        length_input=None,
        gender_input=None,
    ):
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
        # print(conv1_out.size())
        feats1 = F.max_pool1d(conv1_out, 5, stride=1)  # .squeeze(dim=2)
        # print(feats1.shape)
        conv2_out = F.leaky_relu(self.conv2(inputs))
        feats2 = F.max_pool1d(conv2_out, 4, stride=1)  # .squeeze(dim=2)
        # print(feats1.shape)
        conv3_out = F.leaky_relu(self.conv3(inputs))
        feats3 = F.max_pool1d(conv3_out, 3, stride=1)  # .squeeze(dim=2)
        # print(feats1.shape)

        # combine output of convolutional layers
        # intermediate = torch.cat((conv1_out, conv2_out, conv3_out), 1)
        intermediate = torch.cat((feats1, feats2, feats3), 1)

        # conv4_out = F.leaky_relu(self.conv4(intermediate))
        # feats4 = F.max_pool1d(conv4_out, conv4_out.size(dim=2)).squeeze(dim=2)
        # feats4 = F.max_pool1d(conv4_out, 3)
        # conv5_out = F.leaky_relu(self.conv5(intermediate))
        # feats5 = F.max_pool1d(conv5_out, 3)

        # feats5 = F.max_pool1d(conv5_out, conv5_out.size(dim=2)).squeeze(dim=2)
        # conv6_out = F.leaky_relu(self.conv6(intermediate))
        # conv6_out = F.leaky_relu(self.conv6(torch.cat((feats4, feats5), 1)))

        # feats6
        all_feats = F.max_pool1d(intermediate, intermediate.size(dim=2)).squeeze(dim=2)

        # all_feats = torch.cat((feats4, feats5, feats6), 1)

        # feed this through fully connected layer
        fc1_out = torch.tanh(self.fc1((F.dropout(all_feats, self.dropout))))
        # fc1_out = torch.tanh(self.fc1((F.dropout(intermediate, self.dropout))))

        output = torch.relu(self.fc2(F.dropout(fc1_out, self.dropout)))
        # output = self.fc2(fc1_out)

        return output.squeeze(dim=1)


class TextOnlyRNN(nn.Module):
    """
    An RNN used with only text modality.
    """

    def __init__(self, params, num_embeddings, pretrained_embeddings=None):
        super(TextOnlyRNN, self).__init__()
        # input dimensions
        self.text_dim = params.text_dim

        # number of classes
        self.output_dim = params.output_dim

        # self.num_cnn_layers = params.num_cnn_layers
        self.dropout = params.dropout

        # word embeddings
        if pretrained_embeddings is None:
            self.embedding = nn.Embedding(
                num_embeddings, self.text_dim, padding_idx=0, max_norm=1.0
            )
            self.pretrained_embeddings = False
        else:
            self.embedding = nn.Embedding(
                num_embeddings,
                self.text_dim,
                padding_idx=0,
                _weight=pretrained_embeddings,
                max_norm=1.0,
            )
            self.pretrained_embeddings = True

        self.short_embedding = nn.Embedding(num_embeddings, params.short_emb_dim)

        # initialize speaker embeddings
        self.speaker_embedding = nn.Embedding(
            params.num_speakers, params.speaker_emb_dim
        )

        self.gender_embedding = nn.Embedding(3, params.gender_emb_dim)

        # if we feed text through additional layer(s)
        self.text_rnn = nn.LSTM(
            input_size=params.text_dim + params.short_emb_dim,
            hidden_size=params.text_gru_hidden_dim,
            num_layers=params.num_gru_layers,
            batch_first=True,
            bidirectional=True,
        )
        # input into fc layers
        self.fc_input_dim = params.text_gru_hidden_dim

        if params.use_speaker:
            self.fc_input_dim = self.fc_input_dim + params.speaker_emb_dim
        elif params.use_gender:
            self.fc_input_dim = self.fc_input_dim + params.gender_emb_dim

        # fully connected layers
        self.fc1 = nn.Linear(self.fc_input_dim, params.fc_hidden_dim)
        self.fc2 = nn.Linear(params.fc_hidden_dim, params.final_hidden_dim)

        self.out_dims = params.final_hidden_dim

    def forward(
        self,
        text_input,
        speaker_input=None,
        length_input=None,
        gender_input=None,
    ):
        # using pretrained embeddings, so detach to not update weights
        # embs: (batch_size, seq_len, emb_dim)
        embs = F.dropout(self.embedding(text_input), 0.1).detach()

        short_embs = F.dropout(self.short_embedding(text_input), 0.1)

        all_embs = torch.cat((embs, short_embs), dim=2)

        # get speaker embeddings, if needed
        if speaker_input is not None:
            speaker_embs = self.speaker_embedding(speaker_input).squeeze(dim=1)
        if gender_input is not None:
            gender_embs = self.gender_embedding(gender_input)

        packed = nn.utils.rnn.pack_padded_sequence(
            all_embs, length_input, batch_first=True, enforce_sorted=False
        )

        # feed embeddings through GRU
        packed_output, (hidden, cell) = self.text_rnn(packed)
        encoded_text = F.dropout(hidden[-1], 0.3)

        # combine modalities as required by architecture
        if speaker_input is not None:
            inputs = torch.cat((encoded_text, speaker_embs), 1)
        elif gender_input is not None:
            inputs = torch.cat((encoded_text, gender_embs), 1)
        else:
            inputs = encoded_text

        # use pooled, squeezed feats as input into fc layers
        output = torch.tanh(F.dropout(self.fc1(inputs), 0.5))

        if self.out_dims == 1:
            output = torch.sigmoid(output)

        # return the output
        return output


class PredictionLayer(nn.Module):
    """
    A final layer for predictions
    """

    def __init__(self, params, out_dim):
        super(PredictionLayer, self).__init__()
        self.input_dim = params.output_dim
        self.inter_fc_prediction_dim = params.final_hidden_dim
        self.dropout = params.dropout

        # specify out_dim explicity so we can do multiple tasks at once
        self.output_dim = out_dim

        # self.fc1 = nn.Linear(self.input_dim, self.output_dim)
        self.fc1 = nn.Linear(self.input_dim, self.inter_fc_prediction_dim)
        self.fc2 = nn.Linear(self.inter_fc_prediction_dim, self.output_dim)

    def forward(self, combined_inputs):
        out = torch.relu(F.dropout(self.fc1(combined_inputs), self.dropout))
        out = torch.relu(self.fc2(out))
        # out = torch.relu(self.fc1(F.dropout(combined_inputs, self.dropout)))

        if self.output_dim == 1:
            out = torch.sigmoid(out)

        return out


class MultitaskModel(nn.Module):
    """
    A model combining base + output layers for multitask learning
    """

    def __init__(
        self, params, num_embeddings=None, pretrained_embeddings=None, multi_dataset=True
    ):
        super(MultitaskModel, self).__init__()
        # save whether there are multiple datasets
        # if so, assumes each dataset has its own task
        self.multi_dataset = multi_dataset

        # let the network know if it's only using text features
        self.text_only = params.text_only

        # # set base of model
        # comment this out and uncomment the below to try late fusion model
        if params.audio_only is True:
            self.base = EarlyFusionAcousticOnlyModel(
                params, num_embeddings, pretrained_embeddings
            )
        elif params.text_only is False:
            self.base = EarlyFusionMultimodalModel(
                params, num_embeddings, pretrained_embeddings
            )
        else:
            self.base = EarlyFusionTextOnlyModel(
                params, num_embeddings, pretrained_embeddings
            )
            # self.base = TextOnlyRNN(
            #     params, num_embeddings, pretrained_embeddings
            # )

        # uncomment this and comment the above to try the late fusion model
        # self.base = LateFusionMultimodalModel(
        #     params, num_embeddings, pretrained_embeddings
        # )

        # set output layers
        self.task_0_predictor = PredictionLayer(params, params.output_0_dim)
        self.task_1_predictor = PredictionLayer(params, params.output_1_dim)
        self.task_2_predictor = PredictionLayer(params, params.output_2_dim)
        self.task_3_predictor = PredictionLayer(params, params.output_3_dim)

    def forward(
        self,
        acoustic_input,
        text_input,
        speaker_input=None,
        length_input=None,
        acoustic_len_input=None,
        gender_input=None,
        task_num=0
    ):
        # call forward on base model
        if self.text_only:
            final_base_layer = self.base(
                acoustic_input,
                text_input,
                speaker_input=speaker_input,
                length_input=length_input,
                gender_input=gender_input,
            )
        else:
            final_base_layer = self.base(
                acoustic_input,
                text_input,
                speaker_input=speaker_input,
                length_input=length_input,
                acoustic_len_input=acoustic_len_input,
                gender_input=gender_input,
            )

        task_0_out = None
        task_1_out = None
        task_2_out = None
        task_3_out = None

        if not self.multi_dataset:
            task_0_out = self.task_0_predictor(final_base_layer)
            task_1_out = self.task_1_predictor(final_base_layer)
            task_2_out = self.task_2_predictor(final_base_layer)
            task_3_out = self.task_3_predictor(final_base_layer)
        else:
            if task_num == 0:
                task_0_out = self.task_0_predictor(final_base_layer)
            elif task_num == 1:
                task_1_out = self.task_1_predictor(final_base_layer)
            elif task_num == 2:
                task_2_out = self.task_2_predictor(final_base_layer)
            elif task_num == 3:
                task_3_out = self.task_3_predictor(final_base_layer)
            else:
                sys.exit(f"Task {task_num} not defined")

        return task_0_out, task_1_out, task_2_out, task_3_out


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
            self.acoustic_fc1 = nn.Linear(params.audio_dim, params.acoustic_gru_hidden_dim)
            self.acoustic_fc2 = nn.Linear(params.acoustic_gru_hidden_dim, params.output_dim)

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
            feats = torch.relu(F.dropout(self.acoustic_fc1(acoustic_input), self.dropout))
            encoded_acoustic = torch.relu(F.dropout(self.acoustic_fc2(feats), self.dropout))

        return encoded_acoustic


class TextPlusPredictionLayer(nn.Module):
    """
    Contains text processing + a prediction layer
    Needs the output of an acoustic only layer
    """
    def __init__(self, params, out_dim, num_embeddings=None, pretrained_embeddings=None, multi_dataset=True):
        super(TextPlusPredictionLayer, self).__init__()

        # specify out_dim explicity so we can do multiple tasks at once
        self.output_dim = out_dim

        # save whether there are multiple datasets
        self.multi_dataset = multi_dataset

        # if we feed text through additional layer(s)
        self.text_rnn = nn.LSTM(
            input_size=params.text_dim + params.short_emb_dim,
            hidden_size=params.text_gru_hidden_dim,
            num_layers=params.num_gru_layers,
            batch_first=True,
            bidirectional=True,
        )

        # input text + acoustic + speaker
        self.text_dim = params.text_dim
        self.num_embeddings = num_embeddings
        self.text_gru_hidden_dim = params.text_gru_hidden_dim

        # get number of output dims
        self.acoustic_input_dim = params.output_dim

        # set the size of the input into the fc layers
        # if params.use_speaker:
        #     self.fc_input_dim = self.acoustic_input_dim + params.text_gru_hidden_dim + params.speaker_emb_dim
        # elif params.use_gender:
        #     self.fc_input_dim = self.acoustic_input_dim + params.text_gru_hidden_dim + params.gender_emb_dim

        self.fc_input_dim = self.acoustic_input_dim

        # set number of layers and dropout
        self.dropout = params.dropout

        # initialize word embeddings
        self.embedding = nn.Embedding(
            num_embeddings, self.text_dim, _weight=pretrained_embeddings
        )
        self.short_embedding = nn.Embedding(num_embeddings, params.short_emb_dim)
        # self.text_batch_norm = nn.BatchNorm1d(self.text_dim + params.short_emb_dim)

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
        gender_input=None,
        ):
        # here, acoustic_input is the output of the acoustic layers

        # # using pretrained embeddings, so detach to not update weights
        # # embs: (batch_size, seq_len, emb_dim)
        # embs = F.dropout(self.embedding(text_input), 0.1).detach()
        # short_embs = F.dropout(self.short_embedding(text_input), 0.1)
        #
        # all_embs = torch.cat((embs, short_embs), dim=2)
        #
        # packed = nn.utils.rnn.pack_padded_sequence(
        #     all_embs, length_input, batch_first=True, enforce_sorted=False
        # )
        #
        # # feed embeddings through GRU
        # packed_output, (hidden, cell) = self.text_rnn(packed)
        # encoded_text = F.dropout(hidden[-1], 0.3)
        #
        # # get speaker embeddings, if needed
        # if speaker_input is not None:
        #     speaker_embs = self.speaker_embedding(speaker_input).squeeze(dim=1)
        #     encoded_text = torch.cat((encoded_text, speaker_embs), dim=1)
        # elif gender_input is not None:
        #     gend_embs = self.gender_embedding(gender_input)
        #     encoded_text = torch.cat((encoded_text, gend_embs), dim=1)
        #
        # # combine all
        # encoded_data = torch.cat((encoded_text, acoustic_input), dim=1)
        #
        encoded_data = acoustic_input

        intermediate = torch.tanh(F.dropout(self.fc1(encoded_data), self.dropout))
        predictions = torch.relu(self.fc2(intermediate))

        if self.output_dim == 1:
            predictions = torch.sigmoid(predictions)

        # return the output
        return predictions


class MultitaskAcousticShared(nn.Module):
    """
    A model combining base + output layers for multitask learning
    """

    def __init__(
        self, params, num_embeddings=None, pretrained_embeddings=None, multi_dataset=True
    ):
        super(MultitaskAcousticShared, self).__init__()
        # save whether there are multiple datasets
        # if so, assumes each dataset has its own task
        self.multi_dataset = multi_dataset

        # # set base of model
        # comment this out and uncomment the below to try late fusion model
        self.acoustic_base = AcousticOnlyForMultitask(
            params, multi_dataset, use_rnn=(not (params.avgd_acoustic or params.add_avging))
        )

        # set output layers
        self.task_0_predictor = TextPlusPredictionLayer(params, params.output_0_dim,
                                                        num_embeddings=num_embeddings,
                                                        pretrained_embeddings=pretrained_embeddings,
                                                        multi_dataset=multi_dataset)
        self.task_1_predictor = TextPlusPredictionLayer(params, params.output_1_dim,
                                                        num_embeddings=num_embeddings,
                                                        pretrained_embeddings=pretrained_embeddings,
                                                        multi_dataset=multi_dataset)
        self.task_2_predictor = TextPlusPredictionLayer(params, params.output_2_dim,
                                                        num_embeddings=num_embeddings,
                                                        pretrained_embeddings=pretrained_embeddings,
                                                        multi_dataset=multi_dataset)
        self.task_3_predictor = TextPlusPredictionLayer(params, params.output_3_dim,
                                                        num_embeddings=num_embeddings,
                                                        pretrained_embeddings=pretrained_embeddings,
                                                        multi_dataset=multi_dataset)

    def forward(
        self,
        acoustic_input,
        text_input,
        speaker_input=None,
        length_input=None,
        acoustic_len_input=None,
        gender_input=None,
        task_num=0
    ):
        # call forward on base model
        final_base_layer = self.acoustic_base(
            acoustic_input,
            length_input=acoustic_len_input,
        )

        task_0_out = None
        task_1_out = None
        task_2_out = None
        task_3_out = None

        if not self.multi_dataset:
            task_0_out = self.task_0_predictor(final_base_layer, text_input, speaker_input, length_input, gender_input)
            task_1_out = self.task_1_predictor(final_base_layer, text_input, speaker_input, length_input, gender_input)
            task_2_out = self.task_2_predictor(final_base_layer, text_input, speaker_input, length_input, gender_input)
            task_3_out = self.task_3_predictor(final_base_layer, text_input, speaker_input, length_input, gender_input)
        else:
            if task_num == 0:
                task_0_out = self.task_0_predictor(final_base_layer, text_input, speaker_input, length_input, gender_input)
            elif task_num == 1:
                task_1_out = self.task_1_predictor(final_base_layer, text_input, speaker_input, length_input, gender_input)
            elif task_num == 2:
                task_2_out = self.task_2_predictor(final_base_layer, text_input, speaker_input, length_input, gender_input)
            elif task_num == 3:
                task_3_out = self.task_3_predictor(final_base_layer, text_input, speaker_input, length_input, gender_input)
            else:
                sys.exit(f"Task {task_num} not defined")

        return task_0_out, task_1_out, task_2_out, task_3_out


class MultitaskAcousticOnly(nn.Module):
    """
    A model combining base + output layers for multitask learning
    """

    def __init__(
        self, params, multi_dataset=True
    ):
        super(MultitaskAcousticOnly, self).__init__()
        # save whether there are multiple datasets
        # if so, assumes each dataset has its own task
        self.multi_dataset = multi_dataset

        # # set base of model
        # comment this out and uncomment the below to try late fusion model
        self.acoustic_base = AcousticOnlyForMultitask(
            params, multi_dataset, use_rnn=(not (params.avgd_acoustic or params.add_avging))
        )

        # set output layers
        self.task_0_predictor = PredictionLayer(params, params.output_0_dim)
        self.task_1_predictor = PredictionLayer(params, params.output_1_dim)
        self.task_2_predictor = PredictionLayer(params, params.output_2_dim)
        self.task_3_predictor = PredictionLayer(params, params.output_3_dim)

    def forward(
        self,
        acoustic_input,
        speaker_input=None,
        acoustic_len_input=None,
        gender_input=None,
        task_num=0
    ):
        # call forward on base model
        final_base_layer = self.acoustic_base(
            acoustic_input,
            length_input=acoustic_len_input,
        )

        if gender_input is not None:
            gender_embs = self.gender_embedding(gender_input)

        inputs = torch.cat((final_base_layer, gender_embs), 1)

        task_0_out = None
        task_1_out = None
        task_2_out = None
        task_3_out = None

        if not self.multi_dataset:
            task_0_out = self.task_0_predictor(inputs)
            task_1_out = self.task_1_predictor(inputs)
            task_2_out = self.task_2_predictor(inputs)
            task_3_out = self.task_3_predictor(inputs)
        else:
            if task_num == 0:
                task_0_out = self.task_0_predictor(inputs)
            elif task_num == 1:
                task_1_out = self.task_1_predictor(inputs)
            elif task_num == 2:
                task_2_out = self.task_2_predictor(inputs)
            elif task_num == 3:
                task_3_out = self.task_3_predictor(inputs)
            else:
                sys.exit(f"Task {task_num} not defined")

        return task_0_out, task_1_out, task_2_out, task_3_out

class EarlyFusionAcousticOnlyModel(nn.Module):
    """
    An encoder to take a sequence of inputs and produce a sequence of intermediate representations
    Can include convolutions over text input and/or acoustic input--BUT NOT TOGETHER bc MELD isn't
    aligned at the word-level
    """

    def __init__(self, params, num_embeddings=None, pretrained_embeddings=None):
        super(EarlyFusionAcousticOnlyModel, self).__init__()
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


class EarlyFusionTextOnlyModel(nn.Module):
    """
    An encoder to take a sequence of inputs and produce a sequence of intermediate representations
    Can include convolutions over text input and/or acoustic input--BUT NOT TOGETHER bc MELD isn't
    aligned at the word-level
    """

    def __init__(self, params, num_embeddings=None, pretrained_embeddings=None):
        super(EarlyFusionTextOnlyModel, self).__init__()
        # input text + acoustic + speaker
        self.text_dim = params.text_dim
        self.num_embeddings = num_embeddings
        self.num_speakers = params.num_speakers
        self.text_gru_hidden_dim = params.text_gru_hidden_dim

        # get number of output dims
        self.out_dims = params.output_dim

        # if we feed text through additional layer(s)
        self.text_rnn = nn.LSTM(
            input_size=params.text_dim + params.short_emb_dim,
            hidden_size=params.text_gru_hidden_dim,
            num_layers=params.num_gru_layers,
            batch_first=True,
            bidirectional=True,
        )

        # set the size of the input into the fc layers
        # if params.avgd_acoustic or params.add_avging:
        self.fc_input_dim = params.text_gru_hidden_dim

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

        # initialize word embeddings
        self.embedding = nn.Embedding(
            num_embeddings, self.text_dim, _weight=pretrained_embeddings
        )
        self.short_embedding = nn.Embedding(num_embeddings, params.short_emb_dim)

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
        # using pretrained embeddings, so detach to not update weights
        # embs: (batch_size, seq_len, emb_dim)
        embs = F.dropout(self.embedding(text_input), 0.1).detach()

        short_embs = F.dropout(self.short_embedding(text_input), 0.1)

        all_embs = torch.cat((embs, short_embs), dim=2)

        # get speaker embeddings, if needed
        if speaker_input is not None:
            speaker_embs = self.speaker_embedding(speaker_input).squeeze(dim=1)
        if gender_input is not None:
            gender_embs = self.gender_embedding(gender_input)

        packed = nn.utils.rnn.pack_padded_sequence(
            all_embs, length_input, batch_first=True, enforce_sorted=False
        )

        # feed embeddings through GRU
        packed_output, (hidden, cell) = self.text_rnn(packed)
        encoded_text = F.dropout(hidden[-1], 0.3)

        # combine modalities as required by architecture
        if speaker_input is not None:
            inputs = torch.cat((encoded_text, speaker_embs), 1)
        elif gender_input is not None:
            inputs = torch.cat((encoded_text, gender_embs), 1)
        else:
            inputs = encoded_text

        # use pooled, squeezed feats as input into fc layers
        output = torch.tanh(F.dropout(self.fc1(inputs), 0.5))

        if self.out_dims == 1:
            output = torch.sigmoid(output)

        # return the output
        return output


class MultitaskDuplicateInputModel(nn.Module):
    """
    A multimodal model that duplicates input modality tensors
    Inspired by:
    Frustratingly easy domain adaptation
    """

    def __init__(
        self, params, num_embeddings=None, pretrained_embeddings=None, multi_dataset=True,
            num_tasks=2
    ):
        super(MultitaskDuplicateInputModel, self).__init__()
        # save whether there are multiple datasets
        # if so, assumes each dataset has its own task
        self.multi_dataset = multi_dataset

        # # set base of model
        # comment this out and uncomment the below to try late fusion model
        self.base = MultimodalBaseDuplicateInput(
            params, num_embeddings, pretrained_embeddings, num_tasks
        )

        # set output layers
        self.task_0_predictor = PredictionLayer(params, params.output_0_dim)
        self.task_1_predictor = PredictionLayer(params, params.output_1_dim)
        self.task_2_predictor = PredictionLayer(params, params.output_2_dim)

    def forward(
        self,
        acoustic_input,
        text_input,
        speaker_input=None,
        length_input=None,
        acoustic_len_input=None,
        gender_input=None,
        task_num=0
    ):
        # call forward on base model
        final_base_layer = self.base(
            acoustic_input,
            text_input,
            length_input=length_input,
            gender_input=gender_input,
            task_num=task_num
        )

        task_0_out = None
        task_1_out = None
        task_2_out = None

        if not self.multi_dataset:
            task_0_out = self.task_0_predictor(final_base_layer)
            task_1_out = self.task_1_predictor(final_base_layer)
            task_2_out = self.task_2_predictor(final_base_layer)
        else:
            if task_num == 0:
                task_0_out = self.task_0_predictor(final_base_layer)
            elif task_num == 1:
                task_1_out = self.task_1_predictor(final_base_layer)
            elif task_num == 2:
                task_2_out = self.task_2_predictor(final_base_layer)
            else:
                sys.exit(f"Task {task_num} not defined")

        return task_0_out, task_1_out, task_2_out


class MultimodalBaseDuplicateInput(nn.Module):
    """
    A multimodal model that duplicates input modality tensors
    Inspired by:
    Frustratingly easy domain adaptation
    """

    def __init__(self, params, num_embeddings=None, pretrained_embeddings=None,
                 num_tasks=2):
        super(MultimodalBaseDuplicateInput, self).__init__()
        # get the number of duplicates
        self.mult_size = num_tasks + 1

        # input text + acoustic + speaker
        self.text_dim = params.text_dim * self.mult_size
        self.short_dim = params.short_emb_dim * self.mult_size
        self.audio_dim = params.audio_dim * self.mult_size

        self.num_embeddings = num_embeddings
        self.num_speakers = params.num_speakers
        self.text_gru_hidden_dim = params.text_gru_hidden_dim

        # get number of output dims
        self.out_dims = params.output_dim

        # if we feed text through additional layer(s)
        self.text_rnn = nn.LSTM(
            input_size=self.text_dim + self.short_dim,
            hidden_size=params.text_gru_hidden_dim,
            num_layers=params.num_gru_layers,
            batch_first=True,
            bidirectional=True,
        )

        # set the size of acoustic fc layers
        self.acoustic_fc_1 = nn.Linear(self.audio_dim, 50)
        self.acoustic_fc_2 = nn.Linear(50, params.audio_dim)

        # set the size of the input into the multimodal fc layers
        self.fc_input_dim = params.text_gru_hidden_dim + params.audio_dim

        if params.use_gender:
            self.fc_input_dim = self.fc_input_dim + params.gender_emb_dim

        # set number of classes
        self.output_dim = params.output_dim

        # set number of layers and dropout
        self.dropout = params.dropout

        # initialize word embeddings
        self.embedding = nn.Embedding(
            num_embeddings, params.text_dim, _weight=pretrained_embeddings
        )
        self.short_embedding = nn.Embedding(num_embeddings, params.short_emb_dim)

        # initialize speaker embeddings
        self.speaker_embedding = nn.Embedding(
            params.num_speakers, params.speaker_emb_dim
        )

        self.gender_embedding = nn.Embedding(3, params.gender_emb_dim)

        # initialize fully connected layers
        self.fc1 = nn.Linear(self.fc_input_dim, params.fc_hidden_dim)

    def forward(
        self,
        acoustic_input,
        text_input,
        length_input=None,
        gender_input=None,
        task_num=0,
    ):
        # using pretrained embeddings, so detach to not update weights
        # embs: (batch_size, seq_len, emb_dim)
        embs = self.embedding(text_input).detach()
        temp_embs = torch.clone(embs).detach()

        # do the same with trainable short embeddings
        short_embs = self.short_embedding(text_input)

        # add a tensor for each task; if not the tensor for this task,
        # fill it with zeros; otherwise, just copy the tensor
        for num in range(1, self.mult_size):
            # don't want to detach so recalculating short embs here
            # for safety; may not be necessary
            temp_short_embs = self.short_embedding(text_input)

            if num != task_num + 1:
                temp_embs = torch.zeros_like(temp_embs).detach()
                temp_short_embs = torch.zeros_like(temp_short_embs)

            embs = torch.cat((embs, temp_embs), dim=2)
            short_embs = torch.cat((short_embs, temp_short_embs), dim=2)

        # add dropout
        embs = F.dropout(embs, 0.1)
        short_embs = F.dropout(short_embs, 0.1)

        # concatenate short and regular embeddings
        all_embs = torch.cat((embs, short_embs), dim=2)

        # get gender embeddings, if needed
        if gender_input is not None:
            gender_embs = self.gender_embedding(gender_input)

            for num in range(1, self.mult_size):
                temp_gend_embs = self.gender_embedding(gender_input)

                if num != task_num + 1:
                    temp_gend_embs = torch.zeros_like(temp_gend_embs)

                gender_embs = torch.cat((gender_embs, temp_gend_embs), dim=-1)

        # pack embeddings for RNN
        packed = nn.utils.rnn.pack_padded_sequence(
            all_embs, length_input, batch_first=True, enforce_sorted=False
        )

        # feed embeddings through RNN
        packed_output, (hidden, cell) = self.text_rnn(packed)
        encoded_text = F.dropout(hidden[-1], 0.3)

        if len(acoustic_input.shape) > 2:
            acoustic_input = acoustic_input.squeeze()

        temp_acoustic_input = torch.clone(acoustic_input).detach()

        for num in range(1, self.mult_size):
            # i THINK detach here just keeps differentiation separate from the original
            # could check both ways to verify
            if num != task_num + 1:
                temp_acoustic_input = torch.zeros_like(temp_acoustic_input)

            acoustic_input = torch.cat((acoustic_input, temp_acoustic_input), dim=1)

        int_acoustic = torch.tanh(
            F.dropout(self.acoustic_fc_1(acoustic_input), self.dropout)
        )
        encoded_acoustic = torch.tanh(
            F.dropout(self.acoustic_fc_2(int_acoustic), self.dropout)
        )

        # combine modalities as required by architecture
        if gender_input is not None:
            inputs = torch.cat((encoded_acoustic, encoded_text, gender_embs), 1)
        else:
            inputs = torch.cat((encoded_acoustic, encoded_text), 1)

        # use pooled, squeezed feats as input into fc layers
        output = torch.tanh(F.dropout(self.fc1(inputs), 0.5))

        if self.out_dims == 1:
            output = torch.sigmoid(output)

        # return the output
        return output

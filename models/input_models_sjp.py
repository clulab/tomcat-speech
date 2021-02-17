# the models used for multimodal, multitask classification
import sys
from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attn_models import *

class EmbraceNet(nn.Module):
    def __init__(self,
                 input_size_list,
                 params):
        """
        https://github.com/idearibosome/embracenet
        Initialize an EmbraceNet Module
        @param device: torch.device object (cpu / gpu)
        @param input_size_list: list of input sizes [num_feat, shape_input] ("c" in the paper)
        @param embracement_size: the length of the output of the embracement layer
        @param bypass_docking:
            bypass docking step. If True, the shape of input_data should be [batch_size, embracement_size]
        """
        super(EmbraceNet, self).__init__()

        self.device = device
        self.input_size_list = input_size_list
        self.embracement_size = params.embracement_size
        self.bypass_docking = params.bypass_docking
        self.availabilities = params.availabilities
        self.selection_probabilities = params.selection_probabilities

        if not self.bypass_docking:
            for i, input_size in enumerate(input_size_list):
                setattr(self, 'docking_%d' % i, nn.Linear(input_size, self.embracement_size))

    def forward(self, input_list):
        """
        Forward input data to the EmbraceNet module
        @param input_list: A list of input data
        @param availabilities: 2D tensor of shape [batch_size, num_feats],
                               which represents the availability of data for each modality. If None, it assumes that
                               data of all features are available
        @param selection_probabilities: 2D tensor of shape [batch_size, num_feats],
                                      which represents probabilities that output of each docking layer will be
                                      selected ("p" in the paper). If None, same probability will be used.

        @return: 2D tensor of shape [batch_size, embracement_size]
        """

        # check input_data
        assert len(input_list) == len(self.input_size_list)
        num_feats = len(input_list)
        batch_size = input_list[0].shape[0]

        # docking layer
        docking_output_list = []
        if (self.bypass_docking):
            docking_output_list = input_list
        else:
            for i, input_data in enumerate(input_list):
                x = getattr(self, 'docking_%d' % i)(input_data)
                x = nn.functional.relu(x)
                docking_output_list.append(x)

        # check availabilities
        if (self.availabilities is None):
            availabilities = torch.ones(batch_size, len(input_list), dtype=torch.float, device=self.device)
        else:
            availabilities = self.availabilities.float()

        # adjust selection probabilities
        if (self.selection_probabilities is None):
            selection_probabilities = torch.ones(batch_size, len(input_list), dtype=torch.float, device=self.device)
        selection_probabilities = torch.mul(selection_probabilities, availabilities)

        probability_sum = torch.sum(selection_probabilities, dim=-1, keepdim=True)
        selection_probabilities = torch.div(selection_probabilities, probability_sum)

        # stack docking outputs
        docking_output_stack = torch.stack(docking_output_list,
                                           dim=-1)  # [batch_size, embracement_size, num_modalities]

        # embrace
        feature_indices = torch.multinomial(selection_probabilities, num_samples=self.embracement_size,
                                            replacement=True)
        feature_toggles = nn.functional.one_hot(feature_indices,
                                                num_classes=num_feats).float()  # [batch_size, embracement_size, num_feat]

        embracement_output_stack = torch.mul(docking_output_stack, feature_toggles)
        embracement_output = torch.sum(embracement_output_stack, dim=-1)  # [batch_size, embracement_size]

        return embracement_output

class AudioOnlyRNN(nn.Module):
    """
    An RNN used with RAVDESS, where primary information comes from audio_train
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
            bidirectional=False
        )

        # acoustic batch normalization
        self.acoustic_batch_norm = nn.BatchNorm1d(params.audio_dim)

        self.acoustic_fc_1 = nn.Linear(params.acoustic_gru_hidden_dim, 50)
        self.acoustic_fc_2 = nn.Linear(50, params.audio_dim)

        # dimension of input into final fc layers
        self.fc_input_dim = params.acoustic_gru_hidden_dim
        # self.fc_input_dim = params.audio_dim

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
            acoustic_input, acoustic_len_input, batch_first=True, enforce_sorted=False
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

        output = inputs
        # output = torch.tanh(F.dropout(self.fc1(inputs), 0.5))
        # output = torch.relu(self.fc2(output))

        if self.output_dim == 1:
            output = torch.sigmoid(output)

        # return the output
        return output


class AudioCNN(nn.Module):
    """
    A CNN with input channels with different kernel size operating over input
    """

    def __init__(self, params):
        super(AudioCNN, self).__init__()
        # input dimensions
        self.audio_dim = params.audio_dim
        self.in_channels = self.audio_dim[0]
        self.num_features = self.audio_dim[1]

        # number of classes
        self.output_dim = params.output_dim

        # self.num_cnn_layers = params.num_cnn_layers
        self.dropout = params.dropout

        self.conv1 = nn.Conv2d(self.in_channels, out_channels=128, kernel_size=(3, 3), padding=1)
        self.conv1_bn = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1)
        self.conv2_bn = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(kernel_size=(5, 5))
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1)
        self.conv3_bn = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(kernel_size=(5, 6))
        self.conv4 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), padding=1)
        self.conv4_bn = nn.BatchNorm2d(1024)
        self.pool4 = nn.MaxPool2d(kernel_size=(4, 6))
        # self.conv5 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=(3, 3), padding=1)
        # self.conv5_bn = nn.BatchNorm2d(2048)
        # self.pool5 = nn.MaxPool2d(kernel_size=(4, 4))
        # self.fc1 = nn.Linear(in_features=1024, out_features=self.output_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(in_features=1024, out_features=self.output_dim)
        # self.fc2 = nn.Linear(in_features=512, out_features=256)
        # self.fc3 = nn.Linear(in_features=256, out_features=self.output_dim)

    def forward(self, acoustic_input):
        self.inputs = acoustic_input
        # feed data into convolutional layers
        x = self.conv1(self.inputs)
        # print("conv1: ", x.size())
        x = self.pool1(F.elu(self.conv1_bn(x)))
        # print("pool1: ", x.size())
        x = self.pool2(F.elu(self.conv2_bn(self.conv2(x))))
        # print("conv2/pool2: ", x.size())
        x = self.pool3(F.elu(self.conv3_bn(self.conv3(x))))
        # print("conv3/pool3: ", x.size())
        x = self.pool4(F.elu(self.conv4_bn(self.conv4(x))))
        # print("conv4/pool4: ", x.size())
        # x = self.pool5(F.elu(self.conv5_bn(self.conv5(x))))
        # print("conv5/pool5: ", x.size())
        x = x.view(-1, 1024)
        # get predictions
        # output = torch.sigmoid(self.fc1(x))
        # output = nn.Softmax(self.fc1(x))
        output = self.dropout(x)
        # output = F.relu(self.fc1(x))
        # output = self.dropout(output)
        # output = F.relu(self.fc2(output))
        # output = self.dropout(output)
        # output = F.relu(self.fc3(output))
        return output

class EarlyFusionMultimodalModel(nn.Module):
    """
    An encoder to take a sequence of inputs and produce a sequence of intermediate representations
    Can include convolutions over text input and/or acoustic input--BUT NOT TOGETHER bc MELD isn't
    aligned at the word-level
    """

    def __init__(self, params, num_embeddings=None, pretrained_embeddings=None, acoustic_cnn=False):
        super(EarlyFusionMultimodalModel, self).__init__()
        # input text + acoustic + speaker
        self.text_dim = params.text_dim
        self.audio_dim = params.audio_dim
        self.num_embeddings = num_embeddings
        self.num_speakers = params.num_speakers
        self.text_gru_hidden_dim = params.text_gru_hidden_dim
        self.acoustic_cnn = acoustic_cnn

        # get number of output dims
        self.out_dims = params.output_dim

        # if we feed text through additional layer(s)
        # self.text_output_dim = params.text_output_dim
        self.text_rnn = nn.LSTM(
            input_size=params.text_dim + params.short_emb_dim,
            hidden_size=params.text_gru_hidden_dim,
            num_layers=params.num_gru_layers,
            batch_first=True,
            bidirectional=True,
        )

        if self.acoustic_cnn:
            self.acoustic_cnn = nn.Sequential(
                nn.Conv2d(1, out_channels=128, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(128),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(3, 5)),
                nn.Conv2d(128, out_channels=256, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(256),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(3, 5)),
                nn.Conv2d(256, out_channels=512, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(512),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(3, 5)),
                nn.Conv2d(512, out_channels=1024, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(1024),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(4, 3))
            )

        else:
            self.acoustic_rnn = nn.LSTM(
                input_size=params.audio_dim,
                hidden_size=params.acoustic_gru_hidden_dim,
                num_layers=params.num_gru_layers,
                batch_first=True,
                bidirectional=True,
            )

        # set the size of the input into the fc layers
        # if params.avgd_acoustic or params.add_avging:
        self.fc_input_dim = params.text_gru_hidden_dim + params.audio_dim
            # self.fc_input_dim = params.text_gru_hidden_dim + 20

        # else:
        #     self.fc_input_dim = (
        #         params.text_gru_hidden_dim + params.acoustic_gru_hidden_dim
        #     )

        if self.acoustic_cnn:
            self.acoustic_fc_1 = nn.Linear(1024, 100)
        elif params.add_avging is False and params.avgd_acoustic is False:
            # self.acoustic_fc_1 = nn.Linear(params.audio_dim, 100)
            self.acoustic_fc_1 = nn.Linear(params.fc_hidden_dim, 100)
        else:
            self.acoustic_fc_1 = nn.Linear(params.audio_dim, 100)
        # self.acoustic_fc_2 = nn.Linear(100, 20)
        self.acoustic_fc_2 = nn.Linear(100, params.audio_dim)

        if params.use_speaker:
            self.fc_input_dim = self.fc_input_dim + params.speaker_emb_dim
        elif params.use_gender:
            self.fc_input_dim = self.fc_input_dim + params.gender_emb_dim

        # print(self.fc_input_dim)
        # self.fc_input_dim = params.text_output_dim

        # set number of classes
        self.output_dim = params.output_dim

        # set number of layers and dropout
        self.dropout = params.dropout

        # initialize word embeddings
        self.embedding = nn.Embedding(
            num_embeddings, self.text_dim, _weight=pretrained_embeddings
        )
        self.short_embedding = nn.Embedding(num_embeddings, params.short_emb_dim)
        # self.embedding = nn.Embedding(num_embeddings, self.text_dim)
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
        # self.fc1 = nn.Linear(self.fc_input_dim, params.output_dim)
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
            # speaker_embs = self.speaker_batch_norm(speaker_embs)
        if gender_input is not None:
            gender_embs = self.gender_embedding(gender_input)

        packed = nn.utils.rnn.pack_padded_sequence(
            all_embs, length_input, batch_first=True, enforce_sorted=False
        )

        # feed embeddings through GRU
        packed_output, (hidden, cell) = self.text_rnn(packed)
        encoded_text = F.dropout(hidden[-1], 0.3)

        if self.acoustic_cnn:
            output = self.acoustic_cnn(acoustic_input)
            output = output.view(-1, 1024)
            encoded_acoustic = torch.relu(F.dropout(self.acoustic_fc_1(output), self.dropout))
            encoded_acoustic = torch.tanh(F.dropout(self.acoustic_fc_2(encoded_acoustic), self.dropout))

        else:
            if acoustic_len_input is not None:
                # print(acoustic_input.shape)
                # acoustic_input = self.acoustic_batch_norm(acoustic_input.permute(0, 2, 1))
                # print(acoustic_input.shape)
                # acoustic_input = acoustic_input.permute(0, 2, 1)
                packed_acoustic = nn.utils.rnn.pack_padded_sequence(
                    acoustic_input,
                    # acoustic_len_input,
                    acoustic_len_input.clamp(max=1500),
                    batch_first=True,
                    enforce_sorted=False,
                )
                (
                    packed_acoustic_output,
                    (acoustic_hidden, acoustic_cell),
                ) = self.acoustic_rnn(packed_acoustic)
                encoded_acoustic = F.dropout(acoustic_hidden[-1], self.dropout)
                # encoded_acoustic = acoustic_hidden[-1]

            else:
                # print(acoustic_input.shape)
                if len(acoustic_input.shape) > 2:
              # self.acoustic_fc_2 = nn.Linear(100, 20)  F.dropout(self.acoustic_fc_2(encoded_acoustic), self.dropout)
            )
            # print(encoded_acoustic.shape)
            # encoded_acoustic = self.acoustic_batch_norm(encoded_acoustic)

            # inputs = encoded_text
            # print(encoded_acoustic.shape)

        # combine modalities as required by architecture
        # inputs = torch.cat((acoustic_input, encoded_text), 1)
        if speaker_input is not None:
            inputs = torch.cat((encoded_acoustic, encoded_text, speaker_embs), 1)
        elif gender_input is not None:
            inputs = torch.cat((encoded_acoustic, encoded_text, gender_embs), 1)
        else:
            inputs = torch.cat((encoded_acoustic, encoded_text), 1)

        # use pooled, squeezed feats as input into fc layers
        output = torch.tanh(F.dropout(self.fc1(inputs), 0.5))
        # output = torch.tanh(self.fc1(inputs))
        # output = self.interfc_batch_norm(output)
        # todo: abstract this so it's only calculated if not multitask
        # output = torch.relu(self.fc2(output))
        # output = F.softmax(output, dim=1)
        # output = torch.tanh(self.fc1(inputs))

        if self.out_dims == 1:
            output = torch.sigmoid(output)
        # return the output
        # print(f"The output of sub-network is:\n{output}")
        return output

class EarlyFusionEmbraceModel(nn.Module):
    """
    An encoder to take a sequence of inputs and produce a sequence of intermediate representations
    Can include convolutions over text input and/or acoustic input--BUT NOT TOGETHER bc MELD isn't
    aligned at the word-level
    """

    def __init__(self, params, num_embeddings=None, pretrained_embeddings=None, acoustic_cnn=False):
        super(EarlyFusionEmbraceModel, self).__init__()
        # input text + acoustic + speaker
        self.text_dim = params.text_dim
        self.audio_dim = params.audio_dim
        self.num_embeddings = num_embeddings
        self.num_speakers = params.num_speakers
        self.text_gru_hidden_dim = params.text_gru_hidden_dim
        self.acoustic_cnn = acoustic_cnn

        # get number of output dims
        self.out_dims = params.output_dim

        # if we feed text through additional layer(s)
        # self.text_output_dim = params.text_output_dim
        self.text_rnn = nn.LSTM(
            input_size=params.text_dim + params.short_emb_dim,
            hidden_size=params.text_gru_hidden_dim,
            num_layers=params.num_gru_layers,
            batch_first=True,
            bidirectional=True,
        )

        if self.acoustic_cnn:
            self.acoustic_cnn = nn.Sequential(
                nn.Conv2d(1, out_channels=128, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(128),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(3, 5)),
                nn.Conv2d(128, out_channels=256, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(256),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(3, 5)),
                nn.Conv2d(256, out_channels=512, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(512),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(3, 5)),
                nn.Conv2d(512, out_channels=1024, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(1024),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(4, 3))
            )

        else:
            self.acoustic_rnn = nn.LSTM(
                input_size=params.audio_dim,
                hidden_size=params.acoustic_gru_hidden_dim,
                num_layers=params.num_gru_layers,
                batch_first=True,
                bidirectional=True,
            )

        # set the size of the input into the fc layers
        # if params.avgd_acoustic or params.add_avging:
        self.fc_input_dim = params.text_gru_hidden_dim + params.audio_dim
            # self.fc_input_dim = params.text_gru_hidden_dim + 20

        # else:
        #     self.fc_input_dim = (
        #         params.text_gru_hidden_dim + params.acoustic_gru_hidden_dim
        #     )

        if self.acoustic_cnn:
            self.acoustic_fc_1 = nn.Linear(1024, 100)
        elif params.add_avging is False and params.avgd_acoustic is False:
            # self.acoustic_fc_1 = nn.Linear(params.audio_dim, 100)
            self.acoustic_fc_1 = nn.Linear(params.fc_hidden_dim, 100)
        else:
            self.acoustic_fc_1 = nn.Linear(params.audio_dim, 100)
        # self.acoustic_fc_2 = nn.Linear(100, 20)
        self.acoustic_fc_2 = nn.Linear(100, params.audio_dim)

        # if params.use_speaker:
        #     self.fc_input_dim = self.fc_input_dim + params.speaker_emb_dim
        # elif params.use_gender:
            # self.fc_input_dim = self.fc_input_dim + params.gender_emb_dim

        # print(self.fc_input_dim)
        # self.fc_input_dim = params.text_output_dim

        # set number of classes
        self.output_dim = params.output_dim

        # set number of layers and dropout
        self.dropout = params.dropout

        # initialize word embeddings
        self.embedding = nn.Embedding(
            num_embeddings, self.text_dim, _weight=pretrained_embeddings
        )
        self.short_embedding = nn.Embedding(num_embeddings, params.short_emb_dim)
        # self.embedding = nn.Embedding(num_embeddings, self.text_dim)
        # self.text_batch_norm = nn.BatchNorm1d(self.text_dim + params.short_emb_dim)

        # initialize speaker embeddings
        self.speaker_embedding = nn.Embedding(
            params.num_speakers, params.speaker_emb_dim
        )

        # self.speaker_batch_norm = nn.BatchNorm1d(params.speaker_emb_dim)

        self.gender_embedding = nn.Embedding(3, params.gender_emb_dim)

        input_size = [params.text_gru_hidden_dim, params.audio_dim]

        self.multi_input_net = EmbraceNet(input_size_list = input_size, params=params)

        self.fc_input_dim = sum(input_size)


        # acoustic batch normalization
        # self.acoustic_batch_norm = nn.BatchNorm1d(params.audio_dim)
        # self.acoustic_unk_norm = nn.BatchNorm1d(params.audio_dim)
        # self.acoustic_female_norm = nn.BatchNorm1d(params.audio_dim)
        # self.acoustic_male_norm = nn.BatchNorm1d(params.audio_dim)

        # initialize fully connected layers
        # self.fc1 = nn.Linear(self.fc_input_dim, params.output_dim)
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
        packed = nn.utils.rnn.pack_padded_sequence(
            all_embs, length_input, batch_first=True, enforce_sorted=False
        )

        # feed embeddings through GRU
        packed_output, (hidden, cell) = self.text_rnn(packed)
        encoded_text = F.dropout(hidden[-1], 0.3)

        if self.acoustic_cnn:
            output = self.acoustic_cnn(acoustic_input)
            output = output.view(-1, 1024)
            encoded_acoustic = torch.relu(F.dropout(self.acoustic_fc_1(output), self.dropout))
            encoded_acoustic = torch.tanh(F.dropout(self.acoustic_fc_2(encoded_acoustic), self.dropout))

        else:
            if acoustic_len_input is not None:
                # print(acoustic_input.shape)
                # acoustic_input = self.acoustic_batch_norm(acoustic_input.permute(0, 2, 1))
                # print(acoustic_input.shape)
                # acoustic_input = acoustic_input.permute(0, 2, 1)
                packed_acoustic = nn.utils.rnn.pack_padded_sequence(
                    acoustic_input,
                    # acoustic_len_input,
                    acoustic_len_input.clamp(max=1500),
                    batch_first=True,
                    enforce_sorted=False,
                )
                (
                    packed_acoustic_output,
                    (acoustic_hidden, acoustic_cell),
                ) = self.acoustic_rnn(packed_acoustic)
                encoded_acoustic = F.dropout(acoustic_hidden[-1], self.dropout)
                # encoded_acoustic = acoustic_hidden[-1]

            else:
                # print(acoustic_input.shape)
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
            # print(encoded_acoustic.shape)
            # encoded_acoustic = self.acoustic_batch_norm(encoded_acoustic)

            # inputs = encoded_text
            # print(encoded_acoustic.shape)

        # combine modalities as required by architecture
        # inputs = torch.cat((acoustic_input, encoded_text), 1)
        # if speaker_input is not None:
        #     input_size = [encoded_acoustic.size()[1], encoded_text.size()[1], speaker_embs.size()[1]]
            
        # elif gender_input is not None:
        #     inputs = torch.cat((encoded_acoustic, encoded_text, gender_embs), 1)
        # else:
        #     inputs = torch.cat((encoded_acoustic, encoded_text), 1)

        inputs = self.multi_input_net([encoded_text, encoded_acoustic])

        # use pooled, squeezed feats as input into fc layers
        output = torch.tanh(F.dropout(self.fc1(inputs), 0.5))
        # output = torch.tanh(self.fc1(inputs))
        # output = self.interfc_batch_norm(output)
        # todo: abstract this so it's only calculated if not multitask
        # output = torch.relu(self.fc2(output))
        # output = F.softmax(output, dim=1)
        # output = torch.tanh(self.fc1(inputs))

        if self.out_dims == 1:
            output = torch.sigmoid(output)
        # return the output
        # print(f"The output of sub-network is:\n{output}")
        return output


class MultiAcousticModelEarly(nn.Module):
    def __init__(self, params):
        super(MultiAcousticModelEarly, self).__init__()
        # input dimensions
        self.attn_dim = params.attn_dim

        self.audio_dim = params.audio_dim

        self.acoustic_rnn = nn.LSTM(
            input_size=params.audio_dim,
            hidden_size=params.acoustic_gru_hidden_dim,
            num_layers=params.num_gru_layers,
            batch_first=True,
            bidirectional=params.bidirectional
        )

        encoder = Encoder(input_dim=params.attn_dim,
                          hidden_dim=params.acoustic_gru_hidden_dim,
                          num_gru_layers=params.num_gru_layers,
                          dropout=params.dropout,
                          bidirectional=params.bidirectional)

        attention_dim = params.acoustic_gru_hidden_dim if not params.bidirectional else 2 * params.acoustic_gru_hidden_dim
        attention = Attention(attention_dim, attention_dim, attention_dim)

        self.acoustic_model = AcousticAttn(
            encoder=encoder,
            attention=attention,
            hidden_dim=attention_dim,
            num_classes=params.output_dim
        )

        if params.bidirectional:
            self.audio_hidden_dim = params.acoustic_gru_hidden_dim
            self.attn_hidden_dim = params.acoustic_gru_hidden_dim * 2
            self.fc_input_dim = self.audio_hidden_dim + self.attn_hidden_dim
        else:
            self.fc_input_dim = 2 * params.acoustic_gru_hidden_dim

        self.fc1 = nn.Linear(self.fc_input_dim, params.fc_hidden_dim)
        self.fc2 = nn.Linear(params.fc_hidden_dim, 64)
        self.fc3 = nn.Linear(64, params.output_dim)

    def forward(self,
                audio_input,
                audio_length,
                acoustic_input,
                acoustic_length):

        audio_input = audio_input.transpose(1, 2)
        attn_output, _ = self.acoustic_model(audio_input, audio_length)
        # print("before: ", acoustic_input.size())
        acoustic_input = torch.squeeze(acoustic_input, 1).transpose(1, 2)
        # print("after: ", acoustic_input.size())
        packed = nn.utils.rnn.pack_padded_sequence(
            acoustic_input, acoustic_length, batch_first=True, enforce_sorted=False
        )

        packed_output, (hidden, cell) = self.acoustic_rnn(packed)

        rnn_output = F.dropout(hidden[-1], 0.3)

        # print("attn size: ", attn_output.size())
        # print("rnn size: ", rnn_output.size())

        inputs = torch.cat((attn_output, rnn_output), 1)
        # print("cat. input size: ", inputs.size())

        output = torch.tanh(F.dropout(self.fc1(inputs), 0.3))
        output = torch.tanh(F.dropout(self.fc2(output), 0.3))
        output = torch.tanh(self.fc3(output))

        return output


class MultiAcousticModelLate(nn.Module):
    def __init__(self, params):
        super(MultiAcousticModelLate, self).__init__()
        # input dimensions
        self.attn_dim = params.attn_dim

        self.audio_dim = params.audio_dim

        self.acoustic_rnn = nn.LSTM(
            input_size=params.audio_dim,
            hidden_size=params.acoustic_gru_hidden_dim,
            num_layers=params.num_gru_layers,
            batch_first=True,
            bidirectional=params.bidirectional
        )

        encoder = Encoder(input_dim=params.attn_dim,
                          hidden_dim=params.acoustic_gru_hidden_dim,
                          num_gru_layers=params.num_gru_layers,
                          dropout=params.dropout,
                          bidirectional=params.bidirectional)

        attention_dim = params.acoustic_gru_hidden_dim if not params.bidirectional else 2 * params.acoustic_gru_hidden_dim
        attention = Attention(attention_dim, attention_dim, attention_dim)

        self.acoustic_model = AcousticAttn(
            encoder=encoder,
            attention=attention,
            hidden_dim=attention_dim,
            num_classes=params.output_dim
        )

        if params.bidirectional:
            self.audio_hidden_dim = params.acoustic_gru_hidden_dim
            self.attn_hidden_dim = params.acoustic_gru_hidden_dim * 2
        else:
            self.audio_hidden_dim = params.acoustic_gru_hidden_dim
            self.attn_hidden_dim = params.acoustic_gru_hidden_dim

        self.fc1_attn = nn.Linear(self.attn_hidden_dim, params.fc_hidden_dim)
        self.fc2_attn = nn.Linear(params.fc_hidden_dim, 64)
        self.fc3_attn = nn.Linear(64, params.output_dim)

        self.fc1_audio = nn.Linear(self.audio_hidden_dim, params.fc_hidden_dim)
        self.fc2_audio = nn.Linear(params.fc_hidden_dim, 64)
        self.fc3_audio = nn.Linear(64, params.output_dim)

        self.fc_input_size = params.output_dim * 2

        self.fc1 = nn.Linear(self.fc_input_size, 64)
        self.fc2 = nn.Linear(64, params.output_dim)

    def forward(self,
                audio_input,
                audio_length,
                acoustic_input,
                acoustic_length):

        audio_input = audio_input.transpose(1, 2)
        attn_output, _ = self.acoustic_model(audio_input, audio_length)
        # print("before: ", acoustic_input.size())
        acoustic_input = torch.squeeze(acoustic_input, 1).transpose(1, 2)
        # print("after: ", acoustic_input.size())
        packed = nn.utils.rnn.pack_padded_sequence(
            acoustic_input, acoustic_length, batch_first=True, enforce_sorted=False
        )

        packed_output, (hidden, cell) = self.acoustic_rnn(packed)

        rnn_output = F.dropout(hidden[-1], 0.3)

        # print("attn size: ", attn_output.size())
        # print("rnn size: ", rnn_output.size())

        # print("cat. input size: ", inputs.size())

        attn_prediction = torch.tanh(F.dropout(self.fc1_attn(attn_output), 0.3))
        attn_prediction = torch.tanh(F.dropout(self.fc2_attn(attn_prediction), 0.3))
        attn_prediction = torch.tanh(F.dropout(self.fc3_attn(attn_prediction), 0.3))

        audio_prediction = torch.tanh(F.dropout(self.fc1_audio(rnn_output), 0.3))
        audio_prediction = torch.tanh(F.dropout(self.fc2_audio(audio_prediction), 0.3))
        audio_prediction = torch.tanh(F.dropout(self.fc3_audio(audio_prediction), 0.3))

        inputs = torch.cat((attn_prediction, audio_prediction), 1)
        # print("cat size: ", inputs.size())
        output = torch.tanh(F.dropout(self.fc1(inputs), 0.3))
        output = torch.tanh(F.dropout(self.fc2(output), 0.3))

        return output

class MultiAcousticModelEarlyMTL(nn.Module):
    def __init__(self, params):
        super(MultiAcousticModelEarlyMTL, self).__init__()
        # input dimensions
        self.attn_dim = params.attn_dim

        self.audio_dim = params.audio_dim

        self.acoustic_rnn = nn.LSTM(
            input_size=params.audio_dim,
            hidden_size=params.acoustic_gru_hidden_dim,
            num_layers=params.num_gru_layers,
            batch_first=True,
            bidirectional=params.bidirectional
        )

        encoder = Encoder(input_dim=params.attn_dim,
                          hidden_dim=params.acoustic_gru_hidden_dim,
                          num_gru_layers=params.num_gru_layers,
                          dropout=params.dropout,
                          bidirectional=params.bidirectional)

        attention_dim = params.acoustic_gru_hidden_dim if not params.bidirectional else 2 * params.acoustic_gru_hidden_dim
        attention = Attention(attention_dim, attention_dim, attention_dim)

        self.acoustic_model = AcousticAttn(
            encoder=encoder,
            attention=attention
        )

        if params.bidirectional:
            self.audio_hidden_dim = params.acoustic_gru_hidden_dim
            self.attn_hidden_dim = params.acoustic_gru_hidden_dim * 2
            self.fc_input_dim = self.audio_hidden_dim + self.attn_hidden_dim
        else:
            self.fc_input_dim = 2 * params.acoustic_gru_hidden_dim

        self.fc1_spk = nn.Linear(self.fc_input_dim, params.fc_hidden_dim)
        self.fc2_spk = nn.Linear(params.fc_hidden_dim, 64)
        self.fc3_spk = nn.Linear(64, params.speaker_output_dim)

        self.fc1 = nn.Linear(self.fc_input_dim, params.fc_hidden_dim)
        self.fc2 = nn.Linear(params.fc_hidden_dim, 64)
        self.fc3 = nn.Linear(64, params.sarc_output_dim)

    def forward(self,
                audio_input,
                audio_length,
                acoustic_input,
                acoustic_length):

        audio_input = audio_input.transpose(1, 2)
        attn_output, _ = self.acoustic_model(audio_input, audio_length)
        # print("before: ", acoustic_input.size())
        acoustic_input = torch.squeeze(acoustic_input, 1).transpose(1, 2)
        # print("after: ", acoustic_input.size())
        packed = nn.utils.rnn.pack_padded_sequence(
            acoustic_input, acoustic_length, batch_first=True, enforce_sorted=False
        )

        packed_output, (hidden, cell) = self.acoustic_rnn(packed)

        rnn_output = F.dropout(hidden[-1], 0.3)

        # print("attn size: ", attn_output.size())
        # print("rnn size: ", rnn_output.size())

        inputs = torch.cat((attn_output, rnn_output), 1)
        # print("cat. input size: ", inputs.size())

        sarc_output = torch.tanh(F.dropout(self.fc1(inputs), 0.3))
        sarc_output = torch.tanh(F.dropout(self.fc2(sarc_output), 0.3))
        sarc_output = self.fc3_spk(sarc_output)

        spk_output = torch.tanh(F.dropout(self.fc1_spk(inputs), 0.3))
        spk_output = torch.tanh(F.dropout(self.fc2_spk(spk_output), 0.3))
        spk_output = self.fc3_spk(spk_output)

        return sarc_output, spk_output

# the models used for multimodal, multitask classification
import sys
from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attn_models import *


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
        # self.acoustic_fc_2 = nn.Linear(100, 20)
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


class MultiAcousticModel(nn.Module):
    def __init__(self, params):
        super(MultiAcousticModel, self).__init__()
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
            self.audio_hidden_dim = params.acoustic_gru_hidden_dim * 2
            self.attn_hidden_dim = params.acoustic_gru_hidden_dim * 4

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

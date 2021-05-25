# create a class with basic model architecture for bimodal data

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultichannelCNN(nn.Module):
    """
    A CNN with multiple input channels with different kernel size operating over input
    """

    def __init__(self, params, num_embeddings, pretrained_embeddings=None):
        super(MultichannelCNN, self).__init__()
        # input dimensions
        self.text_dim = params.text_dim
        self.audio_dim = params.audio_dim
        self.in_channels = params.text_dim + params.audio_dim + params.spkr_emb_dim

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

        # speaker embeddings
        if params.spkr_embedding_size > 0:
            self.speaker_embeddings = nn.Embedding(
                params.num_speakers, params.spkr_emb_dim, max_norm=1.0
            )

        # word embeddings
        if pretrained_embeddings is None:
            self.embedding = nn.Embedding(
                num_embeddings, self.text_dim, params.padding_idx, max_norm=1.0
            )
            self.pretrained_embeddings = False
        else:
            self.embedding = nn.Embedding(
                num_embeddings,
                self.text_dim,
                params.padding_idx,
                _weight=pretrained_embeddings,
                max_norm=1.0,
            )
            self.pretrained_embeddings = True

        # convolutional layers and max pool of outputs
        self.conv1 = nn.Conv1d(self.in_channels, self.out_channels, self.k1_size)
        self.maxconv1 = nn.MaxPool1d(kernel_size=self.k1_size)
        self.conv2 = nn.Conv1d(self.in_channels, self.out_channels, self.k2_size)
        self.maxconv2 = nn.MaxPool1d(kernel_size=self.k2_size)
        self.conv3 = nn.Conv1d(self.in_channels, self.out_channels, self.k3_size)
        self.maxconv3 = nn.MaxPool1d(kernel_size=self.k3_size)

        # fully connected layers
        self.fc1 = nn.Linear(self.out_channels * 3, params.hidden_dim)
        self.fc2 = nn.Linear(params.hidden_dim, self.output_dim)

    def forward(self, acoustic_input, text_input, speaker_input=None):
        # get word embeddings
        if self.pretrained_embeddings:
            # detach to avoid training them if using pretrained
            embs = self.embedding(text_input).detach()
        else:
            embs = self.embedding(text_input)

        # get speaker embeddings if using
        # concatenate and perform permutation on input data
        if speaker_input is not None:
            spk_embs = self.speaker_embeddings(speaker_input)
            inputs = torch.cat((acoustic_input, embs, spk_embs), 2).permute(0, 2, 1)
        else:
            inputs = torch.cat((acoustic_input, embs), 2).permute(0, 2, 1)

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
        fc1_out = F.leaky_relu(self.fc1((F.dropout(intermediate, self.dropout))))
        output = self.fc2(F.dropout(fc1_out, self.dropout))

        # get predictions
        output = torch.sigmoid(output)

        return output.squeeze(dim=1)


class BimodalCNN(nn.Module):
    """
    A CNN for bimodal (text, audio) data; layers after 1 operate over output of layer 1?
    text_dim : length of each input text vector
    audio_dim : length of each input audio vector
    hidden_dim : size of hidden layer
    output_dim : length of output vector
    """

    def __init__(self, params, num_embeddings=None, pretrained_embeddings=None):
        super(BimodalCNN, self).__init__()
        # set dimensions of input
        self.text_dim = params.text_dim
        self.audio_dim = params.audio_dim
        self.num_embeddings = num_embeddings
        if num_embeddings is not None:
            self.in_channels = params.text_dim + params.audio_dim + params.spkr_emb_dim
        else:
            self.in_channels = params.audio_dim + params.spkr_emb_dim

        # set number of classes
        self.output_dim = params.output_dim

        # set number of layers and dropout
        self.num_layers = params.num_layers
        self.dropout = params.dropout

        # set convolutional kernel size and number of channels
        self.kernel_size = params.kernel_size
        self.out_channels = params.out_channels

        # initialize optional convolutional layers
        self.conv2 = None
        self.conv3 = None
        self.conv4 = None

        # initialize speaker embeddings
        if params.use_speaker:
            self.speaker_embeddings = nn.Embedding(
                params.num_speakers, params.spkr_emb_dim, max_norm=1.0
            )

        # initialize word embeddings
        if num_embeddings is not None:
            if pretrained_embeddings is None:
                self.embedding = nn.Embedding(
                    num_embeddings, self.text_dim, params.padding_idx, max_norm=1.0,
                )
                self.pretrained_embeddings = False
            else:
                self.embedding = nn.Embedding(
                    num_embeddings,
                    self.text_dim,
                    params.padding_idx,
                    _weight=pretrained_embeddings,
                    max_norm=1.0,
                )
                self.pretrained_embeddings = True

        # initialize mandatory convolutional layer
        self.conv1 = nn.Conv1d(
            self.in_channels, params.out_channels, self.kernel_size, stride=1
        )

        # initialize fully connected layers
        if params.num_fc_layers == 1:
            self.fc1 = nn.Linear(params.out_channels, params.output_dim)
            self.fc2 = None
        elif params.num_fc_layers == 2:
            self.fc1 = nn.Linear(params.out_channels, params.hidden_dim)
            self.fc2 = nn.Linear(params.hidden_dim, params.output_dim)

        # add optional layers as required
        if params.num_layers > 1:
            self.conv2 = nn.Conv1d(
                params.out_channels, params.out_channels, self.kernel_size, stride=2,
            )
            if params.num_layers == 3:
                self.conv3 = nn.Conv1d(
                    params.out_channels,
                    params.out_channels,
                    self.kernel_size,
                    stride=1,
                )
            elif params.num_layers == 4:
                # different stride in layer 3 if using 4 layers
                self.conv3 = nn.Conv1d(
                    params.out_channels,
                    params.out_channels,
                    self.kernel_size,
                    stride=2,
                )
                self.conv4 = nn.Conv1d(
                    params.out_channels, params.out_channels, self.kernel_size
                )

    def forward(self, acoustic_input, text_input, speaker_input=None):
        # create word embeddings
        # detach to ensure no training IF using pretrained
        if self.num_embeddings is not None:
            if self.pretrained_embeddings:
                embs = self.embedding(text_input).detach()
            else:
                embs = self.embedding(text_input)

            # check for speaker input information
            # perform permutation for proper input into convolutional layers
            #   (sequence length moved to last dimension)
            if speaker_input is not None:
                # if present, create speaker embedding + add to
                spk_embs = self.speaker_embeddings(speaker_input)
                inputs = torch.cat((acoustic_input, embs, spk_embs), 2).permute(0, 2, 1)
            else:
                # else, just concatenate acoustic input + word embeddings
                inputs = torch.cat((acoustic_input, embs), 2).permute(0, 2, 1)
        else:
            if speaker_input is not None:
                spk_embs = self.speaker_embeddings(speaker_input)
                inputs = torch.cat((acoustic_input, spk_embs), 2).permute(0, 2, 1)
            else:
                inputs = acoustic_input.permute(0, 2, 1)

        # feed inputs through the convolutional layer(s)
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

        # pool data and remove extra dimension as in book
        squeezed_size = feats.size(dim=2)
        feats = F.max_pool1d(feats, squeezed_size).squeeze(dim=2)
        # use pooled, squeezed feats as input into fc layers
        if self.fc2 is not None:
            fc1_out = torch.tanh(self.fc1((F.dropout(feats, self.dropout))))
            output = self.fc2(F.dropout(fc1_out, self.dropout))
        else:
            output = self.fc1(F.dropout(feats, self.dropout))

        if self.output_dim == 1:
            output = torch.sigmoid(output)
            output = output.squeeze(1)
        else:
            output = F.softmax(output)

        # return the output
        # squeeze to 1 dimension for binary categorization
        return output


# # not currently used
# will need to be updated if used--written at the very beginning
# class BimodalRNN(nn.Module):
#     """
#     An RNN for bimodal (text, audio) data
#     text_dim : length of each input text vector
#     audio_dim : length of each input audio vector
#     hidden_dim : size of hidden layer todo: why is there only one hidden layer?
#     output_dim : length of output vector
#     """
#     def __init__(self, text_dim, audio_dim, hidden_dim, output_dim, num_layers, dropout,
#                  rnn_type='lstm'):
#         super(BimodalRNN, self).__init__()
#         self.text_dim = text_dim
#         self.audio_dim = audio_dim
#         self.data_shape = text_dim + audio_dim
#         self.hidden_dim = hidden_dim
#         self.output_dim = output_dim
#         self.num_layers = num_layers
#         self.dropout = dropout
#         self.rnn_type = rnn_type
#
#         # create model layers
#         # lstm layers
#         if self.rnn_type == "lstm":
#             self.lstm = nn.LSTM(self.data_shape, hidden_dim, self.num_layers, dropout=self.dropout)  # may be data_shape[1:]
#         elif self.rnn_type == "gru":
#             self.gru = nn.GRU(self.data_shape, hidden_dim, self.num_layers, dropout=self.dropout)
#
#         # output layer
#         self.output_layer = nn.Linear(hidden_dim, output_dim)
#
#     def init_hidden(self, batch_size):
#         # initialize the hidden layer
#         return torch.zeros(self.num_layers, batch_size, self.hidden_dim)
#
#     def forward(self, inputs):
#         # set lstm_out, hidden using the lstm layer
#         if self.rnn_type == "lstm":
#             lstm_out = F.relu(self.lstm(inputs)) # fixme: something isn't right here
#         elif self.rnn_type == "gru":
#             lstm_out = F.relu(self.gru(inputs))
#
#         # get last output only
#         last_out = lstm_out[:, -1, :]
#
#         # can apply dropout if needed in lstm layer
#         # last_out = F.dropout(last_out, 0.5)
#         # get output predictions
#         output = F.relu(self.output_layer(last_out))
#
#         # apply softmax and return most likely as prediction
#         output_distribution = self.softmax(output)
#         return max(output_distribution)

##########################################################################
#                                                                        #
#                      UTTERANCE-ALIGNED MODELS                          #
#                                                                        #
##########################################################################


class UttLevelBimodalCNN(nn.Module):
    """
    A CNN for bimodal (text, audio) data; layers after 1 operate over output of layer 1?
    text_dim : length of each input text vector
    audio_dim : length of each input audio vector
    hidden_dim : size of hidden layer
    output_dim : length of output vector
    """

    def __init__(self, params, num_embeddings=None, pretrained_embeddings=None):
        super(UttLevelBimodalCNN, self).__init__()
        # set dimensions of input
        self.text_dim = params.text_dim
        self.audio_dim = params.audio_dim
        self.num_embeddings = num_embeddings
        if num_embeddings is not None:
            self.in_channels = params.text_dim + params.audio_dim + params.spkr_emb_dim
        else:
            self.in_channels = params.audio_dim + params.spkr_emb_dim

        # set number of classes
        self.output_dim = params.output_dim

        # set number of layers and dropout
        self.num_layers = params.num_layers
        self.dropout = params.dropout

        # set convolutional kernel size and number of channels
        self.kernel_size = params.kernel_size
        self.out_channels = params.out_channels

        # initialize optional convolutional layers
        self.conv2 = None
        self.conv3 = None
        self.conv4 = None

        # initialize speaker embeddings
        if params.use_speaker:
            self.speaker_embeddings = nn.Embedding(
                params.num_speakers, params.spkr_emb_dim, max_norm=1.0
            )

        # initialize word embeddings
        if num_embeddings is not None:
            if pretrained_embeddings is None:
                self.embedding = nn.Embedding(
                    num_embeddings, self.text_dim, params.padding_idx, max_norm=1.0,
                )
                self.pretrained_embeddings = False
            else:
                self.embedding = nn.Embedding(
                    num_embeddings,
                    self.text_dim,
                    params.padding_idx,
                    _weight=pretrained_embeddings,
                    max_norm=1.0,
                )
                self.pretrained_embeddings = True

        # initialize mandatory convolutional layer
        self.conv1 = nn.Conv1d(
            self.in_channels, params.out_channels, self.kernel_size, stride=1
        )

        # initialize fully connected layers
        if params.num_fc_layers == 1:
            self.fc1 = nn.Linear(params.out_channels, params.output_dim)
            self.fc2 = None
        elif params.num_fc_layers == 2:
            self.fc1 = nn.Linear(params.out_channels, params.hidden_dim)
            self.fc2 = nn.Linear(params.hidden_dim, params.output_dim)

        # add optional layers as required
        if params.num_layers > 1:
            self.conv2 = nn.Conv1d(
                params.out_channels, params.out_channels, self.kernel_size, stride=2,
            )
            if params.num_layers == 3:
                self.conv3 = nn.Conv1d(
                    params.out_channels,
                    params.out_channels,
                    self.kernel_size,
                    stride=1,
                )
            elif params.num_layers == 4:
                # different stride in layer 3 if using 4 layers
                self.conv3 = nn.Conv1d(
                    params.out_channels,
                    params.out_channels,
                    self.kernel_size,
                    stride=2,
                )
                self.conv4 = nn.Conv1d(
                    params.out_channels, params.out_channels, self.kernel_size
                )

    def forward(self, acoustic_input, text_input, speaker_input=None):
        # create word embeddings
        # detach to ensure no training IF using pretrained
        if self.num_embeddings is not None:
            if self.pretrained_embeddings:
                embs = self.embedding(text_input).detach()
            else:
                embs = self.embedding(text_input)

            # check for speaker input information
            # perform permutation for proper input into convolutional layers
            #   (sequence length moved to last dimension)
            if speaker_input is not None:
                # if present, create speaker embedding + add to
                spk_embs = self.speaker_embeddings(speaker_input)
                inputs = torch.cat((acoustic_input, embs, spk_embs), 2).permute(0, 2, 1)
            else:
                # else, just concatenate acoustic input + word embeddings
                inputs = torch.cat((acoustic_input, embs), 2).permute(0, 2, 1)
        else:
            if speaker_input is not None:
                spk_embs = self.speaker_embeddings(speaker_input)
                inputs = torch.cat((acoustic_input, spk_embs), 2).permute(0, 2, 1)
            else:
                inputs = acoustic_input.permute(0, 2, 1)

        # feed inputs through the convolutional layer(s)
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

        # pool data and remove extra dimension as in book
        squeezed_size = feats.size(dim=2)
        feats = F.max_pool1d(feats, squeezed_size).squeeze(dim=2)
        # use pooled, squeezed feats as input into fc layers
        if self.fc2 is not None:
            fc1_out = torch.tanh(self.fc1((F.dropout(feats, self.dropout))))
            output = self.fc2(F.dropout(fc1_out, self.dropout))
        else:
            output = self.fc1(F.dropout(feats, self.dropout))

        if self.output_dim == 1:
            output = torch.sigmoid(output)
            output = output.squeeze(1)
        else:
            output = F.softmax(output)

        # return the output
        # squeeze to 1 dimension for binary categorization
        return output

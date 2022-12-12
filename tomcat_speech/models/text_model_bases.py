
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self, text_input, speaker_input=None, length_input=None, gender_input=None,
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

class TextRNNBase(nn.Module):
    """
    The text-rnn base for intermediate fusion+ models
    """
    def __init__(self, params, num_embeddings=None, pretrained_embeddings=None, use_distilbert=False):
        super(TextRNNBase, self).__init__()

        self.text_dim = params.text_dim
        self.num_embeddings = num_embeddings
        self.text_gru_hidden_dim = params.text_gru_hidden_dim

        # if we feed text through additional layer(s)
        if not use_distilbert:
            self.text_input_size = params.text_dim + params.short_emb_dim
        else:
            self.text_input_size = params.text_dim

        self.text_rnn = nn.LSTM(
            input_size=self.text_input_size,
            hidden_size=params.text_gru_hidden_dim,
            num_layers=params.num_gru_layers,
            batch_first=True,
            bidirectional=True,
        )
        # self.text_batch_norm = nn.BatchNorm1d(num_features=params.text_gru_hidden_dim)

        # distilbert vs glove initialization
        self.use_distilbert = use_distilbert

        if not use_distilbert:
            # initialize word embeddings
            self.embedding = nn.Embedding(
                num_embeddings, self.text_dim, _weight=pretrained_embeddings
            )
            self.short_embedding = nn.Embedding(num_embeddings, params.short_emb_dim)

    def forward(self, 
        text_input,
        length_input=None,
    ):
        # using pretrained embeddings, so detach to not update weights
        # embs: (batch_size, seq_len, emb_dim)
        if not self.use_distilbert:
            embs = F.dropout(self.embedding(text_input), 0.1).detach()

            short_embs = F.dropout(self.short_embedding(text_input), 0.1)

            all_embs = torch.cat((embs, short_embs), dim=2)
        else:
            all_embs = text_input

        # flatten_parameters() decreases memory usage
        self.text_rnn.flatten_parameters()

        packed = nn.utils.rnn.pack_padded_sequence(
            all_embs, length_input, batch_first=True, enforce_sorted=False
        )

        # feed embeddings through GRU
        packed_output, (hidden, cell) = self.text_rnn(packed)

        return hidden[-1]


class TextOnlyModel(nn.Module):
    """
    An encoder to take a sequence of inputs and produce a sequence of intermediate representations
    Can include convolutions over text input and/or acoustic input--BUT NOT TOGETHER bc MELD isn't
    aligned at the word-level
    """

    def __init__(self, params, num_embeddings=None, pretrained_embeddings=None, use_distilbert=False):
        super(TextOnlyModel, self).__init__()
        # input text + acoustic + speaker
        self.text_dim = params.text_dim
        self.num_embeddings = num_embeddings
        self.num_speakers = params.num_speakers
        self.text_gru_hidden_dim = params.text_gru_hidden_dim

        # get number of output dims
        self.out_dims = params.output_0_dim #12/06/22

        # if we feed text through additional layer(s)
        if not use_distilbert:
            self.text_input_size = params.text_dim + params.short_emb_dim
        else:
            self.text_input_size = params.text_dim

        # if we feed text through additional layer(s)
        self.text_rnn = nn.LSTM(
            input_size=self.text_input_size,
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
        self.output_dim = params.output_0_dim #revised output_dim => output_0_dim 12/06/22
        

        # set number of layers and dropout
        self.dropout = params.dropout

        # distilbert vs glove initialization
        self.use_distilbert = use_distilbert

        if not use_distilbert:
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
        self.fc2 = nn.Linear(params.fc_hidden_dim, params.output_0_dim) #output_dim was from multithing - actually for 2 because we are using "text only" model as a separate model

    def forward(
        self,
        text_input,
        speaker_input=None,
        length_input=None,
        gender_input=None,
        get_prob_dist=False,
        save_encoded_data=False,
        spec_input = None #added 10/24/22
    ):
        print(f"text input received by model is of size: {text_input.size()}")
        # using pretrained embeddings, so detach to not update weights
        # embs: (batch_size, seq_len, emb_dim)
        if not self.use_distilbert:
            embs = F.dropout(self.embedding(text_input), 0.1).detach()

            short_embs = F.dropout(self.short_embedding(text_input), 0.1)

            all_embs = torch.cat((embs, short_embs), dim=2)
        else:
            #print("We accessed the code correctly.") # added 11/29/22
            all_embs = text_input

        # get speaker embeddings, if needed
        if speaker_input is not None:
            speaker_embs = self.speaker_embedding(speaker_input).squeeze(dim=1)
        if gender_input is not None:
            gender_embs = self.gender_embedding(gender_input)

        # added 11/29/22 to check the size
        #print(all_embs.size())

        packed = nn.utils.rnn.pack_padded_sequence(
            all_embs, length_input, batch_first=True, enforce_sorted=False
        )

        # added 11/29/22 to check the size
        print(packed.data.size())

        # feed embeddings through GRU
        packed_output, (hidden, cell) = self.text_rnn(packed)
        encoded_text = F.dropout(hidden[-1], 0.3)

        print(f"size of encoded text at line 412 of model: {encoded_text.size()}")

        # combine modalities as required by architecture
        if speaker_input is not None:
            inputs = torch.cat((encoded_text, speaker_embs), 1)
        elif gender_input is not None:
            inputs = torch.cat((encoded_text, gender_embs), 1)
        else:
            inputs = encoded_text

        print(f"inputs at line 421 of model: {inputs.size()}")
        # use pooled, squeezed feats as input into fc layers
        output = torch.tanh(F.dropout(self.fc1(inputs), 0.5))
        print(f"output size at line 423 of model: {output.size()}")
        output = self.fc2(output) # 12/06/22
        print(f"output size at line 425 of model: {output.size()}")

        if self.out_dims == 1: 
            output = torch.sigmoid(output)
        elif get_prob_dist:
            prob = nn.Softmax(dim=1)
            output = prob(output)

        print("Printing output of model forward pass")
        print(output)
        print(output.size())
        # return the output
        return output

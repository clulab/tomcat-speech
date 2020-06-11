# baseline models
import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingsOnly(nn.Module):
    """
    Only uses word embeddings to predict call outcome
    """
    def __init__(self, params, num_embeddings, pretrained_embeddings=None):
        super(EmbeddingsOnly, self).__init__()

        # input dimensions
        self.text_dim = params.text_dim

        # dropout for fully connected layer
        self.dropout = params.dropout

        # word embeddings
        if pretrained_embeddings is None:
            self.embedding = nn.Embedding(num_embeddings, self.text_dim, params.padding_idx)
        else:
            self.embedding = nn.Embedding(num_embeddings, self.text_dim, params.padding_idx,
                                          _weight=pretrained_embeddings)

        # convolutional layers
        self.conv1 = nn.Conv1d(self.text_dim, params.out_channels, kernel_size=params.kernel_size)
        self.conv2 = nn.Conv1d(params.out_channels, params.out_channels, kernel_size=params.kernel_size, stride=2)
        self.conv3 = nn.Conv1d(params.out_channels, params.out_channels, kernel_size=params.kernel_size)

        # fully connected layer
        self.linear = nn.Linear(params.out_channels, params.output_dim)

    def forward(self, text_input):
        # get word embeddings
        embs = self.embedding(text_input).permute(0, 2, 1)

        # feed through convolutional layers
        intermediate1 = F.relu(self.conv1(embs))
        intermediate2 = F.relu(self.conv2(intermediate1))
        intermediate3 = F.relu(self.conv3(intermediate2))

        # squeeze and put through fully connected layer
        squeezed_size = intermediate3.size(dim=2)
        feats = F.max_pool1d(intermediate3, squeezed_size).squeeze(dim=2)
        fc_out = torch.sigmoid(self.linear(F.dropout(feats, self.dropout)))

        # return predictions
        return fc_out.squeeze(dim=1)


class LRBaseline(nn.Module):
    """
    A logistic regression model for bimodal (text, audio) data
    text_dim : number of dimensions in each text input vector (e.g. 300)
    audio_dim : number of dimensions in each audio input vector (e.g. 20)
    output_dim : number of dimensions in output vector (binary default)
    """
    def __init__(self, params, num_embeddings, pretrained_embeddings=None):
        super(LRBaseline, self).__init__()
        self.text_dim = params.text_dim
        self.audio_dim = params.audio_dim
        self.input_dim = params.text_dim + params.audio_dim + params.spkr_emb_dim

        if params.use_speaker:
            self.speaker_embeddings = nn.Embedding(params.num_speakers, params.spkr_emb_dim)

        # create model layer
        if pretrained_embeddings is None:
            self.embedding = nn.Embedding(num_embeddings, self.text_dim, params.padding_idx)
            self.pretrained_embeddings = False
        else:
            self.embedding = nn.Embedding(num_embeddings, self.text_dim, params.padding_idx,
                                          _weight=pretrained_embeddings)
            self.pretrained_embeddings = True
        self.linear = nn.Linear(self.input_dim, params.output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, acoustic_input, text_input, speaker_input=None):
        # feed text embeddings ONLY through embedding layer
        if self.pretrained_embeddings:
            embs = self.embedding(text_input).detach()
        else:
            embs = self.embedding(text_input)

        # concatenate embedding + acoustic inut and feed this through linear layer
        if speaker_input is not None:
            spk_embs = self.speaker_embeddings(speaker_input)
            inputs = torch.cat((acoustic_input, embs, spk_embs), 2).permute(0, 2, 1)
        else:
            inputs = torch.cat((acoustic_input, embs), 2).permute(0, 2, 1)

        # pool data and remove extra dimension as in book
        # averages features across all frames
        squeezed_size = inputs.size(dim=2)
        inputs = F.avg_pool1d(inputs, squeezed_size).squeeze(dim=2)
        # could also try max, sum, concatenate at some point

        # feed through layer with sigmoid activation
        outputs = self.sigmoid(self.linear(inputs))

        # get predictions
        return outputs.squeeze(dim=1)

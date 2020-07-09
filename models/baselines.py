# baseline models
import torch
import torch.nn as nn
import torch.nn.functional as F


class UttLRBaseline(nn.Module):
    """
    LR model for utterance-level predictions on bimodal data
    """
    def __init__(self, params, num_embeddings, pretrained_embeddings=None):
        super(UttLRBaseline, self).__init__()
        self.text_dim = params.text_dim
        self.audio_dim = params.audio_dim
        self.input_dim = params.text_dim + params.audio_dim

        # set embeddings layer
        self.embedding = nn.Embedding(num_embeddings, self.text_dim,
                                      _weight=pretrained_embeddings)

        self.linear = nn.Linear(self.input_dim, params.output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, acoustic_input, text_input, length_input=None):
        embs = self.embedding(text_input).detach()

        avgd_embs = []

        for i, emb in enumerate(embs):

            emb.narrow(0, 0, length_input[i])
            emb = torch.mean(emb, dim=0).squeeze()
            avgd_embs.append(emb.tolist())

        embs = torch.tensor(avgd_embs)

        inputs = torch.cat((acoustic_input, embs), 1)

        # feed through layer with sigmoid activation
        outputs = self.sigmoid(self.linear(inputs))

        # get predictions
        return outputs.squeeze(dim=1)

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

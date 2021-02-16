import os
import sys
import numpy as np

import pickle
import torch
import torchaudio
import torch.nn.functional as F

import glob

import argparse

from fairseq.models.wav2vec import Wav2VecModel
from fairseq.models.roberta import RobertaModel

'''
from https://github.com/shamanez/BERT-like-is-All-You-Need/blob/master/SPEECH-BERT-TOKENIZATION/convert_aud_to_token.py
'''

problem_aud = open('problem_aud.txt', 'w')

class EmotionDataPreprocessing():

    def __init__(self, vq_wav2vec, pretrained_roberta):
        # Load vq-wav2vec
        cp = torch.load(vq_wav2vec)
        self.model = Wav2VecModel.build_model(cp['args'], task=None)
        self.model.load_state_dict(cp['model'])
        self.model.eval()

        # Load Speech ROBERTa trained on LibriSpeech
        self.roberta = RobertaModel.from_pretrained(pretrained_roberta, checkpoint_file='bert_kmeans.pt')


        self.roberta.eval()

    def indices_to_string(self, idxs):
        # based on fairseq/examples/wav2vec/vq-wav2vec_featurize.py
        return "<s>"+" " +" ".join("-".join(map(str, a.tolist())) for a in idxs.squeeze(0))

    def preprocess_audio_file(self, filename):
        feats_audio, sr = torchaudio.load(filename, normalization=True)
        feats_audio = feats_audio.squeeze(0)
        assert feats_audio.dim() == 1, feats_audio.dim()
        print("Audio: ", feats_audio.size())
        return feats_audio

    def preprocess_data(self, audio_path):
        num_items = 1e18
        current_num = 0

        # item = {}
        if audio_path:
            audio_files = sorted(glob.glob(audio_path + "*.wav"))
            print(len(audio_files), "audio_files found")

            for audio_file in audio_files:
                audio_features = self.preprocess_audio_file(audio_file).unsqueeze(0)

                # wav2vec
                z = self.model.feature_extractor(audio_features)
                _, idxs = self.model.vector_quantizer.forward_idx(z)

                mel_time = idxs.size()[1]

                # if mel_time > 686:
                #     length = 686
                # else:
                #     length = mel_time

                # audio_name = audio_file.split("/")[-1].replace(".wav", "")
                # print(audio_name, length)
                # item[audio_name] = [length]

                if mel_time > 686:
                    diff = mel_time - 686

                    random_start = np.random.randint(0, diff + 1)
                    end = mel_time - diff + random_start

                    target_tensor = idxs[:, random_start:end, :]
                elif mel_time == 686:
                    target_tensor = idxs
                else:
                    target_tensor = torch.zeros(size=(1, 686, 2), dtype=torch.int32)
                    target_tensor[:, :mel_time, :] = idxs

                idx_str = self.indices_to_string(target_tensor)

                tokens = self.roberta.task.source_dictionary.encode_line(idx_str, append_eos=True, add_if_not_exist=False).detach().numpy()

                output_file = audio_file.replace('wavs', 'wavs_roberta').replace('.wav', '.txt')

                os.makedirs(os.path.dirname(output_file), exist_ok=True)

                with open(output_file, 'w') as f:
                    for item in tokens:
                        f.write(str(item) + '\t')
                current_num += 1
                if current_num > num_items:
                    break

            # with open("../data/data_pt/audio_length_dev.pt", "wb") as out_data:
            #     pickle.dump(item, out_data)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--audio_path', default=None, help='path for raw audio_train files')
    parser.add_argument('-w', '--w2v_path', default=None, help='path for wav2vec model')
    parser.add_argument('-r', '--roberta', default=None, help='path for speech roberta model')

    args = parser.parse_args()

    audio_path = args.audio_path
    w2v_model = args.w2v_path
    roberta_model = args.roberta

    data_processor = EmotionDataPreprocessing(w2v_model, roberta_model)

    data_processor.preprocess_data(audio_path)


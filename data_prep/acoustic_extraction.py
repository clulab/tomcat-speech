import torchaudio, torch
import os
import numpy as np


def read_audio(audio_path="", rnn=True):
    audio_files = os.listdir(audio_path)

    audio_dict = {}

    ### Uncomment this if you want to check max_len of the audio_train file
    # max_time = 0
    # for audio_train in audio_files:
    # if (".mp3") in audio_train:
    #     audio_name = audio_train.replace(".mp3", "")
    #     filename = os.path.join(audio_path, audio_train)
    #
    #     waveform, sample_rate = torchaudio.load(filename, normalization=True)
    #     # print(np.shape(waveform))
    #     mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate)(waveform)
    #     wave_time = np.shape(mel_spectrogram)[2]
    #
    # if wave_time > max_time:
    #     max_time = wave_time

    audio_length = {}
    for audio in audio_files:
        if (".mp3") in audio:
            audio_name = audio.replace(".mp3", "")
            filename = os.path.join(audio_path, audio)

            waveform, sample_rate = torchaudio.load(filename, normalization=True)

            # get mel_spectrogram
            mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate,
                                                                   hop_length=256,
                                                                   n_mels=96,
                                                                   n_fft=256,
                                                                   pad=0)(waveform)
            # get mfcc
            mfcc = torchaudio.transforms.MFCC(sample_rate, n_mfcc=13,
                                              melkwargs={"hop_length": 256, "n_mels": 96, "n_fft": 256})
            mfcc_feature = mfcc.forward(waveform)
            mfcc_delta = torchaudio.transforms.ComputeDeltas().forward(mfcc_feature)
            mfcc_delta_delta = torchaudio.transforms.ComputeDeltas().forward(mfcc_delta)

            ### Choose feature to use (mel_spec = 1, 96, X, mfcc = 1, mfcc, X)
            # concat_feature = torch.cat((mel_spectrogram, mfcc_feature, mfcc_delta, mfcc_delta_delta), dim=1)
            concat_feature = torch.cat((mfcc_feature, mfcc_delta, mfcc_delta_delta), dim=1)
            # concat_feature = mel_spectrogram

            if rnn:
                ### For RNN, clip the audio_train if it's longer than 596
                ### Else, just use it as it is
                mel_time = mel_spectrogram.size()[2]
                if mel_time > 1000:
                    target_tensor = concat_feature[:1, :39, :1000]
                    audio_length[audio_name] = 1000
                else:
                    target_tensor = concat_feature
                    audio_length[audio_name] = mel_time

                audio_dict[audio_name] = target_tensor
            else:
                ### For CNN, clip the audio_train if it's longer than 596
                ### Else, zero-padding
                mel_time = mel_spectrogram.size()[2]
                # audio_length.append(mel_time)

                target_tensor = torch.zeros(1, 90, 1000)

                if mel_time > 1000:
                    target_tensor = concat_feature[:1, :90, :1000]
                    audio_length[audio_name] = 1000
                else:
                    target_tensor[:, :, :mel_time] = concat_feature
                    audio_length[audio_name] = mel_time

                audio_dict[audio_name] = target_tensor
    # print("num. audio_train: ", len(audio_length))
    # print("min: ", min(audio_length))
    # print("median: ", np.median(audio_length))
    # print("mean: ", np.mean(audio_length))
    # print("max: ", max(audio_length))
    # exit()
    return audio_dict, audio_length

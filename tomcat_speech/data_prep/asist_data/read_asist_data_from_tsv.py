# for reading asist data that has been put in tsvs
# the tsv contains the start and end times + utterance info
# similar to read_asist_data_from_replay.py

import sys
import os
import pandas as pd

sys.path.append("../multimodal_data_preprocessing")
from utils.audio_extraction import *


class TSVDataReader:
    def __init__(self, path_to_tsv, audio_name):
        self.tsv_path = path_to_tsv
        # print(self.metadata_path)
        self.path, self.mission = path_to_tsv.rsplit("/", 1)
        self.mission = self.mission.split(".")[0]

        self.audio_path = f"{self.path}/{audio_name}"
        self.audio = audio_name

        self.transcribed_utts = pd.read_csv(path_to_tsv, sep='\t')

        self.multiple_participants = False

    def _split_audio_using_transcriptions(self, start_time, end_time, audio_save_path="split"):
        # make sure the full save path exists; if not, create it
        os.system(f'if [ ! -d "{audio_save_path}" ]; then mkdir -p {audio_save_path}; fi')

        # use timestamps to split
        extracted_wav = extract_portions_of_mp4_or_wav(self.path,
                                                       self.audio,
                                                       start_time,
                                                       end_time,
                                                       save_path=f"{self.path}/{audio_save_path}")

        return extracted_wav

    def get_opensmile_feats_for_all_utts(self):
        # get opensmile features for one utterance from its json
        # get start time
        for row in self.transcribed_utts.itertuples():
            start_time = row.real_start
            end_time = row.real_end

            extracted_wav_path = self._split_audio_using_transcriptions(start_time, end_time)
            extracted_path, extracted_wav = extracted_wav_path.rsplit("/", 1)
            extracted_wav_name = extracted_wav.split(".wav")[0]
            csv_name = f"{extracted_wav_name}_IS13.csv"

            extractor = ExtractAudio(extracted_path, extracted_wav, f"{self.path}/IS13",
                                     smilepath="../../opensmile-3.0")

            extractor.save_acoustic_csv("IS13", csv_name)


if __name__ == "__main__":
    mpath = "../../study3_data_to_annotate"

    metadata = [f"{mpath}/T000601_E000622_gold_annotated.tsv",
                f"{mpath}/T000602_E000622_gold_annotated.tsv",
                f"{mpath}/T000603_E000607_gold_annotated.tsv",
                f"{mpath}/T000603_E000649_gold_annotated.tsv",
                f"{mpath}/T000603_E000651_gold_annotated.tsv",
                # f"{mpath}/T000604_gold.tsv",
                # f"{mpath}/T000605_gold.tsv",
                # f"{mpath}/T000606_gold.tsv",
                # f"{mpath}/T000607_gold.tsv",
                # f"{mpath}/T000608_gold.tsv",
                # f"{mpath}/T000609_gold.tsv",
                # f"{mpath}/T000610_gold.tsv",
                ]
    wav = [
        "study-3_2022_HSRData_ClientAudio_Trial-T000601_Team-TM000201_Member-E000622_CondBtwn-none_CondWin-na_Vers-1.wav",
        "study-3_2022_HSRData_ClientAudio_Trial-T000602_Team-TM000201_Member-E000622_CondBtwn-none_CondWin-na_Vers-1.wav",
        "study-3_2022_HSRData_ClientAudio_Trial-T000603_Team-TM000202_Member-E000607_CondBtwn-none_CondWin-na_Vers-1.wav",
        "study-3_2022_HSRData_ClientAudio_Trial-T000603_Team-TM000202_Member-E000649_CondBtwn-none_CondWin-na_Vers-1.wav",
        "study-3_2022_HSRData_ClientAudio_Trial-T000603_Team-TM000202_Member-E000651_CondBtwn-none_CondWin-na_Vers-1.wav",

    ]

    for i, item in enumerate(metadata):
        print(item)
        reader = TSVDataReader(item, wav[i])

        # reader = MetaDataReader(metadata, wav)

        reader.get_opensmile_feats_for_all_utts()
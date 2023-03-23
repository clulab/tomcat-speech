# for reading asist data that has been put in tsvs
# the tsv contains the start and end times + utterance info
# similar to read_asist_data_from_replay.py

import sys
import os
import pandas as pd

sys.path.append("../multimodal_data_preprocessing")
from utils.audio_extraction import *


class TSVDataReader:
    def __init__(self, path_to_tsv, audio_name, use_tsv=True):
        self.tsv_path = path_to_tsv
        # print(self.metadata_path)
        self.path, self.mission = path_to_tsv.rsplit("/", 1)
        self.mission = self.mission.split(".")[0]

        self.audio_path = f"{self.path}/{audio_name}"
        self.audio = audio_name

        if use_tsv:
            self.transcribed_utts = pd.read_csv(path_to_tsv, sep='\t')
        else:
            self.transcribed_utts = pd.read_csv(path_to_tsv)
        self.multiple_participants = False

    def _split_audio_using_transcriptions(self, start_time, end_time, audio_save_path="split",
                                          short_file_name=None):
        # make sure the full save path exists; if not, create it
        os.system(f'if [ ! -d "{audio_save_path}" ]; then mkdir -p {audio_save_path}; fi')

        # use timestamps to split
        extracted_wav = extract_portions_of_mp4_or_wav(self.path,
                                                       self.audio,
                                                       start_time,
                                                       end_time,
                                                       save_path=f"{self.path}/{audio_save_path}",
                                                       short_file_name=short_file_name)

        return extracted_wav

    def get_opensmile_feats_for_all_utts(self):
        # get opensmile features for one utterance from its json
        # get start time
        for row in self.transcribed_utts.itertuples():
            start_time = row.real_start
            end_time = row.real_end

            utt_id = row.message_id + ".wav"
            extracted_wav_path = self._split_audio_using_transcriptions(start_time, end_time,
                                                                        short_file_name=utt_id)
            extracted_path, extracted_wav = extracted_wav_path.rsplit("/", 1)
            extracted_wav_name = extracted_wav.split(".wav")[0]

            csv_name = f"{extracted_wav_name}_IS13.csv"

            extractor = ExtractAudio(extracted_path, extracted_wav, f"{self.path}/IS13",
                                     smilepath="../../opensmile-3.0")

            extractor.save_acoustic_csv("IS13", csv_name)


if __name__ == "__main__":
    mpath = "../../asist_data"

    gold_data = [
    f"{mpath}/sent-emo/for_PI_meeting_07.22/Trial-T000713_Team-TM000257_gold.csv",
    f"{mpath}/sent-emo/for_PI_meeting_07.22/Trial-T000714_Team-TM000257_gold.csv",
    f"{mpath}/sent-emo/for_PI_meeting_07.22/Trial-T000715_Team-TM000258_gold.csv",
    f"{mpath}/sent-emo/for_PI_meeting_07.22/Trial-T000716_Team-TM000258_gold.csv",
    f"{mpath}/sent-emo/for_PI_meeting_07.22/Trial-T000719_Team-TM000260_gold.csv",
    f"{mpath}/sent-emo/for_PI_meeting_07.22/Trial-T000720_Team-TM000260_gold.csv",
    f"{mpath}/sent-emo/for_PI_meeting_07.22/Trial-T000723_Team-TM000262_gold.csv",
    f"{mpath}/sent-emo/for_PI_meeting_07.22/Trial-T000724_Team-TM000262_gold.csv",
    f"{mpath}/sent-emo/for_PI_meeting_07.22/Trial-T000727_Team-TM000264_gold.csv",
    f"{mpath}/sent-emo/for_PI_meeting_07.22/Trial-T000728_Team-TM000264_gold.csv",
    f"{mpath}/sent-emo/for_PI_meeting_07.22/Trial-T000729_Team-TM000265_gold.csv",
    f"{mpath}/sent-emo/for_PI_meeting_07.22/Trial-T000730_Team-TM000265_gold.csv",
    f"{mpath}/sent-emo/for_PI_meeting_07.22/Trial-T000737_Team-TM000269_gold.csv",
    f"{mpath}/sent-emo/for_PI_meeting_07.22/Trial-T000738_Team-TM000269_gold.csv",
    ]
    wav = [
        f"HSRData_OBVideo_Trial-T000713_Team-TM000257_Member-na_CondBtwn-ASI-CRA-TA1_CondWin-na_Vers-1.wav",
        f"HSRData_OBVideo_Trial-T000714_Team-TM000257_Member-na_CondBtwn-ASI-CRA-TA1_CondWin-na_Vers-1.wav",
        f"HSRData_OBVideo_Trial-T000715_Team-TM000258_Member-na_CondBtwn-ASI-DOLL-TA1_CondWin-na_Vers-1.wav",
        f"HSRData_OBVideo_Trial-T000716_Team-TM000258_Member-na_CondBtwn-ASI-DOLL-TA1_CondWin-na_Vers-1.wav",
        f"HSRData_OBVideo_Trial-T000719_Team-TM000260_Member-na_CondBtwn-ASI-UAZ-TA1_CondWin-na_Vers-1.wav",
        f"HSRData_OBVideo_Trial-T000720_Team-TM000260_Member-na_CondBtwn-ASI-UAZ-TA1_CondWin-na_Vers-1.wav",
        f"HSRData_OBVideo_Trial-T000723_Team-TM000262_Member-na_CondBtwn-ASI-SIFT-TA1_CondWin-na_Vers-1.wav",
        f"HSRData_OBVideo_Trial-T000724_Team-TM000262_Member-na_CondBtwn-ASI-SIFT-TA1_CondWin-na_Vers-1.wav",
        f"HSRData_OBVideo_Trial-T000727_Team-TM000264_Member-na_CondBtwn-ASI-USC-TA1_CondWin-na_Vers-1.wav",
        f"HSRData_OBVideo_Trial-T000728_Team-TM000264_Member-na_CondBtwn-ASI-USC-TA1_CondWin-na_Vers-1.wav",
        f"HSRData_OBVideo_Trial-T000729_Team-TM000265_Member-na_CondBtwn-ASI-DOLL-TA1_CondWin-na_Vers-1.wav",
        f"HSRData_OBVideo_Trial-T000730_Team-TM000265_Member-na_CondBtwn-ASI-DOLL-TA1_CondWin-na_Vers-1.wav",
        f"HSRData_OBVideo_Trial-T000737_Team-TM000269_Member-na_CondBtwn-ASI-SIFT-TA1_CondWin-na_Vers-1.wav",
        f"HSRData_OBVideo_Trial-T000738_Team-TM000269_Member-na_CondBtwn-ASI-SIFT-TA1_CondWin-na_Vers-1.wav",
    ]

    for i, item in enumerate(gold_data):
        print(item)
        reader = TSVDataReader(item, wav[i], use_tsv=False)

        # reader = MetaDataReader(metadata, wav)

        reader.get_opensmile_feats_for_all_utts()
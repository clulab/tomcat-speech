# read in asist data from a metadata_replay file
# use the boundaries of word-level timestamps to select portion of audio to run through opensmile
# get opensmile features and combine with text from metadata file

import json
import sys
import os

import torchaudio
import librosa

sys.path.append("../multimodal_data_preprocessing")
from utils.audio_extraction import *


class MetaDataReader:
    def __init__(self, path_to_metadata, audio_name):
        self.metadata_path = path_to_metadata
        # print(self.metadata_path)
        self.path, self.participant = path_to_metadata.rsplit("/", 1)
        self.participant = self.participant.split(".")[0]

        self.audio_path = f"{self.path}/{audio_name}"
        self.audio = audio_name

        self.transcribed_utts = self._get_json_lines()

        self.multiple_participants = False

    def _get_json_lines(self):
        # whether there are multiple participants in a single metadata file
        all_final_preds = []

        with open(self.metadata_path, 'r') as jf:
            for i in range(2):
                next(jf)
            for line in jf:
                try:
                    line = json.loads(line)
                    if line["data"]["is_final"]:
                        all_final_preds.append(line)
                # if there are 3+ non-json lines at start
                # there are 2+ participants in the file
                except json.decoder.JSONDecodeError:
                    self.multiple_participants = True
                    continue

        return all_final_preds

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

    def get_opensmile_feats_for_utt(self, utt_json):
        # get opensmile features for one utterance from its json
        # get start time
        start_time = utt_json["data"]["features"]["word_messages"][0]["start_time"]
        # get end time
        end_time = utt_json["data"]["features"]["word_messages"][-1]["end_time"]

        extracted_wav_path = self._split_audio_using_transcriptions(start_time, end_time)
        extracted_path, extracted_wav = extracted_wav_path.rsplit("/", 1)
        extracted_wav_name = extracted_wav.split(".wav")[0]
        csv_name = f"{extracted_wav_name}_IS13.csv"

        extractor = ExtractAudio(extracted_path, extracted_wav, f"{self.path}/IS13",
                                 smilepath="../../opensmile-3.0")

        utt_json["opensmile_feats"] = extractor.save_acoustic_csv("IS13", csv_name)

    def get_opensmile_feats_for_all_utts(self):
        for utt in self.transcribed_utts:
            self.get_opensmile_feats_for_utt(utt)


if __name__ == "__main__":
    mpath = "../../PROJECTS/ToMCAT/Evaluating_modelpredictions/data_from_speechAnalyzer/used_for_evaluating_model_results"

    metadata = [f"{mpath}/421_vid7.5boost/T000452.metadata_replay",
                f"{mpath}/422_vid7.5boost/T000451.metadata_replay",
                f"{mpath}/423_vid7.5boost/T000452.metadata_replay",
                f"{mpath}/424_vid7.5boost/T000455.metadata_replay",
                f"{mpath}/425_vid7.5boost/T000455.metadata_replay",
                f"{mpath}/426_vid7.5boost/T000455.metadata_replay",
                f"{mpath}/430_vid7.5boost/T000458.metadata_replay",
                f"{mpath}/431_vid7.5boost/T000458.metadata_replay",
                f"{mpath}/432_vid7.5boost/T000458.metadata_replay"]
    wav = [
        "study-3_spiral-3_pilot_NotHSRData_ClientAudio_Trial-T000452_Team-TM000075_Member-P000421_CondBtwn-ASI-CMU-CRA_CondWin-na_Vers-1.wav",
        "study-3_spiral-3_pilot_NotHSRData_ClientAudio_Trial-T000451_Team-TM000075_Member-P000422_CondBtwn-ASI-CMU-CRA_CondWin-na_Vers-1.wav",
        "study-3_spiral-3_pilot_NotHSRData_ClientAudio_Trial-T000452_Team-TM000075_Member-P000423_CondBtwn-ASI-CMU-CRA_CondWin-na_Vers-1.wav",
        "study-3_spiral-3_pilot_NotHSRData_ClientAudio_Trial-T000455_Team-TM000076_Member-P000424_CondBtwn-ASI-DOLL-SIFT_CondWin-na_Vers-1.wav",
        "study-3_spiral-3_pilot_NotHSRData_ClientAudio_Trial-T000455_Team-TM000076_Member-P000425_CondBtwn-ASI-DOLL-SIFT_CondWin-na_Vers-1.wav",
        "study-3_spiral-3_pilot_NotHSRData_ClientAudio_Trial-T000455_Team-TM000076_Member-P000426_CondBtwn-ASI-DOLL-SIFT_CondWin-na_Vers-1.wav",
        "study-3_spiral-3_pilot_NotHSRData_ClientAudio_Trial-T000458_Team-TM000078_Member-P000430_CondBtwn-ASI-SIFT-USC_CondWin-na_Vers-1.wav",
        "study-3_spiral-3_pilot_NotHSRData_ClientAudio_Trial-T000458_Team-TM000078_Member-P000431_CondBtwn-ASI-SIFT-USC_CondWin-na_Vers-1.wav",
        "study-3_spiral-3_pilot_NotHSRData_ClientAudio_Trial-T000458_Team-TM000078_Member-P000432_CondBtwn-ASI-SIFT-USC_CondWin-na_Vers-1.wav"
    ]

    for i, item in enumerate(metadata):
        print(item)
        reader = MetaDataReader(item, wav[i])

        # reader = MetaDataReader(metadata, wav)

        reader.get_opensmile_feats_for_all_utts()
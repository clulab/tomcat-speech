import os
from pathlib import Path
import re
def mfcc_wav(
    smile_directory,
    file_directory,
    acoustic_feature_set="IS10",
    #savedir
):
    p = Path('~').expanduser()
    smile_path = p / smile_directory
    path_to_files = p / file_directory

    """
    Extract mfcc values by recursively searching for all .wav files in subdirectories and 
    saving a corresponding .csv file in an output folder, appropriately named.
    """
    folder_list = [ f.name for f in os.scandir(path_to_files) if f.is_dir() ]
    # print(folder_list)
    # folder_list = [x[0] for x in os.walk(path_to_files)]
    for folder in folder_list:
        # print(folder)
        input_dir = path_to_files / folder
        savedir = path_to_files / folder / "output"
        os.system('if [ ! -d "{0}" ]; then mkdir -p {0}; fi'.format(savedir))
        for audio_file in os.listdir(folder):
            if re.match(".*wav", audio_file):
                audio_name = audio_file.split(".wav")[0]
                audio_save_name = str(folder) + "_" + str(audio_name) + "_" + acoustic_feature_set + ".csv"
                extractor = ExtractAudio(
                    input_dir, audio_file, savedir, smile_path
                )
                extractor.save_acoustic_csv(
                    feature_set=acoustic_feature_set, savename=audio_save_name
                )

class ExtractAudio:
    """
    Takes audio and extracts features from it using openSMILE
    """
    def __init__(self, path, audiofile, savedir, smilepath):
        self.path = path
        self.afile = path / audiofile
        self.savedir = savedir
        self.smile = smilepath
    def save_acoustic_csv(self, feature_set, savename):
        """
        Get the CSV for set of acoustic features for a .wav file
        feature_set : the feature set to be used
        savename : the name of the saved CSV
        Saves the CSV file
        """
        # todo: can all of these take -lldcsvoutput ?
        if feature_set == "IS09":
            fconf = "IS09_emotion.conf"
        elif feature_set == "IS10":
            fconf = "IS10_paraling.conf"
        elif feature_set == "IS12":
            fconf = "IS12_speaker_trait.conf"
        elif feature_set == "IS13":
            fconf = "IS13_ComParE.conf"
        else:
            fconf = "IS09_emotion.conf"
            # fconf = "emobase.conf"
        # check to see if save path exists; if not, make it
        # os.system('if [ ! -d "{0}" ]; then mkdir -p {0}; fi'.format(self.savedir))
        # run openSMILE
        os.system(
            "{0}/SMILExtract -C {0}/config/{1} -I {2} -lldcsvoutput {3}/{4}".format(
                self.smile, fconf, self.afile, self.savedir, savename
            )
        )
if __name__ == "__main__":
    
    file_directory = "Downloads/RAVDESS"
    smile_directory = "opensmile-2.3.0"

    mfcc_wav(smile_directory,file_directory)
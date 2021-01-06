# transcribe all datasets using sphinx with speech_recognition/sphinx_sr.py
# assumes data files are WAV formatted and organized in directories

from speech_recognition import sphinx_sr
import os


class DatasetTranscriber:
    """
    Transcribes wav files for datasets using pocketsphinx
    currently written for MELD, MUStARD, and ChaLearn
    param dataset should be one of these three
    """
    def __init__(self, dataset, location, extensions=None):
        self.dataset = dataset.lower()  # options: 'meld', 'mustard', 'chalearn'
        self.location = location
        # get list of extensions
        if type(self.extensions) is not str:
            self.extensions = extensions
        else:
            self.extensions = []
            self.extensions.append(extensions)

    def read_in_current_files(self, current_file_location):
        pass

    def transcribe(self, save_location):
        """
        transcribe all available files in the specified location
        """
        if self.extensions is not None:
            for ext in self.extensions:
                # get the location of each dir with files
                location = f"{self.location}/{ext}"
                # find wav files
                for wavfile in os.listdir(location):
                    if wavfile.endswith(".wav"):
                        # transcribe wav files
                        full_path = os.path.join(location, wavfile)
                        transcription = sphinx_sr.transcribe_file(full_path)
                        # save transcribed wav files to new files
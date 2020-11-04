import speech_recognition as sr
import sys

input_file = sys.argv[1]


def transcribe_file(input_file):

    r = sr.Recognizer()
    stim = sr.AudioFile(input_file)
    with stim as source:
        audio = r.record(source)

    transcription = r.recognize_sphinx(audio)

    return transcription


if __name__ == "__main__":
    result = transcribe_file(input_file)
    print(result)
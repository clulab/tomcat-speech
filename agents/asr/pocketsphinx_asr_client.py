import time
from asr_client import ASRClient
import speech_recognition as sr

class PocketSphinxASRClient(ASRClient):
    def __init__(self, rate, chunk_size):
        super().__init__()
        self.recognizer = sr.Recognizer()
        self.source = sr.Microphone(
            sample_rate=rate, chunk_size=chunk_size
        )

    def run(self):
        stop_listening = self.recognizer.listen_in_background(
            self.source, self.callback
        )
        while True:
            time.sleep(0.1)

        stop_listening(wait_for_stop=False)

    def callback(self, recognizer, audio_data):
        transcript = recognizer.recognize_sphinx(audio_data)
        self.publish_transcript(transcript, True, "PocketSphinx")

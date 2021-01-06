from typing import Optional
from logging import info
from utils import get_current_time
from asr_client import ASRClient
import google.cloud.speech

class GoogleASRClient(ASRClient):
    def __init__(
        self,
        audiostream,
        rate: int,
        chunk_size: Optional[int] = None,
        participant_id=None,
    ):
        super().__init__()
        self.rate = rate
        self.chunk_size = (
            int(self.rate / 10) if chunk_size is None else chunk_size
        )
        self.language_code = "en_US"
        self.speech_client = google.cloud.speech.SpeechClient()
        self.stream = audiostream
        # Google Cloud Speech has a limit of 5 minutes for streaming recognition
        # requests (https://cloud.google.com/speech-to-text/quotas)
        # We set a streaming limit of 4 minutes just to be on the safe side.
        self.streaming_limit = 240000
        self.participant_id = participant_id

        recognition_config = google.cloud.speech.RecognitionConfig(
            audio_channel_count=1,
            encoding=google.cloud.speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.rate,
            language_code=self.language_code,
        )

        self.streaming_recognition_config = (
            google.cloud.speech.StreamingRecognitionConfig(
                config=recognition_config, interim_results=True
            )
        )

    def run(self):
        info(
            f"Running Google ASR client for participant {self.participant_id}."
        )
        with self.stream as stream:
            while not stream.closed:

                stream.audio_input = []

                audio_generator = stream.generator()

                requests = (
                    google.cloud.speech.StreamingRecognizeRequest(
                        audio_content=content
                    )
                    for content in audio_generator
                )

                responses = self.speech_client.streaming_recognize(
                    self.streaming_recognition_config, requests
                )

                self.listen_print_loop(responses)
                info("Finished listen print loop")

                if stream.result_end_time > 0:
                    stream.final_request_end_time = stream.is_final_end_time

                stream.result_end_time = 0
                stream.last_audio_input = stream.audio_input
                stream.audio_input = []
                stream.restart_counter = stream.restart_counter + 1
                stream.new_stream = True

    def listen_print_loop(self, responses):
        """Iterates through server responses and prints them.

        The responses passed is a generator that will block until a response
        is provided by the server.

        Each response may contain multiple results, and each result may contain
        multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
        print only the transcription for the top alternative of the top result.

        In this case, responses are provided for interim results as well. If the
        response is an interim one, print a line feed at the end of it, to allow
        the next result to overwrite it, until the response is a final one. For the
        final one, print a newline to preserve the finalized transcription.
        """
        info("Entered listen_print_loop")
        for response in responses:

            if (
                get_current_time() - self.stream.start_time
                > self.streaming_limit
            ):
                self.stream.start_time = get_current_time()
                break

            if not response.results:
                continue

            # The `results` list is consecutive. For streaming, we only care about
            # the first result being considered, since once it's `is_final`, it
            # moves on to considering the next utterance.
            result = response.results[0]
            if not result.alternatives:
                continue

            # Display the transcription of the top alternative.
            transcript = result.alternatives[0].transcript

            result_seconds = 0
            result_micros = 0

            if result.result_end_time.seconds:
                result_seconds = result.result_end_time.seconds

            if result.result_end_time.microseconds:
                result_micros = result.result_end_time.microseconds

            self.stream.result_end_time = int(
                (result_seconds * 1000) + (result_micros / 1000)
            )

            corrected_time = (
                self.stream.result_end_time
                - self.stream.bridging_offset
                + (self.streaming_limit * self.stream.restart_counter)
            )

            if result.is_final:
                self.publish_transcript(transcript, "Google")
                self.stream.is_final_end_time = self.stream.result_end_time
                self.stream.last_transcript_was_final = True
            else:
                self.stream.last_transcript_was_final = False

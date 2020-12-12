#!/usr/bin/env python

import asyncio
import websockets
import logging
from logging import debug, info
from six.moves import queue
import google.cloud.speech
from dataclasses import dataclass, asdict, field, InitVar
import json
from paho.mqtt.client import Client as MQTTClient
from functools import partial


class AudioStream(object):
    """Opens a stream as a generator yielding the audio chunks."""

    def __init__(self, buff):
        # Create a thread-safe buffer of audio data
        self._buff = buff
        self.closed = True

    def __enter__(self):
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)

    def _fill_buffer(self, in_data):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        # return None

    def generator(self):
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                debug("Chunk is None")
                return
            debug("Chunk is not None")
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b"".join(data)


@dataclass(frozen=True)
class TA3Header(object):
    timestamp: str
    message_type: str = "observation"
    version: str = "0.1"


@dataclass(frozen=True)
class TA3Msg(object):
    timestamp: str
    experiment_id: str = None
    trial_id: str = None
    version: str = "0.1"
    source: str = "tomcat_asr_agent"
    sub_type: str = "asr"


@dataclass(frozen=True)
class TA3Data(object):
    text: str
    asr_system: str


@dataclass
class TA3Message(object):
    """Class to represent a TA3 testbed message."""

    data: TA3Data
    header: TA3Header = field(init=False)
    msg: TA3Msg = field(init=False)

    def __post_init__(self):
        timestamp: str = datetime.datetime.utcnow().isoformat() + "Z"
        self.header = TA3Header(timestamp)
        self.msg = TA3Msg(timestamp)


class ASRClient(object):
    def __init__(self, args):
        self.use_mqtt = args.use_mqtt
        if self.use_mqtt:
            # Set up the Paho MQTT client.
            self.mqtt_client = MQTTClient()
            self.mqtt_client.connect(args.host, args.port)
            self.publish_topic = args.publish_topic

    def publish_transcript(self, transcript, asr_system):
        ta3_data = TA3Data(transcript, asr_system)
        json_message_str = json.dumps(asdict(TA3Message(ta3_data)))
        if self.use_mqtt:
            self.mqtt_client.publish(self.publish_topic, json_message_str)
        else:
            print(json_message_str)
            # We call sys.stdout.flush() to make this program work with piping,
            # for example, through the jq program.
            sys.stdout.flush()


class ASRClient(object):
    def __init__(self, args):
        self.use_mqtt = args.use_mqtt
        if self.use_mqtt:
            # Set up the Paho MQTT client.
            self.mqtt_client = MQTTClient()
            self.mqtt_client.connect(args.host, args.port)
            self.publish_topic = args.publish_topic

    def publish_transcript(self, transcript, asr_system):
        ta3_data = TA3Data(transcript, asr_system)
        json_message_str = json.dumps(asdict(TA3Message(ta3_data)))
        if self.use_mqtt:
            self.mqtt_client.publish(self.publish_topic, json_message_str)
        else:
            print(json_message_str)
            # We call sys.stdout.flush() to make this program work with piping,
            # for example, through the jq program.
            sys.stdout.flush()


class GoogleASRClient(ASRClient):
    def __init__(self, audiostream):
        self.rate = 16000
        self.chunk_size = 1600
        self.language_code = "en-US"
        self.speech_client = google.cloud.speech.SpeechClient()
        self.audiostream = audiostream

        recognition_config = google.cloud.speech.RecognitionConfig(
            encoding=google.cloud.speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.rate,
            language_code=self.language_code,
        )

        self.streaming_recognition_config = (
            google.cloud.speech.StreamingRecognitionConfig(
                config=recognition_config, interim_results=True
            )
        )

    async def run(self):
        info("Running google asr client")
        with self.audiostream as audio_stream:
            audio_generator = audio_stream.generator()

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
        for response in responses:
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

            if result.is_final:
                self.publish_transcript(transcript, "Google")


chunk_queue = queue.Queue()


def generator():
    while True:
        chunk = chunk_queue.get()
        if chunk is None:
            return
        data = [chunk]

        # Now consume whatever other data's still buffered.
        while True:
            try:
                chunk = chunk_queue.get(block=False)
                if chunk is None:
                    return
                data.append(chunk)
            except queue.Empty:
                break

        yield b"".join(data)


def process_chunk(chunk):
    global chunk_queue
    chunk_queue.put(chunk)
    info("Chunk processed.")


async def message_handler(websocket, path):
    async for chunk in websocket:
        info(f"Received chunk of size {len(chunk)} bytes from browser.")
        process_chunk(chunk)

def listen_print_loop(responses):
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
    print('listen print loop started')
    for response in responses:
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
        print(transcript)


def process_chunks():
    speech_client = google.cloud.speech.SpeechClient()

    recognition_config = google.cloud.speech.RecognitionConfig(
        encoding=google.cloud.speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en_US",
    )

    streaming_recognition_config = (
        google.cloud.speech.StreamingRecognitionConfig(
            config=recognition_config, interim_results=True
        )
    )

    audio_generator = generator()
    requests = (
        google.cloud.speech.StreamingRecognizeRequest(audio_content=chunk)
        for chunk in audio_generator
    )

    responses = speech_client.streaming_recognize(
        streaming_recognition_config, requests
    )
    listen_print_loop(responses)





if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    info("Starting server")
    import threading
    x = threading.Thread(target=process_chunks)
    x.start()

    asyncio.gather(
        websockets.serve(message_handler, "127.0.0.1", 9000),
        # process_chunks(),
    )
    asyncio.get_event_loop().run_forever()

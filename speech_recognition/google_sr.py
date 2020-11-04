import sys

sample_rate = sys.argv[1]
input_file = sys.argv[2]


def transcribe_file(speech_file, sample_rate):
    """Transcribe the given audio file."""
    from google.cloud import speech
    import io

    client = speech.SpeechClient()

    with io.open(speech_file, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate,
        language_code="en-US",
    )

    response = client.recognize(config=config, audio=audio)

    # Each result is for a consecutive portion of the audio. Iterate through
    # them to get the transcripts for the entire audio file.

    for result in response.results:
        # The first alternative is the most likely one for this portion.
        text_conf = "%s\t%s" % (result.alternatives[0].transcript, result.alternatives[0].confidence)
    # print(response)
    # print(response.results)

    return text_conf


if __name__ == "__main__":
    result = transcribe_file(input_file, sample_rate)
    print(result)

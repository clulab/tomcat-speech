We define 'agents' as programs that are capable of processing streaming input
and producing streaming output.

Docker Compose instructions
---------------------------

To launch the containerized multi-participant speech recognition services, do
the following:

- Set the GOOGLE_APPLICATION_CREDENTIALS environment variable to point to your
  credentials file for the Google Cloud Speech recognition API.
- Ensure that an MQTT broker is running on port 1883.
- Run `docker-compose up --build`
- Go to `localhost:8000` in your browser to access the webmic app. If you want
  to specify a particular participant id, you can do so with the URL query
  parameter, `id`. For example, navigating to `localhost:8000?id=myName` will
  set the participant ID to `myName`. This will affect the `participant_id`
  field in the messages, as well as the name of the saved raw audio file (which
  will be created in the `asr` directory).
- Click the `Connect` button.
- Start speaking. The output ASR messages will be published to the `agents/asr`
  topic on the message bus.

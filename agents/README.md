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
- Go to `localhost:8000` in your browser to access the webmic app
- Click the `Connect` button.

import sys
import json
from dataclasses import asdict
from paho.mqtt.client import Client as MQTTClient
from messages import Data, Message

class ASRClient(object):
    def __init__(
        self,
        use_mqtt: bool=False,
        mqtt_host="localhost",
        mqtt_port=1883,
        publish_topic="agents/asr",
    ):
        self.use_mqtt = use_mqtt
        if self.use_mqtt:
            # Set up the Paho MQTT client.
            self.mqtt_client = MQTTClient()
            self.mqtt_client.connect(mqtt_host, mqtt_port)
            self.publish_topic = publish_topic

    def publish_transcript(self, transcript, asr_system):
        ta3_data = Data(transcript, asr_system)
        json_message_str = json.dumps(asdict(Message(ta3_data)))
        if self.use_mqtt:
            self.mqtt_client.publish(self.publish_topic, json_message_str)
        else:
            print(json_message_str)
            # We call sys.stdout.flush() to make this program work with piping,
            # for example, through the jq program.
            sys.stdout.flush()

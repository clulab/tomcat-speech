import paho.mqtt.client as mqtt
import json
import argparse
from edit_dist import PhonemicMagic

def on_connect(client, userdata, flags, rc):
    client.subscribe("agent/asr/final")

def on_message(client, userdata, msg):
    obj = json.loads(msg.payload.decode("utf-8")
    
    utterance = obj["data"]["text"]
    obj["data"]["repaired"] = phonemic_helper.process_utterance(utterance)
    
    client.publish("agent/asr/repaired", json.dumps(obj))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mqtt_host", type=str, default="mosquitto")
    parser.add_argument("--mqtt_port", type=int, default=1883)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args() 
    phonemic_helper = PhonemicMagic("cmu_feature_key.csv", "cmudict-0.7b.txt","stb_files/CELEXEnglish.fea.stb" ,"domain_words.csv" )

    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(args.mqtt_host, args.mqtt_port, 60)
    client.loop_forever()


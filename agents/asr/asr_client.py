import sys
import json
from dataclasses import asdict
from messages import Data, Message, Msg


class ASRClient(object):
    def __init__(
        self,
        participant_id=None,
    ):
        self.participant_id = participant_id

    def publish_transcript(self, transcript, asr_system):
        ta3_data = Data(transcript, asr_system)
        msg = Msg(participant_id = self.participant_id)
        json_message_str = json.dumps(
            asdict(Message(ta3_data, msg = msg))
        )
        print(json_message_str)
        # We call sys.stdout.flush() to make this program work with piping,
        # for example, through the jq program.
        sys.stdout.flush()

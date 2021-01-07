import datetime
from dataclasses import dataclass, field
from typing import Optional

@dataclass(frozen=True)
class Header(object):
    timestamp: str
    message_type: str = "observation"
    version: str = "0.1"


@dataclass(frozen=True)
class Msg(object):
    timestamp: str
    experiment_id: Optional[str] = None
    participant_id: Optional[str] = None
    trial_id: Optional[str] = None
    version: str = "0.1"
    source: str = "tomcat_asr_agent"
    sub_type: str = "asr"


@dataclass(frozen=True)
class Data(object):
    text: str
    asr_system: str


@dataclass
class Message(object):
    """Class to represent a  testbed message."""

    data: Data
    header: Header = field(init=False)
    msg: Msg = field(init=False)

    def __post_init__(self):
        timestamp: str = datetime.datetime.utcnow().isoformat() + "Z"
        self.header = Header(timestamp)
        self.msg = Msg(timestamp)

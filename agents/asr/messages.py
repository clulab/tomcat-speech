import datetime
from dataclasses import dataclass, field
from typing import Optional

@dataclass(frozen=True)
class Header(object):
    timestamp: Optional[str] = datetime.datetime.utcnow().isoformat() + "Z"
    message_type: str = "observation"
    version: str = "0.1"


@dataclass(frozen=True)
class Msg(object):
    timestamp: Optional[str] = datetime.datetime.utcnow().isoformat() + "Z"
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
    """Class to represent a testbed message."""

    data: Data
    header: Header = Header()
    msg: Msg = Msg()

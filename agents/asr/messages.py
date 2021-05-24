import datetime
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Header(object):
    timestamp: str = field(init=False)
    message_type: str = "observation"
    version: str = "0.1"

    def __post_init__(self):
        object.__setattr__(
            self, "timestamp", datetime.datetime.utcnow().isoformat() + "Z"
        )


@dataclass(frozen=True)
class Msg(object):
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.utcnow().isoformat() + "Z"
    )
    experiment_id: Optional[str] = None
    trial_id: Optional[str] = None
    version: str = "0.1"
    source: str = "tomcat_asr_agent"
    sub_type: str = "asr"


@dataclass(frozen=True)
class Data(object):
    text: str
    is_final: bool
    asr_system: str
    participant_id: Optional[str] = None


@dataclass
class Message(object):
    """Class to represent a testbed message."""

    data: Data
    msg: Msg
    header: Header = field(init=False)

    def __post_init__(self):
        self.header = Header()

## TOPIC

agents/asr

## Message Fields

| Field Name | Type | Description
| --- | --- | ---|
| header.timestamp | string | Timestamp of when the data was generated in ISO 8601 format: YYYY-MM-DDThh:mm:ss.sssz |
| header.message_type | string | One of the defined message types |
| header.version | string | The version of the message type object |
| msg.timestamp | string | Timestamp of when the data was generated in ISO 8601 format: YYYY-MM-DDThh:mm:ss.sssz |
| msg.experiment_id | string | The experiment id this message is associated with |
| msg.trial_id | string | The trial id this message is associate with |
| msg.version | string | The version of the sub_type format |
| msg.source | string | The name of the component that published this data |
| msg.sub_type | string | The subtype of the data. This field describes the format of this particular type of data |
| data.text | string | NA |
| data.is_final | boolean | NA |
| data.asr_system | string | NA |
| data.participant_id | string | NA |

## Message Example

```json
{"header": {
    "timestamp": "2019-12-26T12:47:23.1234Z",
    "message_type": "event",
    "version": "0.4"
    },
"msg": {
    "timestamp" : "2019-12-26T14:05:02.1412Z",
    "experiment_id" : "523e4567-e89b-12d3-a456-426655440000",
    "trial_id" : "123e4567-e89b-12d3-a456-426655440000",
    "version" : "0.1",
    "source" : "tomcat_asr_agent",
    "sub_type" : "asr"
    },
"data": {
    "text" : "",
    "is_final" : "",
    "asr_system" : "",
    "participant_id" : ""
    }
}

```

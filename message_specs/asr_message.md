## TOPIC

agent/asr

## Message Fields

| Field Name          | Type    | Description                                                                              |
| ---                 | ---     | ---                                                                                      |
| header.timestamp    | string  | Timestamp of when the data was generated in ISO 8601 format: YYYY-MM-DDThh:mm:ss.sssz    |
| header.message_type | string  | One of the defined message types                                                         |
| header.version      | string  | The version of the message type object                                                   |
| msg.timestamp       | string  | Timestamp of when the data was generated in ISO 8601 format: YYYY-MM-DDThh:mm:ss.sssz    |
| msg.experiment_id   | string  | The experiment id this message is associated with                                        |
| msg.trial_id        | string  | The trial id this message is associated with                                             |
| msg.version         | string  | The version of the sub_type format                                                       |
| msg.source          | string  | The name of the component that published this data                                       |
| msg.sub_type        | string  | The subtype of the data. This field describes the format of this particular type of data |
| data.text           | string  | The transcription returned from the ASR system                                           |
| data.is_final       | boolean | Indicates whether the transcription is an intermediate or final transcription            |
| data.asr_system     | string  | The system used by the agent for automatic speech recognition                            |
| data.participant_id | string  | The participant id this data is assosiated with                                          |

## Message Example

```json
{
  "data": {
    "text": "I am going to save a green victim.",
    "asr_system": "Google",
    "is_final": true,
    "participant_id": "participant_1"
  },
  "header": {
    "timestamp": "2021-01-19T23:27:58.633076Z",
    "message_type": "observation",
    "version": "0.1"
  },
  "msg": {
    "timestamp": "2021-01-19T23:27:58.633967Z",
    "experiment_id": "e2a3cb96-5f2f-11eb-8971-18810ee8274e",
    "trial_id": "256d1b4a-d81d-465d-8ef0-2162ff96e204",
    "version": "0.1",
    "source": "tomcat_asr_agent",
    "sub_type": "asr"
  }
}
```

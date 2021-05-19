import json

from helper import preprocessing
from helper.wav2vec_helper import load_train_test, extract_all_chars, create_vocab
from helper.wav2vec_helper import load_tokenizer, load_extractor, extract_speech_feature
from helper.wav2vec_helper import DataCollatorCTCWithPadding

from datasets import load_metric

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer

from os import path

from dataclasses import dataclass, field

from typing import Any, Dict, List, Optional, Union

import torch 
import pickle

# if path.exists('/data/seongjinpark/chalearn_asr.pk'):
#     chalearn_prepared = pickle.load(open("/data/seongjinpark/chalearn_asr.pk", "rb"))
# else:
train_path = "/data/seongjinpark/chalearn/train/audio_16"
test_path = "/data/seongjinpark/chalearn/dev/audio_16"
# train_text = "/data/seongjinpark/chalearn/train/gold_and_utts.tsv"
# test_text = "/data/seongjinpark/chalearn/dev/gold_and_utts.tsv"
train_text = "/work/seongjinpark/tomcat-speech/speech_recognition/data/chalearn/train.tsv"
test_text = "/work/seongjinpark/tomcat-speech/speech_recognition/data/chalearn/dev.tsv"

train_csv = "/work/seongjinpark/tomcat-speech/speech_recognition/data/chalearn/train.csv"
test_csv = "/work/seongjinpark/tomcat-speech/speech_recognition/data/chalearn/dev.csv"

vocab_json = "/work/seongjinpark/tomcat-speech/speech_recognition/data/chalearn/vocab.json"

train_dict = preprocessing.generate_dict(train_path, train_text, train_csv, header=False)
test_dict = preprocessing.generate_dict(test_path, test_text, test_csv, header=False)

chalearn_data = load_train_test(train_csv , test_csv)

vocabs = chalearn_data.map(extract_all_chars, 
                            batched=True, 
                            batch_size=-1,
                            keep_in_memory=True,
                            remove_columns=chalearn_data.column_names["train"])

vocab_dict = create_vocab(vocabs=vocabs, output_name=vocab_json)

tokenizer = load_tokenizer(vocab_json=vocab_json, unk_token="[UNK]", pad_token="[PAD]", delim="|")

feature_extractor = load_extractor(return_attention=False)

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

chalearn_data = chalearn_data.map(extract_speech_feature, remove_columns=chalearn_data.column_names["train"], num_proc=4)

def prepare_dataset(batch):
    # check that all files have the correct sampling rate
    assert (
        len(set(batch["sampling_rate"])) == 1
    ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."

    batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values

    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids
    
    return batch

chalearn_prepared = chalearn_data.map(prepare_dataset, remove_columns=chalearn_data.column_names["train"], batch_size=8, num_proc=4, batched=True)
# pickle.dump(chalearn_prepared, open("/data/seongjinpark/chalearn_asr.pk", "wb"))

@dataclass
class DataCollatorCTCWithPadding:
    '''
    Data collator that will dynamically pad the inputs received.
    params:
    @processor (:class:`~transformers.Wav2Vec2Processor`)
        The processor used for proccessing the data.
    @padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
        Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
        among:
        * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
            sequence if provided).
        * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
            maximum acceptable input length for the model if that argument is not provided.
        * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
            different lengths).
    @max_length (:obj:`int`, `optional`):
        Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
    @max_length_labels (:obj:`int`, `optional`):
        Maximum length of the ``labels`` returned list and optionally padding length (see above).
    @pad_to_multiple_of (:obj:`int`, `optional`):
        If set will pad the sequence to a multiple of the provided value.
        This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
        7.5 (Volta).
    '''

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

wer_metric = load_metric("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-large-960h", 
    gradient_checkpointing=True, 
    ctc_loss_reduction="mean", 
    pad_token_id=processor.tokenizer.pad_token_id,
)

model.freeze_feature_extractor()

training_args = TrainingArguments(
  # output_dir="/content/gdrive/MyDrive/wav2vec2-base-timit-demo",
  output_dir="./wav2vec2-large-chalearn",
  group_by_length=True,
  per_device_train_batch_size=4,
  evaluation_strategy="steps",
  num_train_epochs=30,
  fp16=True,
  save_steps=500,
  eval_steps=500,
  logging_steps=500,
  learning_rate=1e-4,
  weight_decay=0.005,
  warmup_steps=1000,
  save_total_limit=2,
  eval_accumulation_steps=1,
)

trainer = Trainer(
    model=model, 
    data_collator=data_collator, 
    args=training_args, 
    compute_metrics=compute_metrics, 
    train_dataset=chalearn_prepared["train"],
    eval_dataset=chalearn_prepared["test"], 
    tokenizer=processor.feature_extractor,
)

trainer.train()
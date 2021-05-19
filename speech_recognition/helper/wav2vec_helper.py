import json
import soundfile as sf

from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2Processor
from transformers import Wav2Vec2FeatureExtractor

from datasets import load_dataset, load_metric, Dataset
from dataclasses import dataclass, field

import torch

from typing import Any, Dict, List, Optional, Union

def load_train_test(train_csv, test_csv):
    return load_dataset('csv', data_files={'train': train_csv, 'test': test_csv})
    # return Dataset.from_dict({'train': train_dict, 'test': test_dict})

def extract_all_chars(batch):
    all_text = " ".join(batch["text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

def create_vocab(vocabs, output_name):
    vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["test"]["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}

    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]

    # Add unknown and padding characters
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    with open(output_name, 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)

    return vocab_dict

def load_tokenizer(vocab_json, unk_token, pad_token, delim):
    '''
    params
    @vocab_json: json file contains dictionary {letter: idx}
    @unk_token: token used to mark unknown words (e.g., "[UNK]")
    @pad_token: token used to mark padded words (e.g., "[PAD]")
    @delim: token used to delimit the words (e.g., "|")
    '''
    return Wav2Vec2CTCTokenizer(vocab_json, unk_token=unk_token, pad_token=pad_token, word_delimiter_token=delim)

def load_extractor(return_attention=False):
    '''
    param
    @return_attention: True when we use wav2vec-large-960-lv60
    '''
    return Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention=return_attention)

def extract_speech_feature(batch):
    speech_feat, sampling_rate = sf.read(batch["file"])
    batch["speech"] = speech_feat
    batch["sampling_rate"] = sampling_rate
    batch["target_text"] = batch["text"]
    return batch

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
import os
import re

from num2words import num2words

char_to_ignore_regex = '\[[A-z]+ [\d][\d]:[\d][\d]:[\d][\d]\]'
int_put_space = '([\d])([A-z\.\,\?\!])'


with open("/data/seongjinpark/chalearn/train/gold_and_utts.tsv", "r") as train:
    train_data = train.readlines()

with open("../data/chalearn/train.tsv", "w") as train_out:
    for i in range(1, len(train_data)):
        items = train_data[i].rstrip().split("\t")
        filename = items[0].replace(".mp4", "")
        try:
            sentence = items[9]
            sentence = re.sub(char_to_ignore_regex, '', sentence)
            sentence = re.sub(r'([\d])([A-z])', r'\1 \2', sentence)

            words = []
            for word in sentence.split():
                if word.isdigit():
                    num_word = num2words(int(word))
                    num_word = num_word.replace("-", " ")
                    # print(word, num_word)
                    words.append(num_word)
                else:
                    words.append(word)
            sentence = " ".join(words)
            # print(sentence)
        except:
            sentence = ""

        result = "%s\t%s\n" % (filename, sentence)
        # print(result)
        train_out.write(result)


with open("/data/seongjinpark/chalearn/dev/gold_and_utts.tsv", "r") as dev:
    dev_data = dev.readlines()

with open("../data/chalearn/dev.tsv", "w") as dev_out:
    for i in range(1, len(dev_data)):
        items = dev_data[i].rstrip().split("\t")

        filename = items[0].replace(".mp4", "")

        sentence = items[9]
        sentence = re.sub(char_to_ignore_regex, '', sentence)
        # sentence = re.sub(r'([\d])([A-z])', r'\1 \2', sentence)

        if re.search(r'[\d]', sentence):
            next
        else:
            result = "%s\t%s\n" % (filename, sentence)
            dev_out.write(result)
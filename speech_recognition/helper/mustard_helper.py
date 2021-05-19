import os
import re
from sklearn.model_selection import train_test_split

with open("/data/seongjinpark/MUStARD/mustard_utts.tsv", "r") as mustard:
    data = mustard.readlines()

mustard_dict = {}

for i in range(1, len(data)):
    items = data[i].rstrip().split("\t")
    filename = items[0]
    sentence = items[1]

    mustard_dict[filename] = sentence

train_files, test_files = train_test_split(list(mustard_dict.keys()), test_size=0.2, random_state=42)


with open("../data/mustard/train.tsv", "w") as train_out:
    for train_file in train_files:
        filename = train_file
        sentence = mustard_dict[filename]
        
        result = "%s\t%s\n" % (filename, sentence)
        # print(result)
        train_out.write(result)

with open("../data/mustard/test.tsv", "w") as dev_out:
    for test_file in test_files:
        filename = test_file
        sentence = mustard_dict[filename]
        result = "%s\t%s\n" % (filename, sentence)
        dev_out.write(result)
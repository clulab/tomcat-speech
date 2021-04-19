import re
from collections import Counter
# import pandas
import os.path
# import matplotlib.pyplot as plt

#extremely slow
def load_freq(freq_path):
    freq_list = {}
    f = open(freq_path)
    lines = f.readlines()
    f.close()
    print("file ready")

    for line in lines:
        word, freq = line.split("	")
        if re.match('((^\w*\'\w+$)|(^\w+\'\w*$)|(^\w+-*\w*$))', word):
            freq_list[word] = freq.rstrip("\n")

    return freq_list

def tokenize_input(string_of_text, option = "file"):
    if option == "file":
        if os.path.isfile(string_of_text):
            input = open(string_of_text, "r").read()
        else:
            print("filepath error")
    else:
        input = string_of_text
    transcripts = nlp(input)
    non_stop = [token.lower_ for token in transcripts
                if not token.is_space and not token.is_punct and not token.is_stop]

    return non_stop
def find_freq(list, words):
    output = {}
    missing = {}
    for word in words:
        if word not in output:
            if word in list:
                output[word] = int(list[word])
            else:
                missing[word] = None
        else:
            pass
    return dict(sorted(output.items(), key=lambda item:item[1])), missing

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

#extremely slow
def load_dict(cmu_path):
    cmu_dict = {}

    if not os.path.isfile("cmudict-0.7b.txt"):
        bash_command = "curl http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b >> cmudict-0.7b.txt"
        os.system(bash_command)

    f = open("cmudict-0.7b.txt", encoding="ISO-8859-1")
    lines = f.readlines()[56:-1]
    f.close()

    for line in lines:
        line = line.rstrip()
        key = line[0:line.find(" ")]
        value = line[line.find(" "):].lstrip()
        cmu_dict[key] = value

    return cmu_dict
def known(words,dict): return set(w for w in words if w in dict)


if __name__ == "__main__":
    print("initialised")
    #load gigawords:
    frequencies = load_freq("gigaword_lean_head.txt")

    print(len(frequencies))

    cmu = load_dict("cmudict-0.7b.txt")
    print("files loaded")

    # within text analysis:
    def remove_stop_sort(input):
        import spacy
        nlp = spacy.load('en_core_web_sm')
        print("spacy loaded")
        transcripts = nlp(input)
        no_stop = [token.lower_ for token in transcripts
                    if not token.is_space and not token.is_punct and not token.is_stop]
        return {input : no_stop}


    import spacy
    nlp = spacy.load('en_core_web_sm')
    print("spacy loaded")


    input = open("input.txt", "r").read()
    transcripts = nlp(input)
    bag = [token.lower_ for token in transcripts
           if not token.is_space and not token.is_punct]
    non_stop = [token.lower_ for token in transcripts
           if not token.is_space and not token.is_punct and not token.is_stop]

    assert bag != non_stop

    # within-oc analysis:
    word_freq_doc = dict(sorted(Counter(bag).most_common(), key=lambda item: item[1])) #use Counter.items() for ascending order
    print("most frequent words in doc:")
    n = 0
    for i in word_freq_doc:
        if n < 10:
            print(i, ":", word_freq_doc[i])
            n += 1


    word_freq_content = dict(sorted(Counter(non_stop).items(), key=lambda item: item[1]))
    print("most frequent content words in doc:")
    n = 0
    for i in word_freq_content:
        if n < 10:
            print(i, ":", word_freq_content[i])
            n += 1
    with open('gigaword_lean.txt', 'w') as file:
        for key in frequencies:
            x = key + "\t" + frequencies[key] + "\n"
            file.write(x)
    edits1("hello")

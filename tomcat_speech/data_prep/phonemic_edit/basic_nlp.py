import re
from collections import Counter
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
# def known(words,dict): return set(w for w in words if w in dict)
def remove_stops(string_of_text, option = "text"):
    if option == "file":
        if os.path.isfile(string_of_text):
            input = open(string_of_text, "r").read()
        else:
            print("filepath error")
    elif option == "text":
        input = string_of_text
    transcripts = nlp(input)
    output = []
    bag = []
    for token in transcripts:
        if not token.is_space and not token.is_punct and not (token.is_stop and token.text != "'s"):
            bag.append(token)
    count = 0
    for i in range(0, len(bag)):
        if bag[i].text == "'s":
            output[i-1-count] = str(output[i-1-count]) + str(bag[i].text) #this number is not ok
            count += 1
        else:
            output.append(str(bag[i]))

            # elif token.text == "'s": # then merge previous token and this one
        #     print(str(token.text))
        #     output.append(str(bag[x - 1]) + str(token.text))

    return output

# def find_candidates(token, k):
#     # remove stops
#     if self.is_stop(token):
#         return []
#     # get phonemic
#     phonemic = cmu_lookup(token)
#     if phonemic is None:
#         return []
#     # get freq
#     token_freq = self.get_frequency(token)
#     # compare to the domain words
#     scored_domain_words = self.compare_token(phonemic)
#     # List[(domain_word, score, freq)] -- not sorted yet
#     # prune ones above threshold
#     filtered = [x for x in scored_domain_words if x.score <= self.threshold and x.score > 0]
#     # sort by frequency
#     # return List(domain_word, score, freq)[0:k]
#     tokens = self.tokenize(utt, k=1)
#     candidates = [self.find_candidates(x) for x in utt]
#     # cadidates: List[List[(domain_word, score, freq)]]
#     for i, token in enumerate(tokens):
#         options = [token, candidate]
#         # Vincent do smart recursive thing here
#         scored = score(repaired) # ??
	# Output format: List[(alternative_transcription, score)] ← avg replacement, or inspired by the WER
    # Choose top 1 for each word_to_be_replaced, and orig, cartesian product


if __name__ == "__main__":
    print("initialised")
    #load gigawords:
    frequencies = load_freq("gigaword_lean_head.txt")
    print(len(frequencies))

    cmu = load_dict("cmudict-0.7b.txt")
    print("files loaded")

    import spacy
    nlp = spacy.load('en_core_web_sm')
    print("spacy loaded")
    # within text analysis:
    text = "John's simple routine's great! You should try it too."
    print(remove_stops(text))

    def remove_stop_sort(input):

        transcripts = nlp(input)
        no_stop = [token.lower_ for token in transcripts
                    if not token.is_space and not token.is_punct and not token.is_stop]
        return {input : no_stop}



    input = open("input.txt", "r").read()
    transcripts = nlp(input)
    #I'm losing possessives and contractions here: find a way to run CMU dict search on original words, not tokens.
    bag = [token.lower_ for token in transcripts
           if not token.is_space and not token.is_punct]
    non_stop = [token.lower_ for token in transcripts
           if not token.is_space and not token.is_punct and not token.is_stop]
    possessive = [bag[index - 1] + bag[index] for index, token in enumerate(bag) if token == "'s"]

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

import re
# import pandas
import os.path
import sys
import spacy
nlp = spacy.load('en_core_web_sm')

# if os.path.isfile("cmudict-0.7b.txt"):
#     print("Dictionary Accessed")
# else:
#     print("Dictionary does not exist. Downloading...")
#     bash_command = "curl http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b >> cmudict-0.7b.txt"
#     os.system(bash_command)

# initialise cmudict:
# f = open("cmudict-0.7b.txt","r",encoding = "ISO-8859-1")
# cmu_dict = f.readlines()
# f.close()
#
# # initialise phonemic key:
# phonemes = open("cmu_feature_key.csv","r",encoding = "ISO-8859-1")
# phoneme_dict = phonemes.readlines()
# phonemes.close()


# def capitalize(utt):
#     input = re.compile('[^\W_]+\'*[^\W_]*').findall(utt)
#     words = []
#     for i in input:
#         j = i.upper()
#         words.append(j)
#     return words
#
# def cmudict_search(lst):
#     out = []
#     for word in lst:
#         for line in cmu_dict:
#             reg = "^"+word+" \s"
#             if re.search(reg, line, re.I):
#                 result = line.rstrip('\n')
#                 y = [word, result.split("  ")[1]]
#                 out.append(y)
#
#     # check for absent words, save them with error message
#     s = []
#     for i in out:
#         s.append(i[0])
#     for j in lst:
#         if j not in s:
#             out.append([j, "pronunciation entry not found"])
#     missing_words = []
#     for i in out:
#         if i[1] == "pronunciation entry not found":
#             missing_words.append(i[0])
#     if len(missing_words) > 0:
#         print("some words were not found in the pronunciation dictionary")
#     return out, missing_words
#
# def phoneme(token):
#     phones = []
#     for i in token:
#         for line in phoneme_dict:
#             words = line.rstrip('\n').split("	")
#             print(words)
#             reg = "^" + words[0] + ".*"
#             if re.search(reg, i, re.I):
#                 print("found")
#                 if not re.match(i, "^AH"):
#                     re.sub(r'\d', '', i)
#                     sub = words[1]
#                     phones.append(sub)
#                 else:
#                     sub = words[1]
#                     phones.append(sub)
#     return phones
def load_freq(freq_path):
    freq_list = {}
    f = open(freq_path)
    lines = f.readlines()
    f.close()

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

if __name__ == "__main__":

    # domain = open("sample_data.txt", "r")
    # target = domain.readlines()
    # target = "Pycharm is good...But I need to really see this ?? ## !$% ^4@ ,. _ happen"
    # for i in target[:5]:
    #     line = i.rstrip('\n')
    #     word = line.split("    ")[0]
    #     g = capitalize(word)
    #     pronunciation = cmudict_search(g)
    #     # print(pronunciation)
    # output = phoneme('D AH1 Z AH0 N T')
    # print(output)

    #load gigawords:
    frequencies = load_freq("gigaword_lean_head.txt")
    # n=0
    # for i in frequencies:
    #     if n < 400:
    #         print(i, ":", frequencies[i])
    #         n +=1
    #get content words and frequency:
    non_stop = tokenize_input("input.txt", option = "file")
    input = open("input.txt", "r").read()
    transcripts = nlp(input)
    bag = [token.lower_ for token in transcripts
           if not token.is_space and not token.is_punct]
    # non_stop = [token.lower_ for token in transcripts
    #        if not token.is_space and not token.is_punct and not token.is_stop]
    # print("no. of words, :", len(non_stop))
    # print(bag[:100])
    possessive = [bag[index-1]+bag[index] for index, token in enumerate(bag) if token == "'s"]
    # print("possessives:", possessive[:100])
    from collections import Counter
    word_freq = Counter(bag)
    # print(word_freq)
    # word_freq_content = Counter(non_stop)
    # print(word_freq_content)
    # print("word_freq list length:", len(word_freq))
    # print("most common tokens:", word_freq.most_common(50))
    # print("most common content words:", word_freq_content.most_common(50))
    frequencies_input, missing_tokens = find_freq(frequencies, non_stop)
    n=0
    for i in frequencies_input:
        if n < 40:
            print(i, ":", frequencies_input[i])
            n +=1


# class ParseUtt:
#     def __init__(self, freq_path): #add input requirements here
#         #open these files
#         self.freq_list = self.load_freq(freq_path)
#
#     def load_freq(self, freq_path):
#         freq_list = {}
#         f = open(freq_path)
#         lines = f.readlines()
#         f.close()
#
#         for line in lines:
#             word, freq = line.split("	")
#             if re.match('((^\w*\'\w+$)|(^\w+\'\w*$)|(^\w+-*\w*$))', word):
#                 freq_list[word] = freq.rstrip("\n")
#
#         return freq_list
#
#     def find_freq(self,list, words):
#         output = {}
#         for word in words:
#             if word in list:
#                 print(word, list[word])
#                 output[word] = list[word]
#             else:
#                 output[word] = None
#         return output

##################################################################
    with open('gigaword_lean.txt', 'w') as file:
        for key in frequencies:
            x = key + "\t" + frequencies[key] + "\n"
            file.write(x)








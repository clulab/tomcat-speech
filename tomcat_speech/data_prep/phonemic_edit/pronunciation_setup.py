import re
import pandas
import os.path

if os.path.isfile("cmudict-0.7b.txt"):
    print("Dictionary Accessed")
else:
    print("Dictionary does not exist. Downloading...")
    bash_command = "curl http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b >> cmudict-0.7b.txt"
    os.system(bash_command)

# initialise cmudict:
f = open("cmudict-0.7b.txt","r",encoding = "ISO-8859-1")
cmu_dict = f.readlines()

# initialise phonemic key:
phones = pandas.read_csv(open("cmu_feature_key.csv",encoding = "ISO-8859-1"), sep = ":")


def capitalize(utt):
    input = re.compile('[^\W_]+').findall(utt)
    words = []
    for i in input:
        j = i.upper()
        words.append(j)
    return words

def cmudict_search(lst, missing_words):
    out = []
    for word in lst:
        for line in cmu_dict:
            reg = "^"+word+" \s"
            if re.search(reg, line, re.I):
                result = line.rstrip('\n')
                y = [word, result.split("  ")[1]]
                out.append(y)
    s = []
    for i in out:
        s.append(i[0])
    # print("success list:", s)
    for j in lst:
        if j not in s:
            out.append([j, "pronunciation entry not found"])
    missing_words = []
    for i in out:
        # print(i[0], i[1])
        if i[1] == 'pronunciation entry not found':
            missing_words.append(i[0])
    if len(missing_words) > 0:
        print("some words were not found in the pronunciation dictionary")
        # return missing_words
    return out, missing_words

# def missing(lst):
#     missing_words = []
#     for i in lst:
#         print(i[0], i[1])
#         if i[1] == 'pronunciation entry not found':
#             missing_words.append(i[0])
#     if len(missing_words) > 0:
#         return missing_words


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    domain = open("domain_words.csv", "r")
    target = domain.readlines()
    # target = "Pycharm is good...But I need to really see this ?? ## !$% ^4@ ,. _ happen"
    for i in target[:5]:
        line = i.rstrip('\n')
        word = line.split("    ")[0]
        g = capitalize(word)
        missing = []
        pronunciation = cmudict_search(g,missing)
        print(pronunciation)




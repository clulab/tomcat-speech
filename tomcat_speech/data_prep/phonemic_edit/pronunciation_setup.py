import sys
import re
import pandas

# initialise cmudict:
f = open("cmudict/cmudict-0.7b.txt","r",encoding = "ISO-8859-1")
x = f.readlines()   #y = f.read()

# initialise phonemic key:
phones = pandas.read_csv(open('SimilarityCalculator/cmu_feature_key.txt',encoding = "ISO-8859-1"), sep = ":")


def capitalize(utt):
    ls = utt.split(" ")
    input = re.compile('[^\W_]+').findall(utt)
    words = []
    for i in input:
        j = i.upper()
        # print(j)
        words.append(j)
    return words

def cmudict_search(lst):
    out = []
    for word in lst:
        for line in x:
            reg = "^"+word+" \s"
            if re.search(reg, line, re.I):
                y = [word, line.split("  ")[1].split("\n")[0]]
                out.append(y)
    s = []
    for i in out:
        s.append(i[0])
    for j in lst:
        if j not in s:
            out.extend([[j]])
    return out

def pronun(lst):
    cmu_pron = []
    for i in lst:
        if len(i)>=2:
            cmu_pron.append(i)
    return cmu_pron



# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    target = "Pycharm is good...But I need to really see this ?? ## !$% ^4@ ,. _ happen"
    g = capitalize(target)
    # print(g)
    pronunciation = cmudict_search(g)
    print(pronun(pronunciation))




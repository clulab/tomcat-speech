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
        for line in cmu_dict:
            reg = "^"+word+" \s"
            if re.search(reg, line, re.I):
                y = [word, line.split("  ")[1].split("\n")[0]]
                # print(y)
                out.append(y)
    # print(out)
    s = []
    for i in out:
        s.append(i[0])
    # print("success list:", s)
    for j in lst:
        if j not in s:
            print("word not found")
            # out.extend([[j]])
    return out

def pronun(lst):
    cmu_pron = []
    missing_words = []
    for i in lst:
        if len(i)>=2:
            cmu_pron.append(i)
        elif len(i)<2:
            missing_words.append(i)
    # print("list of available and non-available pronunciations created")
    return cmu_pron, missing_words


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    domain = open("domain_words.csv", "r")
    target = domain.readlines()
    # target = "Pycharm is good...But I need to really see this ?? ## !$% ^4@ ,. _ happen"
    for i in target:
        word = i.rstrip('\n')
        g = capitalize(word)
        pronunciation = cmudict_search(g)
        print(pronunciation)
        # print(pronun(pronunciation))




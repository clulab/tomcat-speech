# To Do:
# 1. read tsv, split lines, find utterance
# 2. split utterance identify words, capitalise
# 3. find word in dictionary, extract line
# 4. split line by "  ", extract spelling
# 5. Substitute spelling with symbols
# 6. save as list of lists
import collections
import re
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('utt')
    parser.add_argument('--thresh', type=float, default=1)
    args = parser.parse_args()
    return args

class PhonemicMagic:
    def __init__(self, map_path, cmu_path, stb_file, domain_word_path):
        #open these files
        self.cmu_to_pronunc_map = self.load_map(map_path)
        self.stb_table = self.load_stb(stb_file)
        self.cmu_dict = self.load_dict(cmu_path)
        self.domain_word_map = self.load_word_path(domain_word_path)
    
    def load_map(self, map_path):
        cmu_to_pronuc_map = {}
        f = open(map_path)
        lines = f.readlines()[1:]
        f.close()

        for line in lines:
            cmu, pronun = line.split("\t")
            cmu_to_pronuc_map[cmu] = pronun.rstrip()
        
        return cmu_to_pronuc_map

    def load_stb(self, stb_file):
        stb_table = {}

        f = open(stb_file, encoding="ISO-8859-1")
        lines = f.readlines()[1:]
        f.close()

        for line in lines:
            line = line.rstrip()
            class1, class2, shared, total, similarity = line.split("\t")
           
            key = str(sorted([class1, class2]))
            value = {"shared":shared, "total":total, "similarity":float(similarity)} 
        
            stb_table[key] = value
        return stb_table
    
    def load_dict(self, cmu_path):
        cmu_dict = {}

        if not os.path.isfile("cmudict-0.7b.txt"):
            bash_command = "curl http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b >> cmudict-0.7b.txt"
            os.system(bash_command)

        f = open("cmudict-0.7b.txt", encoding = "ISO-8859-1")
        lines = f.readlines()[56:-1]
        f.close()
        
        for line in lines:
            line = line.rstrip()
            key = line[0:line.find(" ")]
            value = line[line.find(" "):].lstrip()    
            cmu_dict[key] = value

        return cmu_dict

    def load_word_path(self, domain_word_path):
        domain_word_map = {}

        f = open(domain_word_path)
        lines = f.readlines()[1:] 
        f.close()

        for line in lines:
            line = line.rstrip()
            domain_word = line[0:line.find(" ")] 
            pronunciation = line[line.find(" "):].lstrip()
            domain_word_map[domain_word] = pronunciation
        return domain_word_map        

    # Function for listing words, ignoring punctuation, witespaces from an utterance:
    def tokenize(self, utt):
        input = re.findall('((\w+\'*\w*)|\.|\?|\!|,|:|;|\")', utt)
        output = [m[0] for m in input]
        return output

    # process utterance and retrieve pronunciation from CMU dictionary. Input must be a list:
    def cmudict_search(self, lst):
        if isinstance(lst, list):
            out = []
            missing_words = []
            for word in lst:
                output = None
                if word.upper() in self.cmu_dict:
                    output = [word, self.cmu_dict[word.upper()]] #cmu_lookup(word)
                if output is not None:
                    out.append(output)
                else:
                    out.append([word, None])
                    missing_words.append(word)

            if len(missing_words) > 0:
                pass
                #print("some words were not found in the pronunciation dictionary")
                # return missing_words
            return out, missing_words
        else:
            print("input not formatted")

#store CMU Dictionary as a lookup table, with only entryies, not comments
    def cmu_lookup(self, token):
        token = token.upper()
        return self.cmu_dict[token]

#optimise by saving in memory by dict
    def phoneme(self, token):
        if not token:
            return None
        
        converted = ""
        for cmu in token.split():
            if "AH0" in cmu:
                cmu = "AH0"
            elif "AH" in cmu:
                cmu = "AH"
            else:
                cmu = ''.join([c for c in cmu if not c.isdigit()])
            converted += self.cmu_to_pronunc_map[cmu]

        return converted 

    # Returning a cost
    def weighted_levenshtein(self, s1, s2):
        if len(s1) < len(s2):
            return self.weighted_levenshtein(s2, s1)

        # len(s1) >= len(s2)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[
                                 j + 1] + 1  # j+1 instead of j since previous_row and current_row are one character longer
                deletions = current_row[j] + 1  # than s2
                if c1 == c2: 
                    substitutions = previous_row[j]
                else:
                    substitutions = previous_row[j] + (1-self.stb_table[str(sorted([c1, c2]))]["similarity"])
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

       #print("previous_row   ", previous_row) 
        return previous_row[-1]

    def process_utterance(self, utterance,thresh=1):
        candidates = []
        # put this into a method, call method for each item
        asr_tokens = self.tokenize(utterance)
        out, missing = self.cmudict_search(asr_tokens)
        for original, phonemic in out:
            c = [original]  # find original utterance, score
            if phonemic is None:
                continue
            scored = []
                #need to get scoresTuple
    
            for domain_word in self.domain_word_map: #if CMU pronunciation exists, expensive loop
                pronunciation = self.domain_word_map[domain_word]
                phonemic_domain = self.phoneme(pronunciation) #this is expensive, store domain words in memory so this doesn't happen
                phonemic_original = self.phoneme(phonemic)
                score = self.weighted_levenshtein(phonemic_original, phonemic_domain)
                scoredTuple = EditScore(original, phonemic_original, domain_word, phonemic_domain, score)
                scored.append(scoredTuple)
            # is it a similarity or distance???? if similarity, then it's reverse=True
            # TODO: in place? returns?
            scored.sort(key=lambda x: x.score)
            if scored[0].score < thresh and scored[0].score > 0 :
                print(scored[0])
            # sort them, keep top n candidates, append to c
            candidates.append(c)
        return candidates
            # keep things > X??? keep top Y???
            # these are the cadidates
EditScore = collections.namedtuple('EditScore', 'asr_token asr_phonemes domain_token domain_phonemes score')
def main(args):
    
    #input_utterances = "Doesn't simplify the mission file. understand Find the first flower. dig rebel"
    # load utterances, loop through them here
    phonemic_helper = PhonemicMagic("cmu_feature_key.csv", "cmudict-0.7b.txt","stb_files/CELEXEnglish.fea.stb" ,"domain_words.csv" )
    utt = "Rubble revel bow"
    phonemic_helper.process_utterance(args.utt, args.thresh)

        
    # TODO: load from Adarsh dictionary file, get the utterance, tokenize
    # TODO: server/client interface
    # asr_tokens = [] # fill in

    # if you process at the utterance level...
    # make all possible combinations of candidates
    # dump
    pass
if __name__ == "__main__":
    args = parse_args()
    main(args)

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
import spacy
import spacy

nlp = spacy.load('en_core_web_sm')

# Returns frequencies for content words in text, but not for stop words.
class ParseUtt:
    def __init__(self, freq_path, utt, opt):  # add input requirements here
        # open these files
        self.nlp = spacy.load('en_core_web_sm')
        self.frequencies = self.find_freq(self.remove_stops(utt, option=opt))

    def find_freq(self, list, words):
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
        return dict(sorted(output.items(), key=lambda item: item[1])), missing

    def remove_stops(string_of_text, option="text"):
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
            if not token.is_space and not token.is_punct and not token.is_stop:
                bag.append(token)

        return bag


##################################################################

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('utt')
    parser.add_argument('--thresh', type=float, default=1)
    args = parser.parse_args()
    return args


class PhonemicMagic:
    def __init__(self, map_path, cmu_path, stb_file, domain_word_path):
        # open these files
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
            value = {"shared": shared, "total": total, "similarity": float(similarity)}

            stb_table[key] = value
        return stb_table

    def load_dict(self, cmu_path):
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

    # ToDo: Get the input as a bad of phones. Should we keep or toss spaces?
    # ToDo: Instead of returning None for missing words, find closest match in CMU dictionary
    def cmudict_search(self, lst):
        if isinstance(lst, list):
            out = []
            missing_words = []
            for word in lst:
                output = None
                if word.upper() in self.cmu_dict:
                    output = [word, self.cmu_dict[word.upper()]]  # cmu_lookup(word)
                if output is not None:
                    out.append(output)
                else:
                    out.append([word, None])
                    missing_words.append(word)

            if len(missing_words) > 0:
                pass
            return out, missing_words
        else:
            print("input not formatted")

    #ToDo: Maybe we can return this as a bag of phones?
    def cmu_lookup(self, token):
        token = token.upper()
        return self.cmu_dict[token]

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

    # for multi-word phrase, add scores
    # ToDo: we need to run this like a moving window: so for every word in the domain list, x no. of phones will be used for the search
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
                    substitutions = previous_row[j] + (1 - self.stb_table[str(sorted([c1, c2]))]["similarity"])
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        # print("previous_row   ", previous_row)
        return previous_row[-1]

    def process_utterance(self, utterance, thresh=1):

        candidates = []
        asr_tokens = ParseUtt.remove_stops(utterance)
        # asr_tokens = self.tokenize(utterance)
        out, missing = self.cmudict_search(asr_tokens)
        for original, phonemic in out:
            c = [original]  # find original utterance, score
            if phonemic is None:  # we may be losing words if their phonemic spelling isn't available
                continue
            scored = []

            # need to get scoresTuple

            for domain_word in self.domain_word_map:  # if CMU pronunciation exists, expensive loop
                pronunciation = self.domain_word_map[domain_word]
                phonemic_domain = self.phoneme(
                    pronunciation)  # this is expensive, store domain words in memory so this doesn't happen
                phonemic_original = self.phoneme(phonemic)
                score = self.weighted_levenshtein(phonemic_original, phonemic_domain)
                # simplify this, scoredTuple has too much redundancy (?)
                scoredTuple = EditScore(original, phonemic_original, domain_word, phonemic_domain, score)
                scored.append(scoredTuple)

            # TODO: in place? returns?
            scored.sort(key=lambda x: x.score)
            if scored[0].score < thresh and scored[0].score > 0:
                print("asr_token:", scored[0][0], "\tdomain_match:", scored[0][2], "\tsimilarity:", scored[0][4])
            # sort them, keep top n candidates, append to c
            candidates.append(c)  # we are not returning the output of the processing, rather, getting back original
        return candidates
        # ToDo: we need to find a way to have a meaningful utterance returned: Higher rank for low frequency words,
        #  for close phonemic matches. And then, find close matches for the phones around potential matches.

    def output_from_candidates(self, candidates):
        graph = [["start"]] + candidates + [["end"]]
        path = []
        paths = []

        def dfs(src_ind, dest_ind):
            if src_ind == dest_ind:
                paths.append(path[0:-1])
            else:
                for adj in graph[src_ind + 1]:
                    path.append(adj)
                    dfs(src_ind + 1, dest_ind)
                    path.pop()

        dfs(0, len(graph) - 1)
        return paths


# dfs("start", "end", 0, len(graph)-1, graph)
EditScore = collections.namedtuple('EditScore', 'asr_token asr_phonemes domain_token domain_phonemes score')


def main(args):
    # input_utterances = "Doesn't simplify the mission file. understand Find the first flower. dig rebel"
    # load utterances, loop through them here
    phonemic_helper = PhonemicMagic("cmu_feature_key.csv", "cmudict-0.7b.txt", "stb_files/CELEXEnglish.fea.stb",
                                    "domain_words.csv")
    word_cleanup = ParseUtt("gigaword_lean.txt")
    utt = "Rubble revel bow"
    # processed_utt = word_cleanup.
    phonemic_helper.process_utterance(args.utt, args.thresh)

    # TODO: instead of utt, run on output of class ParseUtt: a dict with the utterance as key, and a dict with word: frequency as item.
    # TODO: server/client interface

    # if you process at the utterance level...
    # make all possible combinations of candidates
    # dump
    pass
    # TO DO:
    # 1. we don't want to lose stop words: so, let Phonemic Magic decide if its a stop word or not [DONE]
    # 2. maybe incorporate elements in PhonemicMagic, not separate class
    # 3. Find an aggregate score based on gigaword
    # 4. Output: an ordered list of all repaired transcripts, in all combinations, sorted by frequency.


if __name__ == "__main__":
    args = parse_args()
    main(args)

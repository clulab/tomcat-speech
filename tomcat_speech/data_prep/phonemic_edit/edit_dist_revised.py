# To Do:
# 1. read tsv, split lines, find utterance
# 2. split utterance identify words, capitalise
# 3. find word in dictionary, extract line
# 4. split line by "  ", extract spelling
# 5. Substitute spelling with symbols
# 6. save as list of lists
import collections
import os
import argparse
import spacy
import itertools


##################################################################

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('utt')
    parser.add_argument('--thresh', type=float, default=2)
    parser.add_argument('--input_type', default="text")  # ensure only "text" or "file" are accepted
    args = parser.parse_args()
    return args


class PhonemicMagic:
    def __init__(self, map_path, cmu_path, stb_file, domain_word_path, gigaword_path, thresh, frequency_threshold=None):

        # initialize resources
        self.gigaword = Gigaword(gigaword_path)
        self.cmu_to_pronunc_map = self.load_map(map_path)
        self.stb_table = self.load_stb(stb_file)
        self.cmu_dict = self.load_cmu_dict(cmu_path)
        self.domain_words = self.load_domain_words(domain_word_path)

        # thresholds and ranges
        self.thresh = thresh
        self.frequency_threshold = frequency_threshold
        self.domain_word_min_phonemes = min([dw.num_phonemes() for dw in self.domain_words])
        self.domain_word_max_phonemes = max([dw.num_phonemes() for dw in self.domain_words])

    def set_frequency_threshold(self):
        # todo Megh or someone, dynamically calculate based on the domain word frequencies etc.
        self.frequency_threshold = None


    # -----------------------------------------------
    #        Methods for loading resources
    # -----------------------------------------------

    # Load a dictionary to convert from cmu representation to phonemes
    def load_map(self, map_path):
        cmu_to_pronuc_map = {}
        f = open(map_path)
        lines = f.readlines()[1:]
        f.close()

        for line in lines:
            cmu, pronun = line.split("\t")
            cmu_to_pronuc_map[cmu] = pronun.rstrip()

        return cmu_to_pronuc_map

    # Load the feature file for the phonemic distances
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

    # load cmudict, maps from orthographic to cmu representation
    def load_cmu_dict(self, cmu_path):
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

    # Load the domain words
    def load_domain_words(self, domain_word_path):
        domain_word_set = set()

        with open(domain_word_path) as f:
            # skip the first line, just labels
            f.readline()
            for line in f:
                if (len(line.strip().split("\t")) != 2):
                    print(line.strip().split("\t"))
                domain_word, cmu_pronunciation = line.strip().split("\t")
                phonemic = self.cmu_to_phonemes(cmu_pronunciation)
                freq = self.gigaword.find_freq(domain_word)
                domain_word_set.add(Token(original=domain_word, phonemic=phonemic, frequency=freq, is_domain=True))

        return domain_word_set

    # -----------------------------------------------------------------
    #        Methods for getting phonemic representations for words
    # -----------------------------------------------------------------

    # given an orthographic word, return the phonemic "spelling"
    def get_phonemic(self, original):
        cmu = self._cmu_lookup(original)
        converted = self.cmu_to_phonemes(cmu)
        return converted

    # gives the CMU entry
    def _cmu_lookup(self, token):
        token = token.upper()
        return self.cmu_dict[token]

    # takes cmu entry and gives the phonemic spelling
    def cmu_to_phonemes(self, token):
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

    # -----------------------------------------------------------------
    #        Phonemic Edit Distance / Scoring
    # -----------------------------------------------------------------

    # Returns a cost, higher is worse
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

    # -----------------------------------------------------------------
    #        Entry points for processing input
    # -----------------------------------------------------------------

    # TODO
    def process_utterance_by_window(self, utterance, window_size=3):

        candidates = []

        # sliding window of size 3

        # make windows

        # process each


    def process_window(self, t1, t2, t3):
        # remove special characters, use only sounds, no spaces (string of sounds)
        concatenated = t1.phonemic + t2.phonemic + t3.phonemic
        window_best_cost = 100
        window_best_domain_word = None

        # look for all possible spans of all sizes

        # [a,b,c,d,e,f,g]
        # happy -> hapi
        # sad -> sAd
        # umbrella => umbrEla

        for slice_size in range(max(self.domain_word_min_phonemes-1, 0), self.domain_word_max_phonemes+1):
            # make all slices
            for start in range(0, len(concatenated)):
                slice = concatenated[start: start+slice_size]

                for domain_word in self.domain_words:  # if CMU pronunciation exists, expensive loop
                    cost = self.weighted_levenshtein(''.join(slice), domain_word.phonemic)

                    if cost < window_best_cost:
                        window_best_cost = cost
                        window_best_domain_word = domain_word

        return window_best_domain_word, window_best_cost


    def process_token(self, token):
        if token.phonemic is None:
            pass
        # this frequency threshold should also remove stop words
        elif self.frequency_threshold is not None and token.frequency > self.frequency_threshold:
            pass
        else:
            for domain_word in self.domain_words:  # if CMU pronunciation exists, expensive loop
                cost = self.weighted_levenshtein(token.phonemic, domain_word.phonemic)
                # if a good enough match and not identical:
                if cost <= self.thresh and cost > 0:
                    print(f"considering '{token.original}' and '{domain_word.original}")
                    print(f"   cost: {cost}")
                    # store as one of the viable matches
                    token.add_match(domain_word, cost)



    def process_utterance(self, utterance):
        # List[SpacyToken]
        spacy_tokens = self.gigaword.tokenize_spacy(utterance)

        # List[Token] => [  Token(original=happy, phonemic='hApi', frequency=1003, is_domain=False, domain_matches=[(Token(origina....), 2.4), ...],
        #                   Token(original=sad, phonemic='sAd', frequency=403, is_domain=False, domain_matches=[(Token(origina....), 4.8), ...]
        #                   ...
        #                ]
        our_tokens = [self.convert_spacy_token(token) for token in spacy_tokens]

        for token in our_tokens:
            # the domain matches are added in place
            self.process_token(token)

        return our_tokens

    # -----------------------------------------------------------------
    #                           Helper Methods
    # -----------------------------------------------------------------

    # Convert a spacy token into one of our local Token objects
    def convert_spacy_token(self, spacy_token):
        original = spacy_token.text
        frequency = self.gigaword.find_freq(original)
        return Token(original, self.get_phonemic(original), frequency)



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




class Token():
    def __init__(self, original, phonemic, frequency, is_domain=False):
        self.original = original
        self.phonemic = phonemic
        self.frequency = frequency
        self.is_domain = is_domain
        self.domain_matches = [] # list of (domain_word: Token, cost: Double) tuples
        self.best_domain_match = None
        self.best_cost = None

    def add_match(self, matched_word, cost):
        self.domain_matches.append((matched_word, cost))

    def num_phonemes(self):
        return len(self.phonemic)

    # sort the found domain matches and return the k with the lowest cost
    def top_matches(self, k, include_self=False):
        # sort in place
        if include_self:
            self.add_match(self, 0)
        self.domain_matches.sort(key=lambda x: x[1], reverse=False)
        return self.domain_matches[:k]



class Gigaword:
    def __init__(self, freq_path):  # add input requirements here
        # open these files
        self.nlp = spacy.load('en_core_web_sm')
        self.frequencies = self.load_map(freq_path)

    def load_map(self, file):
        freqs = {}
        with open(file) as infile:
            for line in infile:
               word, freq  = line.strip().split('\t')
               freqs[word] = int(freq)
        return freqs

    def find_freq(self, word):
        if word in self.frequencies:
            return self.frequencies[word]
        else:
            return None

    def tokenize_spacy(self, text):
        # return spacy tokens
        return self.nlp(text)

    # def has_content(self, spacy_token):
    #     return not (spacy_token.is_space or spacy_token.is_punct or spacy_token.is_stop)



# tokens is List[Token]
def enumerate_utterance_options(tokens, topk=2):
    # [token1(match1, match2, ...), token2(), token3(match1, match2), ...]

    # candidates for each token in the utterance
    # List[List[String]]
    utterance_token_candidates = []
    for token in tokens:
        # List[String]
        curr_candidates = [token.original]
        for match, _ in token.top_matches(topk, include_self=False):
            # append the original string for the domain word matched
            # to the current candidates for this token
            curr_candidates.append(match.original)
        # Once done, add the cadidates for this token to the overall list for the utterance
        utterance_token_candidates.append(curr_candidates)

    # cartesian product of all the options
    # List[List[String]]
    for combination in itertools.product(*utterance_token_candidates):
        yield ' '.join(combination)



def main(args):
    # input_utterances = "Doesn't simplify the mission file. understand Find the first flower. dig rebel"

    # load utterances, loop through them here
    phonemic_helper = PhonemicMagic("cmu_feature_key.csv", "cmudict-0.7b.txt", "stb_files/CELEXEnglish.fea.stb",
                                    "domain_words.tsv", "gigaword_lean_head.txt", args.thresh)

    # todo: Megh
    phonemic_helper.set_frequency_threshold()

    # Todo: add arg to argparse to define Gigaword options
    with open("test_data.txt") as f:
        for utt in f:
    # utt = "Rubble revel bow"
    # processed_utt = word_cleanup.

            processed_tokens = phonemic_helper.process_utterance(utt)
            output = list(enumerate_utterance_options(processed_tokens))
            print(output)

    # TODO: server/client interface

    # if you process at the utterance level...
    # make all possible combinations of candidates
    # dump

    # TODO:
    # 3. Find an aggregate score based on gigaword [added class, need to integrate]
    # 4. Output: an ordered list of all repaired transcripts, in all combinations, sorted by frequency.


if __name__ == "__main__":
    args = parse_args()
    main(args)

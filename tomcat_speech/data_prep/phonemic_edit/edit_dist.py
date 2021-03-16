# To Do:
# 1. read tsv, split lines, find utterance
# 2. split utterance identify words, capitalise
# 3. find word in dictionary, extract line
# 4. split line by "  ", extract spelling
# 5. Substitute spelling with symbols
# 6. save as list of lists
import collections
class PhonemicMagic:
    def __init__(self, map_path, cmu_path, stb_file):
        self.cmu_to_pronunc_map = self.load_map(map_path)
        self.stb_table = self.load_stb(stb_file)
        self.cmu = self.load(cmu_path)
    def translate(self, token):
        raise NotImplementedError
    def weighted_levenshtein(self, s1, s2):
        raise NotImplementedError
EditScore = collections.namedtuple('EditScore', 'asr_token asr_phonemes domain_token domain_phonemes score')
def main():
    threshold = 0 # TODO
    phonemic_helper = PhonemicMagic(None, None, None)
    domain_words = set() # fill in, leave out ones not in CMU??
    # TODO: load from Adarsh dictionary file, get the utterance, tokenize
    # TODO: server/client interface
    asr_tokens = [] # fill in
    for asr_token in asr_tokens:
        scored = []
        phonemic_asr = phonemic_helper.translate(asr_token)
        for domain_word in domain_words:
            phonemic_domain = phonemic_helper.translate(domain_word)
            score = phonemic_helper.weighted_levenshtein(phonemic_asr, phonemic_domain)
            scoredTuple = EditScore(asr_token, phonemic_asr, domain_word, phonemic_domain, score)
            scored.append(scoredTuple)
        # is it a similarity or distance???? if similarity, then it's reverse=True
        # TODO: in place? returns?
        scored.sort(key= lambda x: x.score)
        # keep things > X??? keep top Y???
        #these are the cadidates
    # if you process at the utterance level...
    # make all possible combinations of candidates
    # dump
    pass
if __name__ == "__main__":
    main()
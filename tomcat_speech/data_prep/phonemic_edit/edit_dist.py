# To Do:
# 1. read tsv, split lines, find utterance
# 2. split utterance identify words, capitalise
# 3. find word in dictionary, extract line
# 4. split line by "  ", extract spelling
# 5. Substitute spelling with symbols
# 6. save as list of lists
import collections
class PhonemicMagic:
    def __init__(self, map_path, cmu_path, stb_file, domain_word_path):
        #open these files
        self.cmu_to_pronunc_map = self.load_map(map_path)
        self.stb_table = self.load_stb(stb_file)
        self.cmu_dict = self.load(cmu_path)
        self.domain_word_path = self.load(domain_word_path)

    # Function for listing words, ignoring punctuation, witespaces from an utterance:
    def tokenize(self, utt):
        input = re.findall('((\w+\'*\w*)|\.|\?|\!|,|:|;|\")', utt)
        [m[0] for m in input]
        return input

    # process utterance and retrieve pronunciation from CMU dictionary. Input must be a list:
    def cmudict_search(self, lst):
        if isinstance(lst, list):
            out = []
            missing_words = []
            for word in lst:
                output = cmu_lookup(word)
                if output is not None:
                    out.append(output)
                else:
                    out.append([word, None])
                    missing_words.append(word)

            if len(missing_words) > 0:
                print("some words were not found in the pronunciation dictionary")
                # return missing_words
            return out, missing_words
        else:
            print("input not formatted")

#store CMU Dictionary as a lookup table, with only entryies, not comments
    def cmu_lookup(self, token):
        token = token.upper()
        for line in cmu_dict:
            reg = "^" + token + " \s"
            if re.search(reg, line, re.I):
                result = line.rstrip('\n')
                trans = [token, result.split("  ")[1]]
                return trans
        return None

#optimise by saving in memory by dict
    def phoneme(self, token):
        for i in token:
            for line in phoneme_dict: #fetches the CMU-phoneme key. Regex matching is expensive
                words = line.rstrip('\n').split("	")
                reg = "^" + words[0] + ".*"
                if re.search(reg, i, re.I):
                    if not re.match(i, "^AH"):
                        re.sub(r'\d', '', i)
                        sub = words[1]
                    else re.match(i, "AH0"):
                        sub = words[1]
                    # make this return something

        raise NotImplementedError

    def weighted_levenshtein(self, s1, s2):
        if len(s1) < len(s2):
            return levenshtein(s2, s1)

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
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

            return previous_row[-1]

        raise NotImplementedError

    def process_utterance(self, utterance):
        candidates = []
        # put this into a method, call method for each item
        asr_tokens = self.tokenize(utterance)
        phonemic_asr = self.cmudict_search(asr_tokens)
        for original, phonemic in phonemic_asr:
            c = [original]  # find original utterance, score
            if phonemic is None:
                continue
            scored = []
                #need to get scoresTuple
            for domain_word in domain_words: #if CMU pronunciation exists, expensive loop
                phonemic_domain = self.translate(domain_word) #this is expensive, store domain words in memory so this doesn't happen
                score = self.weighted_levenshtein(phonemic, phonemic_domain)
                scoredTuple = EditScore(original, phonemic, domain_word, phonemic_domain, score)
                scored.append(scoredTuple)
            # is it a similarity or distance???? if similarity, then it's reverse=True
            # TODO: in place? returns?
            scored.sort(key=lambda x: x.score)
            # sort them, keep top n candidates, append to c
            candidates.append(c)
        return candidates
            # keep things > X??? keep top Y???
            # these are the cadidates
EditScore = collections.namedtuple('EditScore', 'asr_token asr_phonemes domain_token domain_phonemes score')
def main():
    input_utterances = [
    "Doesn't simplify about the mission file.",
    "understand",
    "Find the first flower",
    "What role does Bravo One start with?",
    "I'm going to go ahead and will start as a medic.",
    "Delta is packing.",
    "Hammer guy might be able to get three",
    "Nexus is Rebel",
    "I do not now.",
    "Yeah, how can we choose the clan map?"
    ]
    # load utterances, loop through them here
    phonemic_helper = PhonemicMagic(None, None, None, None)
    # TODO: load from Adarsh dictionary file, get the utterance, tokenize
    # TODO: server/client interface
    # asr_tokens = [] # fill in

    # if you process at the utterance level...
    # make all possible combinations of candidates
    # dump
    pass
if __name__ == "__main__":
    main()
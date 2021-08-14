import csv
from edit_dist_revised import PhonemicMagic, mk_eval_rows

def main():
    num_utt = 0
    num_no_cands = 0
    num_cands_with_cands = []

    thresh = 2
    phonemic_helper = PhonemicMagic("cmu_feature_key.csv", "cmudict-0.7b.txt", "stb_files/CELEXEnglish.fea.stb",
                                    "domain_words.tsv", "gigaword_lean_head.txt", thresh)

    phonemic_helper.set_frequency_threshold()

    filename = 'transcript.tsv'
    outfilename = filename[:-3] + "eval.csv"
    with open(filename) as csvin, open(outfilename, 'w') as csvout:
        reader = csv.reader(csvin, delimiter=',')
        writer = csv.writer(csvout, delimiter=',')
        writer.writerow(['original', 'candidate repaired', 'candidate score', 'annotation'])
        # skip the two headers
        next(reader)
        next(reader)
        for row in reader:
            if num_utt % 10 == 0:
                print(f'Processed {num_utt} utterances so far...')
            num_utt += 1
            utt = row[8].strip()
            utt_rows = mk_eval_rows(utt, phonemic_helper, 10)
            if len(utt_rows) < 1:
                # no candidates
                num_no_cands += 1
            else:
                num_cands_with_cands.append(len(utt_rows))
                writer.writerows(utt_rows)
                writer.writerow([])
    print(f'Processed {num_utt} utterances.')
    print(f'There were {num_no_cands} utterances with no candidates.')
    print(f'Of those with candidates, the mean num candidates was: {sum(num_cands_with_cands)/len(num_cands_with_cands)}')


if __name__=='__main__':
    main()
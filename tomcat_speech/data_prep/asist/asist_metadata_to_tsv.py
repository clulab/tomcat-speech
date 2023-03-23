# metadata_to_csv
import json
import re


def convert_metadata(metadata_file_path, save_path, n_speakers=1):
    all_lines = []
    with open(metadata_file_path, 'r') as the_file:
        for _ in range(n_speakers + 1):
            next(the_file)
        for line in the_file:
            jline = json.loads(line)
            all_lines.append(jline)

    idx2emo = {0:'anger', 1:'disgust', 2: 'fear', 3: 'joy', 4: 'neutral',
               5: 'sadness', 6: 'surprise'}
    idx2trait = {0:'agreeableness', 1: 'conscientiousness', 2:'extroversion',
                 3: 'neuroticism', 4: 'openness'}

    with open(save_path, 'w') as save_file:
        save_file.write("id\tparticipant\tstart_time\tend_time\ttext" +
                        "\temotion\ttrait\n")
        for line in all_lines:
            if line['data']['is_final'] == True:
                id = line['data']['id']
                text = line['data']['text']
                participant = line['data']['participant_id']
                # get actual start time
                start_time = line['data']['features']['word_messages'][0]['start_time']
                # start_time = line['data']['start_timestamp']
                try:
                    # get actual end time
                    end_time = line['data']['features']['word_messages'][-1]['end_time']
                    # end_time = line['data']['end_timestamp']
                except KeyError:
                    end_time = "None"
                anger = float(line['data']['sentiment']['emotions']['anger'])
                disgust = float(line['data']['sentiment']['emotions']['disgust'])
                fear =  float(line['data']['sentiment']['emotions']['fear'])
                joy =  float(line['data']['sentiment']['emotions']['joy'])
                neutral =  float(line['data']['sentiment']['emotions']['neutral'])
                sadness =  float(line['data']['sentiment']['emotions']['sadness'])
                surprise =  float(line['data']['sentiment']['emotions']['surprise'])

                emos = [anger, disgust, fear, joy, neutral, sadness, surprise]
                max_emo = emos.index(max(emos))

                agree = float(line['data']['sentiment']['traits']['agreeableness'])
                consc = float(line['data']['sentiment']['traits']['conscientiousness'])
                extro = float(line['data']['sentiment']['traits']['extroversion'])
                neur = float(line['data']['sentiment']['traits']['neuroticism'])
                openn = float(line['data']['sentiment']['traits']['openness'])

                traits = [agree, consc, extro, neur, openn]
                max_trait = traits.index(max(traits))

                save_file.write(f"{id}\t{participant}\t{start_time}\t{end_time}" +
                                f"\t{text}\t{idx2emo[max_emo]}\t{idx2trait[max_trait]}\n")


def save_data_to_file(json_data_lines, save_path, header, speaker_name):
    # get all data for one speaker in a file
    with open(save_path, 'w') as save_file:
        save_file.write(header)
        prev_time_start = 0.0
        c = 0
        for line in json_data_lines:
            if line['msg']['source'] == "tomcat_speech_analyzer" and line['topic'] == "agent/asr/final" and \
                    speaker_name in line['data']['participant_id']:
                id = line['data']['id']
                text = line['data']['text']
                participant = speaker_name

                start_time = line['data']['start_timestamp']
                start_time_time = re.search("T(.+)00Z", start_time).group(1)
                # 02:05:41.6262270
                start_time_h = float(start_time_time[:2]) * 60 * 60
                start_time_m = float(start_time_time[3:5]) * 60
                start_time_s = float(start_time_time[6:])
                time_start = start_time_h + start_time_m + start_time_s
                if c == 0:
                    prev_time_start = time_start
                    c += 1
                try:
                    end_time = line['data']['end_timestamp']
                    end_time_time = re.search("T(.+)00Z", end_time).group(1)
                    end_time_h = float(end_time_time[:2]) * 60 * 60
                    end_time_m = float(end_time_time[3:5]) * 60
                    end_time_s = float(end_time_time[6:])
                    time_end = end_time_h + end_time_m + end_time_s
                    time_diff = time_end - time_start + .2  # adding 200ms bc some are 0
                except KeyError:
                    end_time = "None"
                    time_diff = 0.0

                last_start_diff = time_start - prev_time_start

                save_file.write(f"{id}\t{participant}\t{start_time}\t{end_time}" +
                                f"\t{start_time_time}\t{end_time_time}\t{time_diff}\t{last_start_diff}\t{text}\n")

                prev_time_start = time_start


def convert_metadata_orig(metadata_file_path, save_path):
    all_lines = []

    with open(metadata_file_path, 'r') as the_file:
        for line in the_file:
            jline = json.loads(line)
            all_lines.append(jline)

    # get a separate file for each participant
    save_p = save_path.split("_gold.tsv")[0]
    namesdict = {'blue': "BLUE_ASIST",
                 'red': "RED_ASIST",
                 'green': "GREEN_ASIST"}

    header = "id\tparticipant\tstart_time\tend_time\tstart_hms\tend_hms\ttime_diff\ttime_since_last_utt\ttext\n"

    for name in namesdict.keys():
        save_data_to_file(all_lines, f"{save_p}_{name}_gold.tsv", header, namesdict[name])


if __name__ == "__main__":

    base_path = "../../study3_data_to_annotate"
    metadata_path = f"{base_path}/study-3_2022_HSRData_TrialMessages_Trial-T000608_Team-TM000204_Member-na_CondBtwn-none_CondWin-na_Vers-1.metadata"
    save_path = f"{base_path}/T000608_gold.tsv"
    convert_metadata_orig(metadata_path, save_path)

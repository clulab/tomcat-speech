# run bootstrap resampling on the data
from sklearn.utils import resample
import pickle


def bootstrap_resampling(output_list_1, output_list_2, num=10000):
    """
    Run bootstrap resampling as in:
    Assumes we are looking to see if output_list_1 is significantly higher than output_list_2
    DOES NOT check to see if they are significantly different in either direction
    """
    if len(output_list_1) != len(output_list_2):
        exit("Error: Output lists are not the same length")

    data_size = len(output_list_1)
    indices = list(range(data_size))
    num_above_threshold = 0.0

    for n in range(num):
        avg_delta = 0.0

        shuffled_idxs = resample(indices, n_samples=data_size)

        for idx in shuffled_idxs:
            avg_delta += (output_list_1[idx] - output_list_2[idx])

        avg_delta = avg_delta / float(data_size)

        if avg_delta > 0:
            num_above_threshold += 1

    # get p-value
    p = 1 - (num_above_threshold / float(num))

    return p

# def run_bootstrap_resampling(gold_pred_list, n=1000, p=.05):
#     """
#     Run bootstrap resampling on results of network
#     gold_pred_list : a list with (gold, prediction) doubles
#     n : the number of iterations to run through
#     p : the p-value (default p = .05)
#     """
#     all_scores = []
#     num_samples = len(gold_pred_list)
#
#     # for each time
#     for i in range(n):
#         # resample
#         sample_set = resample(gold_pred_list, n_samples=num_samples)
#
#         # calculate avg f1
#         golds = [item[0] for item in sample_set]
#         preds = [item[1] for item in sample_set]
#         task_avg_f1 = precision_recall_fscore_support(golds, preds, average="weighted")
#         # add avg f1 to all_scores
#         all_scores.append(task_avg_f1[2])
#
#     # print all scores sorted
#     print(sorted(all_scores))
#
#     # calculate statistics on all_scores
#     mean_score = statistics.mean(all_scores)
#     stdev_score = statistics.stdev(all_scores, xbar=mean_score)
#
#     # calculate percentile scores
#     # todo: could use scipy.stats scoreatpercentile -- scipy not required for the project yet
#     min_percentile = (p / 2.0) * 100
#     print(min_percentile)
#     max_percentile = (1 - (p / 2.0)) * 100
#     print(max_percentile)
#     min_score, max_score = stats.scoreatpercentile(all_scores, per=[min_percentile, max_percentile])
#
#     return min_score, max_score, mean_score, stdev_score

def order_results(results_tuple_1, results_tuple_2):
    """
    Prepare and order the results for bootstrap resampling
    """
    results_tuple_1 = sorted(results_tuple_1, key=lambda x: x[2])
    results_tuple_2 = sorted(results_tuple_2, key=lambda x: x[2])

    data_1 = []
    data_2 = []

    for i, item in enumerate(results_tuple_1):
        if item[2] != results_tuple_2[i][2]:
            exit("These test sets do not contain the same elements")
        else:
            data_1.append(0 if item[1] != item[0] else 1)
            data_2.append(0 if results_tuple_2[i][1] != results_tuple_2[i][0] else 1)

    return data_1, data_2


if __name__ == "__main__":
    # for dataset MUStARD
    test_1_path = "output/multitask/EACL_TEST_2/TEST1_MUSTARD_AcousticOnly_withoutgender_genericavging_25to75perc_minmaxavgstdevcalcs_IS10_2021-01-25"
    test_2_path = "output/multitask/EACL_TEST_2/TEST1_MUSTARD_GOLD_TEXTONLY_withoutgender_2021-01-25"
    # load data 1
    with open(f"{test_1_path}/test_preds.pickle", 'rb') as test1file:
        test_1_results = pickle.load(test1file)
    # load data 2
    with open(f"{test_2_path}/test_preds.pickle", 'rb') as test2file:
        test_2_results = pickle.load(test2file)

    for task in test_1_results.keys():
        ordered_preds_1, ordered_preds_2 = order_results(test_1_results[task],
                                                         test_2_results[task])

        p_value = bootstrap_resampling(ordered_preds_1, ordered_preds_2)
        print(p_value)


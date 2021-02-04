# run bootstrap resampling on the data
import sys

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
            avg_delta += output_list_1[idx] - output_list_2[idx]

        avg_delta = avg_delta / float(data_size)

        if avg_delta > 0:
            num_above_threshold += 1

    # get p-value
    p = 1 - (num_above_threshold / float(num))

    return p


def order_results(results_tuple_1, results_tuple_2):
    """
    Prepare and order the results for bootstrap resampling
    """
    results_tuple_1 = sorted(results_tuple_1, key=lambda x: x[2])
    results_tuple_2 = sorted(results_tuple_2, key=lambda x: x[2])

    data_1 = []
    data_2 = []

    print(results_tuple_1[:50])
    print(results_tuple_2[:50])
    print(len(results_tuple_1))
    print(len(results_tuple_2))

    for i, item in enumerate(results_tuple_1):
        if item[2] != results_tuple_2[i][2]:
            exit("These test sets do not contain the same elements")
        else:
            data_1.append(0 if item[1] != item[0] else 1)
            data_2.append(0 if results_tuple_2[i][1] != results_tuple_2[i][0] else 1)

    return data_1, data_2


if __name__ == "__main__":
    print(len(sys.argv))
    test_1_path = sys.argv[1]
    test_2_path = sys.argv[2]

    # load data 1
    with open(f"{test_1_path}/test_preds.pickle", "rb") as test1file:
        test_1_results = pickle.load(test1file)
    # load data 2
    with open(f"{test_2_path}/test_preds.pickle", "rb") as test2file:
        test_2_results = pickle.load(test2file)

    # compute bootstrap resampling
    for task in test_1_results.keys():
        ordered_preds_1, ordered_preds_2 = order_results(
            test_1_results[task], test_2_results[task]
        )

        p_value = bootstrap_resampling(ordered_preds_1, ordered_preds_2)
        print(p_value)

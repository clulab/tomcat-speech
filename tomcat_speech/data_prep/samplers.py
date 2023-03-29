import torch
import imblearn
import math
from torch.utils.data import RandomSampler


class BatchSchedulerSampler(torch.utils.data.sampler.Sampler):
    """
    iterate over tasks and provide a random batch per task in each mini-batch
    Slightly altered from: https://gist.github.com/bomri/d93da3e6f840bb93406f40a6590b9c48
    """

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets)
        self.largest_dataset_size = max(
            [len(cur_dataset.samples) for cur_dataset in dataset.datasets]
        )

    def __len__(self):
        return (
            self.batch_size
            * math.ceil(self.largest_dataset_size / self.batch_size)
            * len(self.dataset.datasets)
        )

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = RandomSampler(cur_dataset)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        step = self.batch_size * self.number_of_datasets
        samples_to_grab = self.batch_size
        # for this case we want to get all samples in dataset, this force us to resample from the smaller datasets
        epoch_samples = self.largest_dataset_size * self.number_of_datasets

        final_samples_list = []  # this is a list of indexes from the combined dataset
        for _ in range(0, epoch_samples, step):
            for i in range(self.number_of_datasets):
                cur_batch_sampler = sampler_iterators[i]
                cur_samples = []
                for _ in range(samples_to_grab):
                    try:
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                    except StopIteration:
                        # stop trying to add samples and continue on in the next dataset
                        break
                final_samples_list.extend(cur_samples)

        return iter(final_samples_list)


class RandomOversampler:
    """
    Use a random oversampler to oversample minority classes
    """

    def __init__(self, seed):
        self.sampler = imblearn.over_sampling.RandomOverSampler(
            random_state=seed, shrinkage=2.5
        )

    def _oversample(self, utt_ids, y_values):
        # use the random oversampler
        # utt_ids are called as the Xs
        # ys[0] are y_values
        oversampled_ids, _ = self.sampler.fit_resample(utt_ids, y_values)

        return oversampled_ids

    def _apply_resampling(self, unsampled_data, oversampled_ids):
        # complete resampling
        oversampled_data = []

        # get dict of audio_id : data point idx
        audio_id2idx = {}
        for i, dpoint in enumerate(unsampled_data):
            audio_id2idx[dpoint["audio_id"]] = i

        # add data point with each idx found
        for id in oversampled_ids:
            oversampled_data.append(unsampled_data[audio_id2idx[id[0]]])

        # return
        return oversampled_data

    def prep_data_through_oversampling(self, dataset):
        # oversample ids and then complete the oversampling of data points
        print("starting oversampling")

        all_ids = [[item["audio_id"]] for item in dataset]
        # todo: expand to allow flexibility of which task to select
        #   when there are multiple
        all_ys = [item["ys"][1] for item in dataset]  # going with emotion items

        # do oversampling
        sampled_ids = self._oversample(all_ids, all_ys)

        # oversample the items in the dataset
        sampled_dataset = self._apply_resampling(dataset, sampled_ids)

        print("oversampling complete")

        return sampled_dataset

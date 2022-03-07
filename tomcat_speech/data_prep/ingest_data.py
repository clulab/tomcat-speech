# used for ingesting data which is split into numerous parts
# all data contains audio_id for matching up
import pickle
import torch
import pandas as pd


class DataIngester:
    def __init__(self, data_path, dataset, acoustic_feats="IS13", text_feats="distilbert", use_spec=False):
        """
        Get the data and combine as needed
        Assumes the following directory structure
        Data path: contains subdirs for each data type
        {data_path}/acoustic_data/{acoustic_feats} : path to acoustic feats pickle files
        {data_path}/text_data/{text_feats} : path to text feats pickle files
        {data_path}/spectrogram_data : path to spectrogram feats pickle files
        {data_path}/ys_data : path to ys pickle files
        """
        # get dataset
        self.dataset = dataset

        # save acoustic and text feats
        self.a_feats = acoustic_feats
        self.t_feats = text_feats

        # get paths to data
        self.ys_path = f"{data_path}/ys_data"
        self.spec_path = f"{data_path}/spectrogram_data" if use_spec else None
        self.acoustic_path = f"{data_path}/acoustic_data/{acoustic_feats}" if acoustic_feats is not None else None
        self.text_path = f"{data_path}/text_data/{text_feats}" if text_feats is not None else None

    def get_train_data(self):
        return self._get_data("train")

    def get_dev_data(self):
        return self._get_data("dev")

    def get_test_data(self):
        return self._get_data("test")

    def _get_data(self, partition="train"):
        # load in files
        ys = pickle.load(open(f"{self.ys_path}/{self.dataset}_ys_{partition}.pickle", 'rb'))

        # set other data to None as holder
        spec = None
        acoustic = None
        text = None

        # overwrite if data should be gotten
        if self.spec_path is not None:
            spec = pickle.load(open(f"{self.spec_path}/{self.dataset}_spec_{partition}.pickle", 'rb'))
        if self.acoustic_path is not None:
            acoustic = pickle.load(open(f"{self.acoustic_path}/{self.dataset}_{self.a_feats}_{partition}.pickle", 'rb'))
        if self.text_path is not None:
            text = pickle.load(open(f"{self.text_path}/{self.dataset}_{self.t_feats}_{partition}.pickle", 'rb'))

        data = self._combine_data([ys, spec, acoustic, text])

        return data

    def _combine_data(self, list_of_data):
        """
        Combine data based on what exists
        :param list_of_data : the data (ys, spec, acoustic, text)
            some values may be None
        """
        small_list = [item for item in list_of_data if item is not None]

        combined = pd.DataFrame(small_list[0])
        combined['audio_id'] = combined['audio_id'].astype(str)

        for data in small_list[1:]:
            data = pd.DataFrame(data)
            data['audio_id'] = data['audio_id'].astype(str)
            combined = combined.merge(data, on="audio_id", how="left")

        return combined.to_dict(orient='records')


if __name__ == "__main__":
    base_path = "../../datasets/pickled_data/field_separated_data"

    ingester = DataIngester(base_path, 'mosi', None, "glove", True)

    train = ingester.get_train_data()
    dev = ingester.get_dev_data()
    print(dev[0].keys())
    test = ingester.get_test_data()

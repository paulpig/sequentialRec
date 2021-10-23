from .base import AbstractDataset

import pandas as pd
import pdb


class BeautyDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'toys'

    @classmethod
    def is_zipfile(cls):
        return False

    @classmethod
    def url(cls):
        return 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Beauty.csv' #数据集的量不同导致的;
        # return 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz'

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('ratings.csv')
        df = pd.read_csv(file_path, header=None)
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df

    def load_ratings_df_from_json(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('ratings_5_core.csv')
        # pdb.set_trace()
        # df = pd.read_csv(file_path, header=None, usecols=['reviewerID', 'asin', 'overall', 'unixReviewTime'])
        df = pd.read_csv(file_path, usecols=['reviewerID', 'asin', 'overall', 'unixReviewTime'])
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df

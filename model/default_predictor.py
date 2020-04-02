import pandas as pd
import numpy as np
import json

from util import get_pre_timestamp, ncr

class DefaultPredictor(object):
    """Random Predictor."""
    def __init__(self, slang_propagation_data_path, community_prior, community_to_slang_across_time, slang_distance,
                 community_index, overlap_ratio, train_set_percentage=0.8, start_time="2010-01", seed=0):

        self.train_set_percentage = train_set_percentage
        # All communities should exist from start_time to 2018-10.
        self.start_time = start_time
        self.seed = seed
        self._prepare_data(slang_propagation_data_path, community_index)
        self._prepare_model(community_prior, community_to_slang_across_time, slang_distance, community_index,
                            overlap_ratio)


    def _prepare_data(self, slang_propagation_data_path, community_index):
        with open(slang_propagation_data_path) as f:
            slang_propagation_data = json.load(f)

        df_selected_community_list = pd.read_csv(community_index)
        self.community_to_index = df_selected_community_list.set_index("subreddit")
        self.index_to_community = df_selected_community_list.set_index("subreddit_index")

        self.num_community = len(df_selected_community_list)
        self.num_slang = len(slang_propagation_data)
        # Randomly split slang to train and test set.
        slang_index = [i for i in range(self.num_slang)]
        np.random.seed(self.seed)
        np.random.shuffle(slang_index)
        train_slang_index_set = set(slang_index[:(int)(self.num_slang * self.train_set_percentage)])

        # 1 for not having slang, 0 for already having slang.
        train_community_mask = []
        test_community_mask = []
        # Multi-label one-hot encoding.
        train_labels = []
        test_labels = []

        train_pre_timestamps = []
        test_pre_timestamps = []
        train_pre_propagated_communities_index = []
        test_pre_propagated_communities_index = []
        train_slang_to_propagate = []
        test_slang_to_propagate = []

        for i, slang in enumerate(slang_propagation_data):
            accumulated_community_mask = np.ones(self.num_community)
            pre_propagated_communities_index = None
            slang_propagation_history = slang_propagation_data[slang]
            is_emerged = False

            is_training_data = i in train_slang_index_set
            community_mask = train_community_mask if is_training_data else test_community_mask
            labels = train_labels if is_training_data else test_labels
            pre_timestamps = train_pre_timestamps if is_training_data else test_pre_timestamps
            pre_propagated_communities_index_list = train_pre_propagated_communities_index if is_training_data else test_pre_propagated_communities_index
            slang_to_propagate = train_slang_to_propagate if is_training_data else test_slang_to_propagate

            # time_stamps has to be in increasing order.
            for time_stamp in slang_propagation_history:
                propagated_communities = slang_propagation_history[time_stamp]
                propagated_communities_index = self.community_to_index.loc[propagated_communities]["subreddit_index"].values

                if is_emerged:
                    if time_stamp >= self.start_time:
                        cur_community_mask = np.copy(accumulated_community_mask)
                        community_mask.append(cur_community_mask)
                        cur_label = np.zeros(self.num_community)
                        cur_label[propagated_communities_index] = 1.
                        labels.append(cur_label)
                        pre_timestamp = get_pre_timestamp(time_stamp)
                        pre_timestamps.append(pre_timestamp)
                        pre_propagated_communities_index_list.append(pre_propagated_communities_index)
                        slang_to_propagate.append(slang)
                else:
                    is_emerged = True

                accumulated_community_mask[propagated_communities_index] = 0.
                pre_propagated_communities_index = propagated_communities_index

        self.train_community_mask = np.array(train_community_mask)
        self.train_labels = np.array(train_labels)
        self.train_pre_timestamps = np.array(train_pre_timestamps)
        self.train_pre_propagated_communities_index = train_pre_propagated_communities_index
        self.train_slang_to_propagate = np.array(train_slang_to_propagate)
        self.test_community_mask = np.array(test_community_mask)
        self.test_labels = np.array(test_labels)
        self.test_pre_timestamps = np.array(test_pre_timestamps)
        self.test_pre_propagated_communities_index = test_pre_propagated_communities_index
        self.test_slang_to_propagate = np.array(test_slang_to_propagate)


    def _prepare_model(self, community_prior, community_to_slang_across_time, slang_distance, community_index,
                       overlap_ratio):
        self.model_name = self.__class__.__name__


    def _get_evaluation_stats_on_one_correct_label(self, community_mask, labels):
        num_label = labels.sum(axis=1)
        num_existing_community = (community_mask > 0).sum(axis=1)
        expected_rank = np.mean(1 + (num_existing_community - num_label) / (num_label + 1))

        num_data = community_mask.shape[0]
        num_community = community_mask.shape[1]
        ranks = np.ones((num_data, num_community + 1))

        for i in range(num_data):
            cur_num_existing_community = (int)(num_existing_community[i])
            cur_num_non_label_community = (int)(num_existing_community[i] - num_label[i])
            cur_ranks = np.arange(cur_num_non_label_community + 1, dtype=int)

            func1 = lambda r: ncr(cur_num_non_label_community, r)
            vfunc1 = np.vectorize(func1)
            func2 = lambda r: ncr(cur_num_existing_community, r)
            vfunc2 = np.vectorize(func2)

            ranks[i, :cur_num_non_label_community + 1] = 1. - (vfunc1(cur_ranks) / vfunc2(cur_ranks))

        roc = ranks.mean(axis=0)
        return expected_rank, roc


    def _get_evaluation_stats_on_all_correct_label(self, community_mask, labels):
        num_label = labels.sum(axis=1)
        num_existing_community = (community_mask > 0).sum(axis=1)
        expected_rank = np.mean(num_existing_community - (num_existing_community - num_label) / (num_label + 1))

        num_data = community_mask.shape[0]
        num_community = community_mask.shape[1]
        ranks = np.zeros((num_data, num_community + 1))

        for i in range(num_data):
            cur_num_existing_community = (int)(num_existing_community[i])
            cur_num_label = (int)(num_label[i])
            cur_num_non_label_community = (int)(num_existing_community[i] - num_label[i])
            cur_ranks = np.arange(cur_num_non_label_community + 1, dtype=int) + cur_num_label

            func1 = lambda r: ncr(cur_num_non_label_community, r)
            vfunc1 = np.vectorize(func1)
            func2 = lambda r: ncr(cur_num_existing_community, r)
            vfunc2 = np.vectorize(func2)

            ranks[i, cur_num_label:cur_num_existing_community+1] = vfunc1(cur_ranks - cur_num_label) / vfunc2(cur_ranks)

        roc = ranks.mean(axis=0)
        return expected_rank, roc


    def train(self):
        expected_rank_one, roc_one = self._get_evaluation_stats_on_one_correct_label(self.train_community_mask, self.train_labels)
        expected_rank_all, roc_all = self._get_evaluation_stats_on_all_correct_label(self.train_community_mask, self.train_labels)
        return expected_rank_one, roc_one, expected_rank_all, roc_all


    def predict(self):
        expected_rank_one, roc_one = self._get_evaluation_stats_on_one_correct_label(self.test_community_mask, self.test_labels)
        expected_rank_all, roc_all = self._get_evaluation_stats_on_all_correct_label(self.test_community_mask, self.test_labels)
        return expected_rank_one, roc_one, expected_rank_all, roc_all



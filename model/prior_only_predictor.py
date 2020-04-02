import pandas as pd
import numpy as np
from scipy.stats.mstats import rankdata

from util import ncr
from model.default_predictor import DefaultPredictor

class PriorOnlyPredictor(DefaultPredictor):
    """Prior-Only Predictor."""
    def _prepare_model(self, community_prior, community_to_slang_across_time, slang_distance, community_index,
                       overlap_ratio):

        self._prepare_prior(community_prior)
        self.train_conditional_prior = self._prepare_conditional_prior(overlap_ratio, self.train_community_mask,
                                                                       self.train_pre_timestamps,
                                                                       self.train_pre_propagated_communities_index)
        self.test_conditional_prior = self._prepare_conditional_prior(overlap_ratio, self.test_community_mask,
                                                                      self.test_pre_timestamps,
                                                                      self.test_pre_propagated_communities_index)
        self.model_name = self.__class__.__name__


    def _prepare_prior(self, community_prior, eps=1e-30):
        # prior is uniform distribution if both community_diversity_prior and community_size_prior is None
        if community_prior is None:
            self.train_prior = 1.0
            self.test_prior = 1.0
            return

        df_community_prior = pd.read_csv(community_prior)
        df_community_prior.set_index("subreddit", inplace=True)
        df_selected_community_prior = df_community_prior.loc[self.community_to_index.index.values]

        unmasked_train_prior = df_selected_community_prior[self.train_pre_timestamps].transpose().values
        masked_train_prior = unmasked_train_prior * self.train_community_mask
        self.train_prior = masked_train_prior / (masked_train_prior.sum(axis=1)[:, np.newaxis] + eps)
        unmasked_test_prior = df_selected_community_prior[self.test_pre_timestamps].transpose().values
        masked_test_prior = unmasked_test_prior * self.test_community_mask
        self.test_prior = masked_test_prior / (masked_test_prior.sum(axis=1)[:, np.newaxis] + eps)


    def _prepare_conditional_prior(self, overlap_ratio, community_mask, pre_timestamps,
                                   pre_propagated_communities_index, eps=1e-30):
        if overlap_ratio is None:
            return 1.0

        conditional_prior = np.empty((len(pre_timestamps), self.num_community))
        for i, time_stamp in enumerate(pre_timestamps):
            cur_overlap_ratio = overlap_ratio[time_stamp]
            overlap_with_pre_targets = cur_overlap_ratio[:, pre_propagated_communities_index[i]].sum(axis=1)  # Shape: (num_community,)
            valid_overlap_with_pre_targets = overlap_with_pre_targets * community_mask[i]  # Shape: (num_community,) with value 0 for invalid community

            num_valid_community = community_mask[i].sum().astype(int)
            num_pre_targets = len(pre_propagated_communities_index[i])
            total_num_comp = ncr(num_valid_community - 1, num_pre_targets - 1)
            valid_community_overlap_ratio = cur_overlap_ratio * community_mask[i]
            total_overlap_ratio = valid_community_overlap_ratio.sum(axis=1) + overlap_with_pre_targets  # Shape: (num_community,)
            valid_total_overlap_ratio = total_overlap_ratio * community_mask[i]  # Shape: (num_community,) with value 0 for invalid community
            scaled_valid_total_overlap_ratio = total_num_comp * valid_total_overlap_ratio

            conditional_prior[i] = valid_overlap_with_pre_targets / (scaled_valid_total_overlap_ratio + eps)

        return conditional_prior


    def _get_rank_for_one_correct_label(self, prob, y):
        ranks = rankdata(-prob, axis=1)
        inverse_ranks = self.num_community - ranks
        target_inverse_ranks = inverse_ranks * y
        best_target_inverse_ranks = target_inverse_ranks.max(axis=1)
        best_target_ranks = self.num_community - best_target_inverse_ranks
        rounded_best_target_ranks = np.rint(best_target_ranks).astype(int)
        return rounded_best_target_ranks


    def _get_rank_for_all_correct_label(self, prob, y):
        ranks = rankdata(-prob, axis=1)
        target_ranks = ranks * y
        best_target_ranks = target_ranks.max(axis=1)
        rounded_best_target_ranks = np.rint(best_target_ranks).astype(int)
        return rounded_best_target_ranks


    def _get_evaluation_stats(self, prob, labels, correct_on_all_label=True):
        ranks = self._get_rank_for_all_correct_label(prob, labels) if correct_on_all_label else self._get_rank_for_one_correct_label(prob, labels)
        num_data = ranks.shape[0]

        expected_rank = np.mean(ranks)

        accuracy_per_retrieval = np.zeros(self.num_community + 1)
        for rank in ranks:
            accuracy_per_retrieval[rank] += 1
        for i in range(1, self.num_community + 1):
            accuracy_per_retrieval[i] = accuracy_per_retrieval[i] + accuracy_per_retrieval[i - 1]
        roc = accuracy_per_retrieval / num_data

        return expected_rank, roc


    def train(self):
        prob = self.train_conditional_prior * self.train_prior
        expected_rank_one, roc_one = self._get_evaluation_stats(prob, self.train_labels, correct_on_all_label=False)
        expected_rank_all, roc_all = self._get_evaluation_stats(prob, self.train_labels, correct_on_all_label=True)
        return expected_rank_one, roc_one, expected_rank_all, roc_all


    def predict(self):
        prob = self.test_conditional_prior * self.test_prior
        expected_rank_one, roc_one = self._get_evaluation_stats(prob, self.test_labels, correct_on_all_label=False)
        expected_rank_all, roc_all = self._get_evaluation_stats(prob, self.test_labels, correct_on_all_label=True)
        return expected_rank_one, roc_one, expected_rank_all, roc_all



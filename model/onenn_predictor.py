import os
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from model.prior_only_predictor import PriorOnlyPredictor

class OneNNPredictor(PriorOnlyPredictor):
    """1-NN based predictor."""
    def _prepare_model(self, community_prior, community_to_slang_across_time, slang_distance, community_index,
                       overlap_ratio):
        unique_timestamps = np.unique(np.concatenate((self.train_pre_timestamps, self.test_pre_timestamps), axis=0))
        subreddit_idx_to_slang_across_time = {}
        for subreddit_index, subreddit in enumerate(self.index_to_community.values):
            path = os.path.join(community_to_slang_across_time, subreddit[0] + ".json")
            with open(path) as f:
                data = json.load(f)

            subreddit_idx_to_slang_across_time[subreddit_index] = {}
            for timestamp in unique_timestamps:
                subreddit_idx_to_slang_across_time[subreddit_index][timestamp] = data[timestamp]

        self.train_slang_distances = self._get_slang_distances(self.train_slang_to_propagate, self.train_pre_timestamps,
                                                               slang_distance, subreddit_idx_to_slang_across_time,
                                                               self.train_community_mask)
        self.test_slang_distances = self._get_slang_distances(self.test_slang_to_propagate, self.test_pre_timestamps,
                                                              slang_distance, subreddit_idx_to_slang_across_time,
                                                              self.test_community_mask)
        self._prepare_prior(community_prior)
        self.train_conditional_prior = self._prepare_conditional_prior(overlap_ratio, self.train_community_mask,
                                                                       self.train_pre_timestamps,
                                                                       self.train_pre_propagated_communities_index)
        self.test_conditional_prior = self._prepare_conditional_prior(overlap_ratio, self.test_community_mask,
                                                                      self.test_pre_timestamps,
                                                                      self.test_pre_propagated_communities_index)
        self.model_name = self.__class__.__name__


    def _get_slang_distances(self, slangs, timestamps, slang_distance, subreddit_idx_to_slang_across_time, community_masks):
        X_slang_distances = np.ones((len(timestamps), len(self.index_to_community))) * np.inf
        df_distance = pd.read_csv(slang_distance)
        df_distance.set_index("slang", inplace=True)
        for i, (target_slang, timestamp, community_mask) in enumerate(zip(slangs, timestamps, community_masks)):
            valid_community_idx = np.argwhere(community_mask > 0.).reshape(-1)
            for j in valid_community_idx:
                host_slangs = subreddit_idx_to_slang_across_time[j][timestamp]
                if len(host_slangs) == 0:
                    continue

                distances = df_distance.loc[target_slang][host_slangs].values
                onenn = np.min(distances)
                X_slang_distances[i][j] = onenn

        return X_slang_distances


    def _likelihood_smooth(self, likelihood, eps=1e-30):
        smoothed_likelihood = likelihood + eps
        normalized_likelihood = smoothed_likelihood / smoothed_likelihood.sum(axis=1)[:, np.newaxis]
        return normalized_likelihood


    def _sum_negative_log_posterior(self, weight, eps=1e-30):
        likelihood = np.exp(-np.square(self.train_slang_distances) / weight)
        smoothed_likelihood = self._likelihood_smooth(likelihood)
        posterior = smoothed_likelihood * self.train_conditional_prior * self.train_prior

        negative_log_posterior = np.average(-(self.train_labels * np.log(posterior + eps)).sum(axis=1))
        return negative_log_posterior


    def _get_prob(self, slang_distances, conditional_prior, prior):
        likelihood = np.exp(-np.square(slang_distances) / self.weight)
        smoothed_likelihood = self._likelihood_smooth(likelihood)
        posterior = smoothed_likelihood * conditional_prior * prior
        return posterior


    def train(self):
        boundary = (1e-2, 1e2)
        self.weight = np.array(1.0)
        result = minimize(self._sum_negative_log_posterior, self.weight, bounds=(boundary,))
        self.weight = result.x

        prob = self._get_prob(self.train_slang_distances, self.train_conditional_prior, self.train_prior)
        expected_rank_one, roc_one = self._get_evaluation_stats(prob, self.train_labels, correct_on_all_label=False)
        expected_rank_all, roc_all = self._get_evaluation_stats(prob, self.train_labels, correct_on_all_label=True)
        return expected_rank_one, roc_one, expected_rank_all, roc_all


    def predict(self):
        prob = self._get_prob(self.test_slang_distances, self.test_conditional_prior, self.test_prior)
        expected_rank_one, roc_one = self._get_evaluation_stats(prob, self.test_labels, correct_on_all_label=False)
        expected_rank_all, roc_all = self._get_evaluation_stats(prob, self.test_labels, correct_on_all_label=True)
        return expected_rank_one, roc_one, expected_rank_all, roc_all



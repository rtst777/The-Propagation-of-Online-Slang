import pandas as pd
import numpy as np

from model.onenn_predictor import OneNNPredictor

class ExemplarPredictor(OneNNPredictor):
    """Exemplar based predictor."""
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
                exemplar = np.mean(distances)
                X_slang_distances[i][j] = exemplar

        return X_slang_distances


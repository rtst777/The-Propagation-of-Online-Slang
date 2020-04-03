import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.manifold import TSNE

SLANG_PROPAGATION_DATA_PATH = "data/slang_propagation_history_1000subreddits.json"
COMMUNITY_SIZE_PERCENTAGE_PRIOR = "data/user_num_per_community_across_time.csv"
USER_OVERLAP_RATIO = "data/user_overlap_ratio"
COMMUNITY_INDEX = "data/df_selected_community_list.csv"

TIME = "2018-10"
COMMUNITY_SIZE_SCALE = 3000
TEXT_SIZE = 15
ARROW_SIZE = 0.3
TITLE_SIZE = 50


def plot(community_names, community_sizes, propagated_community_index, slang, xs, ys, is_user_based_embedding):
    scaled_community_size = (community_sizes / community_sizes.sum()) * COMMUNITY_SIZE_SCALE
    plt.figure(figsize=(32, 24))

    # Draws non-propagated communities.
    mask = np.ones(len(community_names), dtype=bool)
    mask[propagated_community_index] = False
    plt.scatter(x=xs[mask], y=ys[mask], marker='o', s=scaled_community_size[mask], c='grey')

    # Draws propagated communities.
    plt.scatter(x=xs[propagated_community_index], y=ys[propagated_community_index],
                s=scaled_community_size[propagated_community_index], marker='o', cmap=plt.cm.Greens, c='green',
                norm=mpl.colors.Normalize(vmin=-2.5, vmax=2.5))

    # Draws community names.
    for idx, name in enumerate(community_names):
        if idx not in propagated_community_index:
            continue

        # text_color = "red" if len(propagated_community_index) > 0 and idx == propagated_community_index[0] else "black"
        text_color = "black"
        plt.annotate(xy=(xs[idx], ys[idx]), xytext=(3, 3), size=TEXT_SIZE,
                     textcoords='offset points', ha='left', va='top', color=text_color, s=name)

    # Saves plot to file.
    folder = 'plot_user_based_embedding' if is_user_based_embedding else 'plot_topic_based_embedding'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(folder + "/" + slang + '.eps', format='eps')


def convert_to_2D(data):
    # scaler = StandardScaler()
    # normalized_data = scaler.fit_transform(data)
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    data_2d = tsne_model.fit_transform(data)
    xs = data_2d[:, 0]
    ys = data_2d[:, 1]
    return xs, ys


def get_community_to_index():
    with open("data/common_word_based_topic_vector_1000subreddits/subreddit_names.json") as f:
        data = json.load(f)
    community_to_index = {community: idx for idx, community in enumerate(data)}
    return data, community_to_index


def visualize_slang_propagation(is_user_based_embedding):
    target_slang = {
        'doja', 'ganje',
    }

    community_names, community_to_index = get_community_to_index()
    if is_user_based_embedding:
        community_embeddings = np.load("data/pca_user_based_community_embedding/2018-10.npy")
    else:
        community_embeddings = np.load("data/common_word_based_topic_vector_1000subreddits/2018-10.npy")
    xs, ys = convert_to_2D(community_embeddings)

    df_community_size = pd.read_csv(COMMUNITY_SIZE_PERCENTAGE_PRIOR)
    df_community_size.set_index('subreddit', inplace=True)
    community_sizes = df_community_size.loc[community_names][TIME].values

    slangs = []
    propagated_community_indexs = []
    with open(SLANG_PROPAGATION_DATA_PATH) as f:
        slang_propagation_data = json.load(f)
        for slang, propagation_history in slang_propagation_data.items():
            if slang not in target_slang:
                continue

            propagated_community_idx_for_cur_slang = []
            for time_stamp, communities in propagation_history.items():
                for c in communities:
                    propagated_community_idx_for_cur_slang.append(community_to_index[c])

            propagated_community_indexs.append(propagated_community_idx_for_cur_slang)
            slangs.append(slang)

    for i, slang in enumerate(slangs):
        plot(community_names, community_sizes, propagated_community_indexs[i], slang, xs, ys, is_user_based_embedding)


def main():
    visualize_slang_propagation(is_user_based_embedding=True)
    visualize_slang_propagation(is_user_based_embedding=False)


if __name__ == "__main__":
    main()

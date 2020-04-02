import time
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from pylab import figure, text, scatter, show
from matplotlib import rcParams

LABEL_DATA = "data/slang_propagation_coverage_prediction/slang_propagation_coverage_label.csv"
FEATURE_DATA = "data/slang_propagation_coverage_prediction/slang_propagation_coverage_features.csv"
SLANG_PROPAGATION_DATA_PATH = "data/df_slang_propagation_label_1000subreddits_all_time.json"

SEED = 0
TRAIN_TEST_SPLIT = 0.8
SLANG_TO_REMOVE = {"B.A", "B.G", "B.M", "D.D", "Ph.D"}


def get_valid_slang_list(df_features):
    # Data with 0 feature value will be dropped.
    features_to_examine = {
        'valence',
        'concreteness',
        # 'contextual_diversity',
    }

    total_valid_slang_set = set(df_features.index.values)
    total_valid_slang_set -= SLANG_TO_REMOVE
    for feature in features_to_examine:
        valid_slang_set = set(df_features[df_features[feature] > 0].index.values)
        total_valid_slang_set = total_valid_slang_set.intersection(valid_slang_set)

    print("resulting valid slang number:", len(total_valid_slang_set))
    total_valid_slang_list = list(total_valid_slang_set)
    return total_valid_slang_list


def get_slang_in_specified_year(valid_slang_list):
    with open(SLANG_PROPAGATION_DATA_PATH) as f:
        slang_propagation_data = json.load(f)

    start_time = "2010-01"
    res_slang_list = []
    for s in valid_slang_list:
        slang_prop_history = slang_propagation_data[s]
        emerge_time = next(iter(slang_prop_history))
        if emerge_time >= start_time:
            res_slang_list.append(s)

    return res_slang_list


def get_slang_with_specified_length(valid_slang_list):
    res_slang_list = []
    for s in valid_slang_list:
        if len(s) > 8:
            res_slang_list.append(s)

    return res_slang_list


def apply_extra_constraint(valid_slang_list):
    # valid_slang_list = get_slang_in_specified_year(valid_slang_list)
    # valid_slang_list = get_slang_with_specified_length(valid_slang_list)
    return valid_slang_list


def drop_selected_features(df_features):
    # df_features = df_features.drop('slang_length', axis=1)
    # df_features = df_features.drop('syllable_count', axis=1)
    # df_features = df_features.drop('definition_length', axis=1)
    # df_features = df_features.drop('valence', axis=1)
    # df_features = df_features.drop('concreteness', axis=1)
    # df_features = df_features.drop('user_iou', axis=1)
    # df_features = df_features.drop('topic_similarity', axis=1)
    # df_features = df_features.drop('community_size', axis=1)
    # df_features = df_features.drop('contextual_diversity', axis=1)
    pass


def get_data():
    label_names_list = [
                     'user_based_propagation_coverage_score',
                     # 'user_based_propagation_coverage_speed',
                     'topic_based_propagation_coverage_score',
                     # 'topic_based_propagation_coverage_speed',
                 ]

    df_features = pd.read_csv(FEATURE_DATA)
    df_features.set_index('slang', inplace=True)
    valid_slang_list = get_valid_slang_list(df_features)
    valid_slang_list = apply_extra_constraint(valid_slang_list)
    df_features = df_features.loc[valid_slang_list]
    drop_selected_features(df_features)

    feature_data = df_features.values
    scaler = MinMaxScaler()
    normalized_feature_data = scaler.fit_transform(feature_data)

    df_label = pd.read_csv(LABEL_DATA)
    df_label.set_index('slang', inplace=True)
    df_label = df_label.loc[valid_slang_list]

    # Randomly split slang to train and test set.
    num_data = len(feature_data)
    slang_index = np.array([i for i in range(num_data)])
    np.random.seed(SEED)
    np.random.shuffle(slang_index)
    train_index = slang_index[:int(TRAIN_TEST_SPLIT * num_data)]
    test_index = slang_index[int(TRAIN_TEST_SPLIT * num_data):]
    trainX = normalized_feature_data[train_index]
    testX = normalized_feature_data[test_index]

    dataset = {}
    for label_name in label_names_list:
        label_data = df_label[label_name].values
        trainY = label_data[train_index]
        testY = label_data[test_index]

        trainset = (trainX, trainY)
        testset = (testX, testY)
        dataset[label_name] = (trainset, testset)

    features_names = df_features.columns.values
    return dataset, features_names


def visualize_coefficient(label_name, raw_feature_name, feature_importance, r2_train, r2_test):
    # Abbreviates feature names.
    feature_name_mapping = {
        "slang_length": "slang_len",
        "syllable_count": "syl_cnt",
        "definition_length": "def_len",
        "valence": "valence",
        "concreteness": "concrete",
        "user_iou": "user_mob",
        "topic_similarity": "topic_sim",
        "community_size": "comm_size",
        "contextual_diversity": "ctx_div",
    }
    new_feature_name = [feature_name_mapping[f] for f in raw_feature_name]

    rcParams.update({'figure.autolayout': True})
    plt.rc('font', size=7)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.grid(axis='y', linestyle='-', alpha=0.3)

    barlist = ax.bar(new_feature_name, feature_importance)
    for idx, bar in enumerate(barlist):
        if feature_importance[idx] >= 0:
            bar.set_color('green')

    # Sets the name of the most important feature to red.
    idx_largest_coefficient = abs(np.array(feature_importance)).argmax()
    ax.xaxis.get_ticklabels()[idx_largest_coefficient].set_color('red')

    plt.xticks(rotation=25)
    plt.xlabel("Feature Name", fontsize=9)
    plt.ylabel("Coefficient Value", fontsize=9)
    ax.set_title(label_name.replace('_', ' ').replace('coverage','spreadness').title() + " Prediction", fontsize=11)

    performance_text = "r2_train: %.3f\nr2_test: %.3f"  % (r2_train, r2_test)
    text(0.03, 0.9, performance_text,
         horizontalalignment='left',
         verticalalignment='center',
         transform=ax.transAxes,
         fontsize=10)

    folder = 'plot_coverage_prediction_model_coefficient'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(folder + "/" + label_name + '.eps', format='eps')


def main():
    model_list = {
        'LinearRegression': LinearRegression,
        # 'Ridge': Ridge,
    }
    dataset, features_names = get_data()

    for label_name, (trainset, testset) in dataset.items():
        print("\n================================= label: %s =================================" % label_name)
        trainX, trainY = trainset
        testX, testY = testset

        for model_name, model in model_list.items():
            print("- - - - - - %s model - - - - - -\n" % model_name)

            fitted_model = model().fit(trainX, trainY)
            fitted_coefficients = fitted_model.coef_
            for idx, coeff in enumerate(fitted_coefficients):
                print('%s: %.5f' % (features_names[idx], fitted_coefficients[idx]))

            predictedY = fitted_model.predict(testX)
            print('Mean squared error: %.5f' % mean_squared_error(testY, predictedY))
            r2_train = r2_score(trainY, fitted_model.predict(trainX))
            r2_test = r2_score(testY, predictedY)
            print('Train Coefficient of determination: %.5f' % r2_train)
            print('Test Coefficient of determination: %.5f' % r2_test)

            visualize_coefficient(label_name, features_names, fitted_coefficients, r2_train, r2_test)


if __name__ == "__main__":
    main()

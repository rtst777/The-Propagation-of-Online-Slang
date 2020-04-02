import time
import numpy as np

from util import *
from model.default_predictor import DefaultPredictor
from model.prior_only_predictor import PriorOnlyPredictor
from model.onenn_predictor import OneNNPredictor
from model.exemplar_predictor import ExemplarPredictor
from model.prototype_predictor import PrototypePredictor

TRAIN_SET_PERCENTAGE = 0.8
START_TIME = "2010-01"
SEED = 0

# SLANG_PROPAGATION_DATA_PATH = "data/df_slang_propagation_label_all.json"
SLANG_PROPAGATION_DATA_PATH = "data/df_slang_propagation_label_after_200912.json"
# SLANG_PROPAGATION_DATA_PATH = "data/df_slang_propagation_label_before_200912.json"

COMMUNITY_DIVERSITY_PERCENTAGE_PRIOR = "data/diversity_per_community_across_time.csv"
COMMUNITY_SIZE_PERCENTAGE_PRIOR = "data/user_num_per_community_across_time.csv"
COMMUNITY_TO_SLANG_ACROSS_TIME = "data/community_to_slang_across_time"
SLANG_COSINE_DISTANCE = "data/slang_distance/slang_cosine_distance.csv"
SLANG_LEVENSHTEIN_DISTANCE = "data/slang_distance/slang_levenshtein_distance.csv"
COMMUNITY_INDEX = "data/df_selected_community_list.csv"
USER_OVERLAP_RATIO = get_overlap_ratio_across_time(ratio_path="data/user_overlap_ratio",
                                                   community_list_path=COMMUNITY_INDEX, start_time=START_TIME)

def get_predictor_setting(model, community_prior=None, slang_distance=None, overlap_ratio=USER_OVERLAP_RATIO,
                          train_set_percentage=0.8):
    predictor_setting_template = {
        "model": model,
        "model_parameter": {"slang_propagation_data_path": SLANG_PROPAGATION_DATA_PATH,
                            "community_prior": community_prior,
                            "community_to_slang_across_time": COMMUNITY_TO_SLANG_ACROSS_TIME,
                            "slang_distance": slang_distance,
                            "community_index": COMMUNITY_INDEX,
                            "overlap_ratio": overlap_ratio,
                            "train_set_percentage": train_set_percentage,
                            "start_time": START_TIME,
                            "seed": SEED},
        "description": "diversity_percentage_size_percentage"
    }

    description = "_(train_set_percent=%0.2f)" % train_set_percentage
    if community_prior is None:
        description += "_uniform_percentage"
    elif "diversity" in community_prior:
        description += "_diversity_percentage"
    else:
        description += "_size_percentage"
    if overlap_ratio is None:
        description += "_uniform_community_relation"
    else:
        description += "_community_user_relation"
    if slang_distance is not None:
        description += "_slang_cosine_distance" if "cosine" in slang_distance else "_slang_levenshtein_distance"
    predictor_setting_template["description"] = description

    return predictor_setting_template


def get_prior_only_predictor_setting_sets():
    predictor_setting_sets = []
    for p in [TRAIN_SET_PERCENTAGE]:
        predictor_setting_sets.extend([
            get_predictor_setting(PriorOnlyPredictor, overlap_ratio=USER_OVERLAP_RATIO,
                                  community_prior=COMMUNITY_DIVERSITY_PERCENTAGE_PRIOR,
                                  train_set_percentage=p),
            get_predictor_setting(PriorOnlyPredictor, overlap_ratio=USER_OVERLAP_RATIO,
                                  community_prior=COMMUNITY_SIZE_PERCENTAGE_PRIOR,
                                  train_set_percentage=p),
            get_predictor_setting(PriorOnlyPredictor, overlap_ratio=USER_OVERLAP_RATIO,
                                  community_prior=None,
                                  train_set_percentage=p),
            get_predictor_setting(PriorOnlyPredictor, overlap_ratio=None,
                                  community_prior=COMMUNITY_DIVERSITY_PERCENTAGE_PRIOR,
                                  train_set_percentage=p),
            get_predictor_setting(PriorOnlyPredictor, overlap_ratio=None,
                                  community_prior=COMMUNITY_SIZE_PERCENTAGE_PRIOR,
                                  train_set_percentage=p),
        ])
    return predictor_setting_sets


def get_proposed_predictor_setting_sets(model):
    predictor_setting_sets = []
    for p in [TRAIN_SET_PERCENTAGE]:
        predictor_setting_sets.extend([
            get_predictor_setting(model, overlap_ratio=USER_OVERLAP_RATIO,
                                  community_prior=COMMUNITY_DIVERSITY_PERCENTAGE_PRIOR,
                                  slang_distance=SLANG_COSINE_DISTANCE,
                                  train_set_percentage=p),
            get_predictor_setting(model, overlap_ratio=USER_OVERLAP_RATIO,
                                  community_prior=COMMUNITY_DIVERSITY_PERCENTAGE_PRIOR,
                                  slang_distance=SLANG_LEVENSHTEIN_DISTANCE,
                                  train_set_percentage=p),

            get_predictor_setting(model, overlap_ratio=USER_OVERLAP_RATIO,
                                  community_prior=COMMUNITY_SIZE_PERCENTAGE_PRIOR,
                                  slang_distance=SLANG_COSINE_DISTANCE,
                                  train_set_percentage=p),
            get_predictor_setting(model, overlap_ratio=USER_OVERLAP_RATIO,
                                  community_prior=COMMUNITY_SIZE_PERCENTAGE_PRIOR,
                                  slang_distance=SLANG_LEVENSHTEIN_DISTANCE,
                                  train_set_percentage=p),

            get_predictor_setting(model, overlap_ratio=USER_OVERLAP_RATIO,
                                  community_prior=None,
                                  slang_distance=SLANG_COSINE_DISTANCE,
                                  train_set_percentage=p),
            get_predictor_setting(model, overlap_ratio=USER_OVERLAP_RATIO,
                                  community_prior=None,
                                  slang_distance=SLANG_LEVENSHTEIN_DISTANCE,
                                  train_set_percentage=p),

            get_predictor_setting(model, overlap_ratio=None,
                                  community_prior=COMMUNITY_DIVERSITY_PERCENTAGE_PRIOR,
                                  slang_distance=SLANG_COSINE_DISTANCE,
                                  train_set_percentage=p),
            get_predictor_setting(model, overlap_ratio=None,
                                  community_prior=COMMUNITY_DIVERSITY_PERCENTAGE_PRIOR,
                                  slang_distance=SLANG_LEVENSHTEIN_DISTANCE,
                                  train_set_percentage=p),

            get_predictor_setting(model, overlap_ratio=None,
                                  community_prior=COMMUNITY_SIZE_PERCENTAGE_PRIOR,
                                  slang_distance=SLANG_COSINE_DISTANCE,
                                  train_set_percentage=p),
            get_predictor_setting(model, overlap_ratio=None,
                                  community_prior=COMMUNITY_SIZE_PERCENTAGE_PRIOR,
                                  slang_distance=SLANG_LEVENSHTEIN_DISTANCE,
                                  train_set_percentage=p),

            get_predictor_setting(model, overlap_ratio=None,
                                  community_prior=None,
                                  slang_distance=SLANG_COSINE_DISTANCE,
                                  train_set_percentage=p),
            get_predictor_setting(model, overlap_ratio=None,
                                  community_prior=None,
                                  slang_distance=SLANG_LEVENSHTEIN_DISTANCE,
                                  train_set_percentage=p),
            ])

    return predictor_setting_sets


def main():
    time_stamp = str(int(time.time()))
    logger = get_logger(time_stamp)
    predictor_settings = [
        get_predictor_setting(DefaultPredictor, community_prior=COMMUNITY_SIZE_PERCENTAGE_PRIOR,
                              train_set_percentage=TRAIN_SET_PERCENTAGE),
        *get_prior_only_predictor_setting_sets(),
        *get_proposed_predictor_setting_sets(OneNNPredictor),
        *get_proposed_predictor_setting_sets(ExemplarPredictor),
        *get_proposed_predictor_setting_sets(PrototypePredictor),
    ]

    model_rocs = {}
    # TODO: parallelize model training.
    for i, predictor_setting in enumerate(predictor_settings):
        predictors = predictor_setting["model"](**predictor_setting["model_parameter"])
        model_saving_name = predictors.model_name + "_" + predictor_setting["description"]
        # save_model(predictors, time_stamp, model_saving_name)

        logger.info(msg="============ %s ============" % model_saving_name)

        training_expected_rank_first, training_roc_first, training_expected_rank_all, training_roc_all = predictors.train()
        logger.info(msg="training_expected_rank_first is %f" % training_expected_rank_first)
        logger.info(msg="training_roc_first is %f" % np.mean(training_roc_first))
        logger.info(msg="training_expected_rank_all is %f" % training_expected_rank_all)
        logger.info(msg="training_roc_all is %f\n" % np.mean(training_roc_all))

        test_expected_rank_first, test_roc_first, test_expected_rank_all, test_roc_all = predictors.predict()
        logger.info(msg="test_expected_rank_first is %f" % test_expected_rank_first)
        logger.info(msg="test_roc_first is %f" % np.mean(test_roc_first))
        logger.info(msg="test_expected_rank_all is %f" % test_expected_rank_all)
        logger.info(msg="test_roc_all is %f\n" % np.mean(test_roc_all))

        model_rocs[model_saving_name] = {
            "training_roc_first": list(training_roc_first),
            "test_roc_first": list(test_roc_first),
            "training_roc_all": list(training_roc_all),
            "test_roc_all": list(test_roc_all),
        }

    save_model_roc(model_rocs, time_stamp)


if __name__ == "__main__":
    main()

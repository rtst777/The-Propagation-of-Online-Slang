import pickle
import os
import logging
import sys
import json
import pandas as pd
import operator as op
from functools import reduce

def ncr(n, r):
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer / denom


def get_overlap_ratio_across_time(ratio_path, community_list_path, start_time):
    df_community = pd.read_csv(community_list_path)
    community_list = df_community["subreddit"].values
    first_pre_time = get_pre_timestamp(start_time)

    overlap_ratio_across_time = {}
    time_ranges = [f[:-4] for f in os.listdir(ratio_path) if os.path.isfile(os.path.join(ratio_path, f)) and ".csv" in f]
    time_ranges.sort()

    for time_stamp in time_ranges:
        if time_stamp < first_pre_time:
            continue
        file_path = os.path.join(ratio_path, time_stamp + ".csv")
        df = pd.read_csv(file_path)
        df.set_index("subreddit", inplace=True)
        overlap_ratio_across_time[time_stamp] = df.loc[community_list][community_list].values

    return overlap_ratio_across_time


def get_pre_timestamp(cur_timestamp):
    # pre_slang_emergence_time = slang_emergence_time - 1
    time_tokens = cur_timestamp.split('-')
    year = (int)(time_tokens[0])
    month = (int)(time_tokens[1])
    if month == 1:
        pre_slang_emergence_time = str(year - 1) + '-12'
    elif month > 10:
        pre_slang_emergence_time = str(year) + '-' + str(month - 1)
    else:
        pre_slang_emergence_time = str(year) + '-0' + str(month - 1)
    return pre_slang_emergence_time


def save_model(model, time_stamp, model_saving_name):
    if not os.path.exists("log"):
        os.makedirs("log")

    directory = os.path.join("log", time_stamp)
    if not os.path.exists(directory):
        os.makedirs(directory)

    path = os.path.join(directory, model_saving_name + ".pkl")
    with open(path, 'wb') as output:
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)


def load_model(time_stamp, model_saving_name):
    path = "log/" + time_stamp + "/" + model_saving_name + ".pkl"
    with open(path, 'rb') as input:
        model = pickle.load(input)
    return model


def save_model_roc(model_rocs, time_stamp):
    if not os.path.exists("log"):
        os.makedirs("log")

    directory = os.path.join("log", time_stamp)
    if not os.path.exists(directory):
        os.makedirs(directory)

    path = os.path.join(directory, "model_rocs.json")
    with open(path, "w") as f:
        json.dump(model_rocs, f, indent=2)


def get_logger(time_stamp):
    if not os.path.exists("log"):
        os.makedirs("log")

    directory = os.path.join("log", time_stamp)
    if not os.path.exists(directory):
        os.makedirs(directory)

    log_level = logging.INFO
    log_format = "%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s"
    logging.basicConfig(filename=directory + "/log_message",
                        format=log_format,
                        datefmt="%H:%M:%S",
                        level=log_level)
    logger = logging.getLogger("Slang_Acceptor_Prediction")

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(handler)

    return logger

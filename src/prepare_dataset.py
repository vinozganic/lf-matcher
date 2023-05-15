import os
import random

import pandas as pd
from pymongo import MongoClient

from constants import ITEMS_URL
from contracts import item_to_process
from data_transformer import DataTransformer


client = MongoClient(ITEMS_URL)

database = client.get_database("lf")

losts = database.get_collection("losts").find()
founds = database.get_collection("founds").find()

losts_list = list(losts)
founds_list = list(founds)

losts_train_list = losts_list[:int(len(losts_list) * 0.8)]
founds_train_list = founds_list[:int(len(founds_list) * 0.8)]

losts_test_list = losts_list[int(len(losts_list) * 0.8):]
founds_test_list = founds_list[int(len(founds_list) * 0.8):]


prepared_matches = []
prepared_non_matches = []

data_transformer = DataTransformer()

type_similarity_matrix = data_transformer.calculate_similarity_matrixes()

matching_pairs = []
non_matching_pairs = []
for lost_idx, lost in enumerate(losts_train_list):
    for found_idx, found in enumerate(founds_train_list):
        if lost_idx == found_idx:
            matching_pairs.append((lost_idx, found_idx))
        elif lost_idx != found_idx:
            non_matching_pairs.append((lost_idx, found_idx))

for lost_idx, found_idx in matching_pairs:
    lost_to_process = item_to_process(id=losts_train_list[lost_idx]["_id"], item_type="lost", type_=losts_train_list[lost_idx]["type"], color=losts_train_list[lost_idx]["color"], location=losts_train_list[lost_idx]["location"], date=losts_train_list[lost_idx]["date"])
    found_to_process = item_to_process(id=founds_train_list[found_idx]["_id"], item_type="found", type_=founds_train_list[found_idx]["type"], color=founds_train_list[found_idx]["color"], location=founds_train_list[found_idx]["location"], date=founds_train_list[found_idx]["date"])
    prepared = data_transformer.prepare_data(lost_to_process, found_to_process)
    prepared_df = pd.DataFrame(prepared, index=[0])

    prepared_df["label"] = 1
    prepared_matches.append(prepared_df)

random.shuffle(non_matching_pairs)
for lost_idx, found_idx in non_matching_pairs[:len(matching_pairs)]:
    lost_to_process = item_to_process(id=losts_train_list[lost_idx]["_id"], item_type="lost", type_=losts_train_list[lost_idx]["type"], color=losts_train_list[lost_idx]["color"], location=losts_train_list[lost_idx]["location"], date=losts_train_list[lost_idx]["date"])
    found_to_process = item_to_process(id=founds_train_list[found_idx]["_id"], item_type="found", type_=founds_train_list[found_idx]["type"], color=founds_train_list[found_idx]["color"], location=founds_train_list[found_idx]["location"], date=founds_train_list[found_idx]["date"])
    prepared = data_transformer.prepare_data(lost_to_process, found_to_process)
    prepared_df = pd.DataFrame(prepared, index=[0])

    prepared_df["label"] = 0
    prepared_non_matches.append(prepared_df)

prepared_balanced = prepared_matches + prepared_non_matches

df = pd.concat(prepared_balanced)   

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df.to_csv(current_dir + "/model/data.csv")

matching_test_pairs = []
non_matching_test_pairs = []

for lost_idx, lost in enumerate(losts_test_list):
    for found_idx, found in enumerate(founds_test_list):
        if lost_idx == found_idx:
            matching_test_pairs.append((lost_idx, found_idx))
        elif lost_idx != found_idx:
            non_matching_test_pairs.append((lost_idx, found_idx))

test_data = []
for lost_idx, found_idx in matching_test_pairs:
    lost_to_process = item_to_process(id=losts_test_list[lost_idx]["_id"], item_type="lost", type_=losts_test_list[lost_idx]["type"], color=losts_test_list[lost_idx]["color"], location=losts_test_list[lost_idx]["location"], date=losts_test_list[lost_idx]["date"])
    found_to_process = item_to_process(id=founds_test_list[found_idx]["_id"], item_type="found", type_=founds_test_list[found_idx]["type"], color=founds_test_list[found_idx]["color"], location=founds_test_list[found_idx]["location"], date=founds_test_list[found_idx]["date"])
    prepared = data_transformer.prepare_data(lost_to_process, found_to_process)
    prepared_df = pd.DataFrame(prepared, index=[0])

    prepared_df["label"] = 1
    test_data.append(prepared_df)

random.shuffle(non_matching_test_pairs)
for lost_idx, found_idx in non_matching_test_pairs[:len(matching_test_pairs)]:
    lost_to_process = item_to_process(id=losts_test_list[lost_idx]["_id"], item_type="lost", type_=losts_test_list[lost_idx]["type"], color=losts_test_list[lost_idx]["color"], location=losts_test_list[lost_idx]["location"], date=losts_test_list[lost_idx]["date"])
    found_to_process = item_to_process(id=founds_test_list[found_idx]["_id"], item_type="found", type_=founds_test_list[found_idx]["type"], color=founds_test_list[found_idx]["color"], location=founds_test_list[found_idx]["location"], date=founds_test_list[found_idx]["date"])
    prepared = data_transformer.prepare_data(lost_to_process, found_to_process)
    prepared_df = pd.DataFrame(prepared, index=[0])

    prepared_df["label"] = 0
    test_data.append(prepared_df)

prepared_test_balanced = test_data

test_df = pd.concat(test_data)
test_df.to_csv(current_dir + "/model/test_data.csv")


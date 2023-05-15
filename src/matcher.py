import logging
import requests
from typing import Generator, List

import numpy as np
import pandas as pd

from constants import API_URL
from contracts import item_to_process, match_result
from exceptions import APIException


def patch_asscalar(a):
    return a.item()

setattr(np, 'asscalar', patch_asscalar)
    

class Matcher:
    def __init__(self, model, data_transformer):
        self.model = model
        self.data_transformer = data_transformer

    def process_message(self, message: item_to_process) -> match_result:
        match_results = self.run_predictions(message)
        self.save_matches_to_db(match_results)
        # TODO: send message to notifier service

    def run_predictions(self, item: item_to_process) -> Generator[match_result, None, None]:
        type_to_query = "found" if item.item_type == "lost" else "lost"
        items_to_compare = self.get_items_from_db(type_to_query)
        for item_to_compare in items_to_compare:
            lost_item = item if item.item_type == "lost" else item_to_process.from_dict(item_to_compare)
            found_item = item if item.item_type == "found" else item_to_process.from_dict(item_to_compare)

            prepared_data = self.data_transformer.prepare_data(lost_item, found_item)
            prepared_df = pd.DataFrame(prepared_data, index=[0])
            prepared_df = prepared_df.reindex(columns=self.model.feature_names, fill_value=0)

            probability = self.model.predict(prepared_df.values)
            if probability > 0.05:
                yield match_result(lost_id=lost_item.id, found_id=found_item.id, match_probability=probability)

    ## SPREMANJE LOKACIJA SAD RADI, DALJE TREBA POBOLJŠATI MODEL
    def get_items_from_db(self, item_type):
        response = requests.get(f"{API_URL}/{item_type}").json()
        if response["success"]:
            data = response["data"]
            for item in data:
                item.update({"item_type": item_type})
            return data
        else:
            raise APIException("Could not get items from database")

    def save_matches_to_db(self, match_results: List[match_result]):
        payload = [result.to_dict() for result in match_results]
        response = requests.post(f"{API_URL}/matches/batch", json=payload).json()
        if not response["success"]:
            raise APIException("Could not save match results to database")
    
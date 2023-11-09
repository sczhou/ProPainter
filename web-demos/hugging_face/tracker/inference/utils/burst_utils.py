from os import path
import copy
import json


class BURSTResultHandler:
    def __init__(self, dataset_json):
        self.dataset_json = copy.deepcopy(dataset_json)

        # get rid of the segmentations while keeping the metadata
        self.dataset_json['sequences'] = []

    def add_sequence(self, sequence_json):
        self.dataset_json['sequences'].append(sequence_json)

    def dump(self, root):
        json_path = path.join(root, 'predictions.json')
        with open(json_path, 'w') as f:
            json.dump(self.dataset_json, f)
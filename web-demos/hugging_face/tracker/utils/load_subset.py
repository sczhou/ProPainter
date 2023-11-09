import json


def load_subset(path):
    with open(path, mode='r') as f:
        subset = set(f.read().splitlines())
    return subset


def load_empty_masks(path):
    with open(path, mode='r') as f:
        empty_masks = json.load(f)
    return empty_masks

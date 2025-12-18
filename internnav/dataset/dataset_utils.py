import json
import re

def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0

def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]
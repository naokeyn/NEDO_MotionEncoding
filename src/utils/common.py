import json

def load_json(path):
    with open(path, "r") as r:
        contents = json.load(r)
    return contents

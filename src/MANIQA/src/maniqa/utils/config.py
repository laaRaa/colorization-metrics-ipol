"""JSON configuration."""

import json


class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        with open(file, encoding="utf-8") as f:
            config = json.loads(f.read())
            return Config(config)


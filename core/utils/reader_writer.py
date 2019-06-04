import json
import pickle as pk
from typing import Dict, Any

import numpy as np


def read_numpy(path: str) -> np.ndarray:
    assert path.endswith(".npy")

    return np.load(path)


def read_json(path: str) -> Dict[Any, Any]:
    assert path.endswith(".json")

    return json.load(open(path, "rb"))


def write_json(f, path: str):
    assert path.endswith(".json")

    json.dump(f, open(path, 'wb'))


def write_pickle(f, path: str):
    assert path.endswith(".pkl")

    pk.dump(f, open(path, 'wb'))


def read_pickle(path: str) -> Any:
    assert path.endswith(".pkl")

    return pk.load(open(path, "rb"))

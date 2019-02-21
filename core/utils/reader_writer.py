import numpy as np
import pandas as pd
import json
import pickle as pk


def read_numpy(path : str):

    assert path.endswith(".npy")

    return np.load(path)


def read_csv(path : str, sep : str =","):

    assert path.endswith(".csv")

    return pd.read_csv(path, sep=sep)


def read_json(path : str):

    assert path.endswith(".json")

    return json.load(open(path, "rb"))


def write_json(f, path : str):

    assert path.endswith(".json")

    json.dump(f, open(path, 'wb'))


def write_pickle(f, path : str):

    assert path.endswith(".pkl")

    pk.dump(f, open(path, 'wb'))


def read_pickle(path : str):

    assert path.endswith(".pkl")

    return pk.load(open(path, "rb"))
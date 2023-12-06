import pandas as pd
import numpy as np
from scipy.stats import entropy, differential_entropy

from typing import Iterable


def get_discrete_variety_score(s: Iterable) -> float:
    v, c = np.unique(s, return_counts=True)
    return entropy(c)

def get_conti_variety_score(s:Iterable)->float:

    return differential_entropy(s)

def log_normalize(s:np.ndarray):
    return normalize(np.log(s))

def normalize(s:np.ndarray):
    """
    z-scroe normalization
    """
    mu = np.mean(s)
    std = np.std(s)
    return (s - mu)/std
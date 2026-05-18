import numpy as np

def scale(x):
    """Standardize a variable with the z-score."""
    return (x - x.mean()) / x.std()


def invlogit(x):
    """Take the inverse logit, or expit, of a variable"""
    return 1 / (1 + np.exp(-x))
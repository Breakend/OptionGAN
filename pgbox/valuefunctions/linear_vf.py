import tensorflow as tf
import numpy as np
from pgbox.utils import *
from pgbox.tf_utils import *


class LinearVF(object):
    coeffs = None

    def __init__(self, args, **kwargs):
        self.kwargs = kwargs

    def _features(self, path):
        o = path["observations"].astype('float32')
        o = o.reshape(o.shape[0], -1)
        l = len(path["rewards"])
        al = np.arange(l).reshape(-1, 1) / 100.0
        return np.concatenate([o, o**2, al, al**2, np.ones((l, 1))], axis=1)

    def fit(self, paths, args, **kwargs):
        featmat = np.concatenate([self._features(path) for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])
        n_col = featmat.shape[1]
        lamb = 2.0
        self.coeffs = np.linalg.lstsq(featmat.T.dot(featmat) + lamb * np.identity(n_col), featmat.T.dot(returns))[0]

    def predict(self, path, args, **kwargs):
        return np.zeros(len(path["rewards"])) if self.coeffs is None else self._features(
            path).dot(self.coeffs)

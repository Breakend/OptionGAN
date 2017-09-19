from .running_stat import RunningStat
import numpy as np

class Composition(object):
    def __init__(self, fs):
        self.fs = fs
    def __call__(self, x, update=True):
        for f in self.fs:
            x = f(x)
        return x
    def output_shape(self, input_space):
        out = input_space.shape
        for f in self.fs:
            out = f.output_shape(out)
        return out

class ZFilter(object):
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=100.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        assert x.shape == self.rs.mean.shape
        if update: self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std+1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x
    def output_shape(self, input_space):
        return input_space.shape


class DefaultFilter:
    def __init__(self, filter_mean=True):
        self.m1 = 0
        self.v = 0
        self.n = 0.
        self.filter_mean = filter_mean

    def __call__(self, o):
        self.m1 = self.m1 * (self.n / (self.n + 1)) + o    * 1/(1 + self.n)
        self.v = self.v * (self.n / (self.n + 1)) + (o - self.m1)**2 * 1/(1 + self.n)
        self.std = (self.v + 1e-6)**.5 # std
        self.n += 1
        if self.filter_mean:
            o1 =  (o - self.m1)/self.std
        else:
            o1 =  o/self.std
        o1 = (o1 > 10) * 10 + (o1 < -10)* (-10) + (o1 < 10) * (o1 > -10) * o1
        return o1

class Flatten(object):
    def __call__(self, x, update=True):
        return x.ravel()
    def output_shape(self, input_space):
        return (int(np.prod(input_space.shape)),)

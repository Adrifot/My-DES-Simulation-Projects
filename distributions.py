"""
Distribution utility classes to be used in other projects.
"""

import numpy as np
from typing import Optional, Union, Tuple, Any
from numpy.random import SeedSequence
from numpy.typing import NDArray, ArrayLike
import math


class Lognormal:
    def __init__ (
        self,
        mean: float,
        stdev: float,
        seed: Optional[Union[int, SeedSequence]] = None
    ):
        self.rng = np.random.default_rng(seed)
        mu, sigma = self.normal_moments_from_lognormal(mean, stdev**2)
        self.mu = mu
        self.sigma = sigma
        self.mean = mean
        self.stdev = stdev
        
        
    def __repr__(self):
        return f"Lognormal(mean={self.mean}, stdev={self.stdev})"
    
    
    def normal_moments_from_lognormal(self, m:float, v:float) -> Tuple[float, float]:
        phi = math.sqrt(v + m**2)
        mu = math.log(m**2 / phi)
        sigma = math.sqrt(math.log(phi**2 / m**2))
        return mu, sigma
    
    
    def sample(self, size: Optional[Union[int, Tuple[int, ...]]] = None):
        return self.rng.lognormal(self.mu, self.sigma, size)
        
        
        
class Exponential:
    def __init__(self, mean:float, seed:Optional[Union[int, SeedSequence]]=None):
        self.rand = np.random.default_rng(seed)
        self.mean = mean
        
        
    def sample(self, size:Optional[Union[int, Tuple[int, ...]]]=None):
        return self.rand.exponential(self.mean, size)
    


class Normal:
    def __init__(self, mean:float, stdev:float, seed:Optional[Union[int, SeedSequence]]=None):
        self.rand = np.random.default_rng(seed)
        self.mean = mean
        self.stdev = stdev

    def sample(self, size:Optional[Union[int, Tuple[int, ...]]]=None):
        return self.rand.normal(self.mean, self.stdev, size)
    


class Bernoulli:
    def __init__(self, p:float, seed:Optional[Union[int, SeedSequence]]=None):
        self.rand = np.random.default_rng(seed)
        self.p = p

    def sample(self, size:Optional[Union[int, Tuple[int, ...]]]=None):
        return self.rand.binomial(n=1, p=self.p, size=size)
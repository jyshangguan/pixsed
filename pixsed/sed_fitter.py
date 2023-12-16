import numpy as np
import matplotlib.pyplot as plt
import dynesty
from sedpy.observate import load_filters

from .sed_model import SEDModel_single

class SEDfitter_single(object):
    '''
    This is the object of a SED fitter using Bayesian inference.
    '''
    def __init__(self, data, models, zred=0, DL=None, verbose=True):
        '''
        Parameters
        ----------
        data : dict
            A dictionary containing the data to be fitted.
            The keys are the names of the bands, and the values are
            dictionaries containing the following keys:
            'flux' : list of float
                The flux of the band in mJy.
            'flux_err' : list of float
                The error of the flux in mJy
            'bands' : list of string
                The name of the bands that should follow the format of sedpy. 
                `https://github.com/bd-j/sedpy/blob/main/sedpy/data/filters/README.md`
        '''
        self._data = data
        self._filters = load_filters(data['bands'])
        self._models = models
        self._verbose = verbose

        self._sedmodel = SEDModel_single(models=models, zred=zred, DL=DL, verbose=verbose)
        self._zred = zred
        self._DL = self._sedmodel._DL


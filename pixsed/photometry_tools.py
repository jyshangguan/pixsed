import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rc("xtick", direction="in", labelsize=16)
mpl.rc("ytick", direction="in", labelsize=16)
mpl.rc("xtick.major", width=1., size=8)
mpl.rc("ytick.major", width=1., size=8)
mpl.rc("xtick.minor", width=1., size=5)
mpl.rc("ytick.minor", width=1., size=5)


class Image(object):
    """
    The class of an image.
    """
    def __init__(self, data, header):
        '''
        Parameters
        ----------
        data : numpy 2D array
            The 2D image data.
        hearder : astropy fits header
            The header of the image data.
        '''
        self._data_org = data
        self._data = data.copy()
        self.header = header
        self.wcs = None
        self.shape = data.shape
        self.background = None
        self.mask = None
        self.segmentation = None

    def get_wcs(self, wcs=None):
        '''
        Get the WCS information.

        Parameters
        ----------
        wcs : WCS or HEADER
            The wcs information.
        '''
        if wcs is None:
            self.wcs = WCS(self.header)
        else:
            self.wcs = wcs

    
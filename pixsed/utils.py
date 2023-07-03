import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
from astropy.visualization import AsinhStretch, SqrtStretch, LogStretch
from astropy.visualization import PercentileInterval
from astropy.visualization.mpl_normalize import ImageNormalize

stretchDict = {'asinh': AsinhStretch(), 'sqrt': SqrtStretch(), 'log': LogStretch()}


def plot_image(data, ax=None, percentile=99.5, vmin=None, vmax=None, stretch=None,
               origin='lower', cmap='gray', **kwargs):
    """
    Plot the image.

    Parameters
    ----------
    data : 2d array
        The image data to plot.
    ax : Figure axis
        The axis handle of the figure.
    vmin : float
        The minimum scale of the image.
    vmax : float
        The maximum scale of the image.
    stretch : stretch object
        The stretch used to normalize the image color scale.
    **kwargs : float
        The parameters of imshow() except the image and norm.

    Returns
    -------
    ax : Figure axis
        The handle of the image axis.

    Notes
    -----
    None.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    if stretch is None:
        stretch = LogStretch()
    else:
        stretch = stretchDict[stretch]

    norm = ImageNormalize(interval=PercentileInterval(percentile), vmin=vmin, vmax=vmax, stretch=stretch)
    if 'norm' not in kwargs:
        kwargs['norm'] = norm

    ax.imshow(data, origin=origin, cmap=cmap, **kwargs)
    ax.minorticks_on()
    return ax


def read_coordinate(ra, dec):
    '''
    Read in the coordinate, either in degree or hourangle. Only use ICRS frame.
    
    Parameters
    ----------
    ra : float or string
        The right ascension (degree or HH:MM:SS).
    dec : float or string
        The declination (degree or DD:MM:SS).
    
    Returns
    -------
    c : SkyCoord
        The coordinate object.
    '''
    if isinstance(ra, str):
        assert isinstance(dec, str)
        c = SkyCoord('{0} {1}'.format(ra, dec), frame='icrs', unit=(u.hourangle, u.deg))
    else:
        c = SkyCoord(ra, dec, frame='icrs', unit='deg')
    return c


def circular_error_estimate(img, mask, radius, nexample):
    std_list = []
    count = 0
    xllim = radius + 1
    xhlim = np.shape(img)[0] - radius - 1
    yllim = radius + 1
    yhlim = np.shape(img)[1] - radius - 1
    while count <= nexample:
        x_cen = np.random.randint(xllim, xhlim)
        y_cen = np.random.randint(yllim, yhlim)
        if mask[y_cen, x_cen] or not isinstance(img[y_cen, x_cen], float):
            continue
        flag = 0
        data_sum = 0
        for ar in range(-radius, radius + 1):
            for br in range(-radius, radius + 1):
                if mask[y_cen + ar, x_cen + br] or not isinstance(img[y_cen + ar, x_cen + br], float):
                    flag = 1
                else:
                    data_sum += img[y_cen + ar, x_cen + br]
        if flag == 0:
            count += 1
            std_list.append(data_sum)
    mean, mead, std = sigma_clipped_stats(np.array(std_list))
    return std

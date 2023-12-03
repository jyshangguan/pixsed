import math
import random

import tqdm
import warnings
import numpy as np
import numpy.ma as ma
from copy import deepcopy
import astropy.units as units
import matplotlib
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats, gaussian_fwhm_to_sigma, SigmaClip
from astropy.visualization import AsinhStretch, SqrtStretch, LogStretch
from astropy.visualization import PercentileInterval, simple_norm
from astropy.visualization.mpl_normalize import ImageNormalize
from astroquery.xmatch import XMatch
from astropy.convolution import convolve
from astropy.wcs import WCS
from astropy.table import Table
from astropy.modeling import models, fitting
from astropy.nddata import Cutout2D
from astropy.utils.exceptions import AstropyUserWarning
from scipy.ndimage import gaussian_filter
from scipy.optimize import root_scalar, differential_evolution
from scipy.interpolate import interp1d

from photutils.segmentation import make_2dgaussian_kernel, detect_sources, deblend_sources
from photutils.segmentation import SourceFinder, SegmentationImage, SourceCatalog
from photutils.morphology import data_properties
from photutils.isophote import EllipseGeometry, Ellipse, EllipseSample, Isophote, IsophoteList
from photutils.aperture import (EllipticalAperture, EllipticalAnnulus,
                                ApertureStats, aperture_photometry)
from rasterio.features import rasterize, shapes
from shapely.geometry import shape, MultiPoint
from shapely.affinity import scale
from reproject import reproject_interp, reproject_adaptive

stretchDict = {'asinh': AsinhStretch(), 'sqrt': SqrtStretch(), 'log': LogStretch()}


def adapt_mask(mask, input_wcs, output_wcs, shape_out, verbose=False):
    '''
    Adapt the mask.

    Notes
    -----
    FIXME: doc!
    '''
    # Check whether the wcs matches
    no_reproject = True
    if (input_wcs.wcs.crpix != output_wcs.wcs.crpix).all():
        no_reproject = False

    if (input_wcs.wcs.crval != output_wcs.wcs.crval).all():
        no_reproject = False

    if (mask.shape[0] != shape_out[0]) | (mask.shape[1] != shape_out[1]):
        no_reproject = False

    if no_reproject:
        mask_o = mask.copy()

        if verbose:
            print('[adapt_segmentation] Same WCS, no reprojection')
    else:
        mask_o, _ = reproject_interp((mask, input_wcs), output_wcs,
                                     shape_out=shape_out, order='nearest-neighbor')
        mask_o = mask_o > 0

    return mask_o


def adapt_segmentation(segm, input_wcs, output_wcs, shape_out, verbose=False):
    '''
    Adapt the segmentation.

    Notes
    -----
    FIXME: doc!
    '''
    # Check whether the wcs matches
    no_reproject = True
    if (input_wcs.wcs.crpix != output_wcs.wcs.crpix).all():
        no_reproject = False

    if (input_wcs.wcs.crval != output_wcs.wcs.crval).all():
        no_reproject = False

    if (segm.shape[0] != shape_out[0]) | (segm.shape[1] != shape_out[1]):
        no_reproject = False

    if no_reproject:
        segm_data = segm.data.copy()

        if verbose:
            print('[adapt_segmentation] Same WCS, no reprojection')
    else:
        segm_data, _ = reproject_interp((segm.data, input_wcs), output_wcs,
                                        shape_out=shape_out, order='nearest-neighbor')

    segm_o = SegmentationImage(np.round(segm_data).astype(int))
    return segm_o


def add_mask_circle(mask, x, y, radius, value=1):
    '''
    Add a circular mask in the input mask array.

    Parameters
    ----------
    mask : 2D bool array
        Input mask, True for masked region.
    x, y : int
        The pixel coordinate in the mask.
    radius : float
        The radius of the added mask.
    value : int or float
        The value put in the mask.

    Returns
    -------
    mask : 2D bool array
        Output mask, True for masked region.
    '''
    nx = np.arange(mask.shape[1])
    ny = np.arange(mask.shape[0])
    xx, yy = np.meshgrid(nx, ny)
    r = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    mask[r <= radius] = value
    return mask


def add_mask_ellipse(mask, x, y, sma, smb, pa, value=1):
    '''
        Add a circular mask in the input mask array.

        Parameters
        ----------
        mask : 2D bool array
            Input mask, True for masked region.
        x, y : int
            The pixel coordinate in the mask.
        sma : float
            The semi-major of the added mask.
        smb : float
            The semi-minor of the added mask.
        pa : float
            The theta of the added mask.
        value : int or float
            The value put in the mask.

        Returns
        -------
        mask : 2D bool array
            Output mask, True for masked region.
        '''
    nx = np.arange(mask.shape[1])
    ny = np.arange(mask.shape[0])
    xx, yy = np.meshgrid(nx, ny)
    sin = -math.sin(pa)
    cos = math.cos(pa)
    r = np.sqrt(((xx - x) * cos - (yy - y) * sin) ** 2 / sma ** 2 + \
                ((xx - x) * sin + (yy - y) * cos) ** 2 / smb ** 2)
    mask[r <= 1.] = value
    return mask


def add_mask_rect(mask, xmin, xmax, ymin, ymax, value=1):
    '''
    Add a rectangular mask in the input mask array.

    Parameters
    ----------
    mask : 2D bool array
        Input mask, True for masked region.
    xmin, xmax, ymin, ymax : int
        The ranges of x and y axes
    value : int or float
        The value put in the mask.

    '''
    nx = np.arange(mask.shape[1])
    ny = np.arange(mask.shape[0])
    xx, yy = np.meshgrid(nx, ny)
    fltr = (xx > xmin) & (xx < xmax) & (yy > ymin) & (yy < ymax)
    mask[fltr] = value
    return mask


def band_center_wavelength(band):
    '''
    Get the central wavelength of the band.
    We adopt the effective wavelength according to SVO
        http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php
    '''
    wDict = {
        'GALEX_FUV': 1548.85*units.Angstrom,
        'GALEX_NUV': 2303.37*units.Angstrom,
        'SDSS_u': 3608.04*units.Angstrom,
        'SDSS_g': 4671.78*units.Angstrom,
        'SDSS_r': 6141.12*units.Angstrom,
        'SDSS_i': 7457.89*units.Angstrom,
        'SDSS_z': 8922.78*units.Angstrom,
        '2MASS_J': 1.2350*units.micron,
        '2MASS_H': 1.6620*units.micron,
        '2MASS_Ks': 2.1590*units.micron,
        'WISE_W1': 3.3526*units.micron,
        'WISE_W2': 4.6028*units.micron,
        'WISE_W3': 11.5608*units.micron,
        'WISE_W4': 22.0883*units.micron,
        'PACS_70': 68.919665*units.micron,
        'PACS_100': 97.898566*units.micron,
        'PACS_160': 154*units.micron,
        'SPIRE_250': 243*units.micron,
        'SPIRE_350': 341*units.micron,
        'SPIRE_500': 483*units.micron,
    }
    w = wDict.get(band, None)
    return w


def circular_error_estimate(img, mask, radius, nexample, percent=85.):
    std_list = []
    count = 0
    xllim = radius + 1
    xhlim = np.shape(img)[0] - radius - 1
    yllim = radius + 1
    yhlim = np.shape(img)[1] - radius - 1
    while count <= nexample:
        x_cen = np.random.randint(xllim, xhlim)
        y_cen = np.random.randint(yllim, yhlim)
        if mask[y_cen, x_cen] or img[y_cen, x_cen] == 'nan':
            continue
        count_mask = 0
        count_num = 0
        data_sum = 0
        for ar in range(-radius, radius + 1):
            for br in range(-radius, radius + 1):
                if mask[y_cen + ar, x_cen + br] or img[y_cen + ar, x_cen + br] == 'nan':
                    count_mask += 1
                else:
                    data_sum += img[y_cen + ar, x_cen + br]
                    count_num += 1
        if count_num / (count_mask + count_num) > percent * 0.01:
            count += 1
            std_list.append(data_sum * (count_mask + count_num) / count_num)
    mean, mead, std = sigma_clipped_stats(np.array(std_list))
    return std


def clean_header_string(s):
    '''
    Clean a string from the header.

    Parameters
    ----------
    s : string
        The header content.

    Returns
    -------
    s : string
        The string without comments and space.
    '''
    if s is None:
        return None

    s = s.split('/')[0]
    s = s.strip()
    return s


def cutout_star(image, segm, coord_pix, extract_size, sigma=1, plot=True):
    '''
    Extract the radial profile around the specified coordinate.

    Parameters
    ----------
    image : 2D array
        The image data.
    segm : SegmentationImage
        The input SegmentationImage.
    coord_pix : tuple
        Coordinate of the target star, units: pixel.
    extract_size : int
        Size of the box to extract the star, units: pixel.
    sigma : float (default: 1)
        The sigma of the Gaussian fit to measure the center of the star,
        units: pixel.
    plot : bool (default: False)
        Plot the results if True.

    Notes
    -----
    FIXME: Finishe the doc!
    '''
    segm = deepcopy(segm)
    l = segm.data[int(coord_pix[1]), int(coord_pix[0])]
    mask = segm.data > 0

    img_c = Cutout2D(image, position=coord_pix, size=extract_size)
    mask_c = Cutout2D(mask, position=coord_pix, size=extract_size)
    data = img_c.data
    mask = mask_c.data

    ny, nx = data.shape
    yy, xx = np.mgrid[:ny, :nx]

    amp = np.percentile(data, 99)
    mean_init = extract_size / 2
    g_init = models.Gaussian2D(amplitude=amp, x_mean=mean_init, y_mean=mean_init, x_stddev=sigma,
                               y_stddev=sigma)
    fitter = fitting.LevMarLSQFitter()

    with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter
        warnings.filterwarnings('ignore', message='Model is linear in parameters',
                                category=AstropyUserWarning)
        g_fit = fitter(g_init, xx, yy, data)

    x = coord_pix + (g_fit.x_mean.value - mean_init)
    y = g_fit.y_mean.value

    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(14, 7))

        ax = axs[0]
        norm = simple_norm(data, 'log', percent=99.0)
        ax.imshow(data, norm=norm, origin='lower', cmap='viridis')
        ax.plot(mean_init, mean_init, marker='+', color='C0')
        ax.plot(g_fit.x_mean, g_fit.y_mean, marker='x', color='C3')

        ax = axs[1]
        ax.imshow(mask, origin='lower', cmap='Greys_r')

    return x, y


def detect_source_extended(image: np.array, target_coord: tuple, target_mask: np.array,
                           threshold_o: float, threshold_i: float, npixels_o=5,
                           npixels_i=5, nlevels_o=32, nlevel_i=256, contrast_o=0.001,
                           contrast_i=1e-6, coverage_mask=None, connectivity=8,
                           kernel_fwhm=0, mode='linear', nproc=1,
                           progress_bar=False, plot=False, fig=None, axs=None,
                           norm_kwargs=None, interactive=False, verbose=False):
    '''
    Detect the image sources for an extended target. This function get
    the segmentations of the image in two steps, one inside the target_mask and
    one outside the target_mask.

    Parameters
    ----------
    image : 2D array
        The image data.
    target_coord : tuple
        The pixel coordinate of the target.
    target_mask (optional) : 2D bool array
        A boolean mask, with the same shape as the input data, where True
        values indicate masked pixels. For extended targets, we generate two
        segmentations with different parameters, one inside the target_mask and
        one outside the target_mask.
    threshold_o : float
        Threshold to generate the segmentation outside target mask.
    threshold_i : float
        Threshold to generate the segmentation inside target mask.
    npixels_o : int (default: 5)
        The minimum number of connected pixels, each greater than threshold,
        that an object must have to be detected. npixels must be a positive
        integer. It is for the outer segmentation.
    npixels_i : int (default: 5)
        The npixel for the inner segmentation.
    nlevels_o : int (default: 32)
        The number of multi-thresholding levels to use for deblending. Each
        source will be re-thresholded at nlevels levels spaced between its
        minimum and maximum values (non-inclusive). The mode keyword determines
        how the levels are spaced. It is for the outer segmentation.
    nlevels_i : int (default: 32)
        The nlevel for the inner segmentation.
    contrast_o : float (default: 0.001)
        The fraction of the total source flux that a local peak must have (at
        any one of the multi-thresholds) to be deblended as a separate object.
        contrast must be between 0 and 1, inclusive. If contrast=0 then every
        local peak will be made a separate object (maximum deblending).
        If contrast=1 then no deblending will occur. The default is 0.001, which
        will deblend sources with a 7.5 magnitude difference. It is for the
        outer segmentation.
    contrast_i : float (default: 1e-6)
        The contrast for the inner segmentation.
    connectivity : {4, 8} optional
        The type of pixel connectivity used in determining how pixels are
        grouped into a detected source. The options are 4 or 8 (default).
        4-connected pixels touch along their edges. 8-connected pixels touch
        along their edges or corners.
    kernel_fwhm : float (default: 0)
        The kernel FWHM to smooth the image. If kernel_fwhm=0, skip the convolution.
    mode : {'exponential', 'linear', 'sinh'}, optional
        The mode used in defining the spacing between the multi-thresholding
        levels (see the nlevels keyword) during deblending. The 'exponential'
        and 'sinh' modes have more threshold levels near the source minimum and
        less near the source maximum. The 'linear' mode evenly spaces
        the threshold levels between the source minimum and maximum.
        The 'exponential' and 'sinh' modes differ in that the 'exponential'
        levels are dependent on the source maximum/minimum ratio (smaller ratios
        are more linear; larger ratios are more exponential), while the 'sinh'
        levels are not. Also, the 'exponential' mode will be changed to 'linear'
        for sources with non-positive minimum data values.
    nproc : int (default: 1)
        The number of processes to use for multiprocessing (if larger than 1).
        If set to 1, then a serial implementation is used instead of a parallel
        one. If None, then the number of processes will be set to the number of
        CPUs detected on the machine. Please note that due to overheads,
        multiprocessing may be slower than serial processing. This is especially
        true if one only has a small number of sources to deblend. The benefits
        of multiprocessing require ~1000 or more sources to deblend, with larger
        gains as the number of sources increase.
    progress_bar : bool (default: False)
        Show the progress bar in various steps.
    plot : bool (default: False)
        Plot the data and segmentation map if True.
    fig : Matplotlib Figure
        The figure to plot.
    axs : Matplotlib Axes
        The axes to plot. Two panels are needed.
    norm_kwargs (optional) : dict
        The keywords to normalize the data image.
    interactive : bool (default: False)
        Use the interactive plot if True.
    verbose : bool (default: True)
        Show details if True.

    Notes
    -----
    [SGJY added]
    '''
    # Get the segments outside the target mask
    if coverage_mask is None:
        mask = target_mask
    else:
        mask = target_mask | coverage_mask

    segm_o, _ = get_image_segmentation(image, threshold_o, npixels=npixels_o,
                                       mask=mask, kernel_fwhm=kernel_fwhm,
                                       nlevels=nlevels_o, contrast=contrast_o,
                                       connectivity=connectivity, mode=mode,
                                       nproc=nproc, deblend=True,
                                       progress_bar=progress_bar)

    # Get the segments inside the target mask
    if coverage_mask is None:
        mask = ~target_mask
    else:
        mask = ~target_mask | coverage_mask

    segm_i0 = detect_sources(image, threshold_i, npixels=npixels_i, mask=mask)
    debl_i = deblend_sources(image, segm_i0, npixels=npixels_i, labels=None,
                             nlevels=nlevel_i, contrast=contrast_i, mode=mode,
                             connectivity=connectivity, relabel=True,
                             nproc=nproc, progress_bar=progress_bar)

    # Get the mask of the galaxy innter region
    x_t, y_t = target_coord
    target_mask_i = segm_i0.data == segm_i0.data[int(y_t), int(x_t)]
    # Final segments; merged the segments inside target_mask_i.
    segm_i = segment_add(debl_i, target_mask_i)

    if plot:
        if interactive:
            ipy = get_ipython()
            ipy.run_line_magic('matplotlib', 'tk')

            def on_close(event):
                ipy.run_line_magic('matplotlib', 'inline')

        if axs is None:
            fig, axs = plt.subplots(1, 3, figsize=(21, 7), sharex=True, sharey=True)
            fig.subplots_adjust(hspace=0.07, wspace=0.05)
        else:
            assert fig is not None, 'Please provide fig together with axs!'

        if interactive:
            fig.canvas.mpl_connect('close_event', on_close)

        if norm_kwargs is None:
            norm_kwargs = dict(percent=99.99, stretch='asinh', asinh_a=0.001)
        norm = simple_norm(image, **norm_kwargs)

        ax = axs[0]
        ax.imshow(image, origin='lower', cmap='Greys_r', norm=norm)
        ax.plot(target_coord[0], target_coord[1], marker='+', ms=10, color='red')
        xlim = ax.get_xlim();
        ylim = ax.get_ylim()
        plot_mask_contours(target_mask, ax=ax, verbose=verbose, color='cyan', lw=0.5)
        plot_mask_contours(target_mask_i, ax=ax, verbose=verbose, color='magenta', lw=0.5)

        ax.set_xlim(xlim);
        ax.set_ylim(ylim)
        ax.set_title('Image', fontsize=18)

        ax = axs[1]
        ax.imshow(segm_i, origin='lower', cmap=segm_i.cmap, interpolation='nearest')
        xlim = ax.get_xlim();
        ylim = ax.get_ylim()
        ax.set_xlim(xlim);
        ax.set_ylim(ylim)
        ax.set_title('Inner segmentation', fontsize=18)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax = axs[2]
        ax.imshow(segm_o, origin='lower', cmap=segm_o.cmap, interpolation='nearest')
        xlim = ax.get_xlim();
        ylim = ax.get_ylim()
        ax.set_xlim(xlim);
        ax.set_ylim(ylim)
        ax.set_title('Outer segmentation', fontsize=18)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    detection_results = {
        'segment_out': segm_o,
        'segment_in': segm_i,
        'mask_out': target_mask,
        'mask_in': target_mask_i,
    }
    return detection_results


def extract_fix_isophotes(image, xcen, ycen, initsma, eps, pa, step=1,
                          linear_growth=False, minsma=None, maxsma=None):
    """
    Function to extract surface brightness profile with fixed center, ellipticity, and position angle.
    """

    minsma = minsma if minsma is not None else 0.5
    maxsma = maxsma if maxsma is not None else max(np.shape(image)) / 2 * 1.3
    isophote_list = []

    geometry = EllipseGeometry(xcen, ycen, initsma, eps, pa, astep=step, linear_growth=linear_growth,
                               fix_center=True, fix_pa=True, fix_eps=True)

    sma = initsma
    while True:
        sample = EllipseSample(image, sma, geometry=geometry)

        sample.update(geometry.fix)
        isophote = Isophote(sample, 0, True, stop_code=4)
        isophote_list.append(isophote)
        sma = isophote.sample.geometry.update_sma(step)
        if maxsma and sma >= maxsma:
            break

    first_isophote = isophote_list[0]
    sma, step = first_isophote.sample.geometry.reset_sma(step)

    while True:
        sample = EllipseSample(image, sma, geometry=geometry)

        sample.update(geometry.fix)
        isophote = Isophote(sample, 0, True, stop_code=4)
        isophote_list.append(isophote)
        sma = isophote.sample.geometry.update_sma(step)
        if minsma and sma <= max(minsma, 0.5):
            break

    isophote_list.sort()
    iso_fix = IsophoteList(isophote_list)

    return iso_fix

def extract_fix_isophotes_ly(image=None, xcen=None, ycen=None, initsma=None, eps=None, pa=None, step=None, 
                          linear_growth=False, minsma=None, maxsma=None, silent=False):
    """
    Function to extract surface brightness profile with fixed center, ellipticity, and position angle.
    """
    syntax = "syntax: results = extract_fix_isophotes(image=, xcen=, ycen=, initsma=, eps=, pa=, step=, linear_growth=False/True, minsma=None, maxsma=None, silent=False/True; minsma maxsma are optional)"
    if (None in [xcen, ycen, initsma, eps, pa, step]) or (image is None):
        print(syntax)
        return []
    print(syntax) if silent == False else print("")
    
    minsma = minsma if minsma is not None else 0.5
    maxsma = maxsma if maxsma is not None else max(np.shape(image))/2*1.3
    isophote_list = []

    geometry = EllipseGeometry(xcen, ycen, initsma, eps, pa, astep=step, linear_growth=False, 
                               fix_center=True, fix_pa=True, fix_eps=True)
    
    sma = initsma
    while True:
        sample = EllipseSample(image, sma, geometry=geometry)

        sample.update(geometry.fix)
        isophote = Isophote(sample, 0, True, stop_code=4)
        isophote_list.append(isophote)
        sma = isophote.sample.geometry.update_sma(step)
        if maxsma and sma >= maxsma:
            break

    first_isophote = isophote_list[0]
    sma, step = first_isophote.sample.geometry.reset_sma(step)

    while True:
        sample = EllipseSample(image, sma, geometry=geometry)

        sample.update(geometry.fix)
        isophote = Isophote(sample, 0, True, stop_code=4)
        isophote_list.append(isophote)
        sma = isophote.sample.geometry.update_sma(step)
        if minsma and sma <= max(minsma, 0.5):
            break

    isophote_list.sort()
    iso_fix = IsophoteList(isophote_list)
    
    return iso_fix
def fit_ellipse(x, y):
    """

    Fit the coefficients a,b,c,d,e,f, representing an ellipse described by
    the formula F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f = 0 to the provided
    arrays of data points x=[x1, x2, ..., xn] and y=[y1, y2, ..., yn].

    Based on the algorithm of Halir and Flusser, "Numerically stable direct
    least squares fitting of ellipses'.


    """

    D1 = np.vstack([x ** 2, x * y, y ** 2]).T
    D2 = np.vstack([x, y, np.ones(len(x))]).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
    M = np.linalg.inv(C) @ M
    eigval, eigvec = np.linalg.eig(M)
    con = 4 * eigvec[0] * eigvec[2] - eigvec[1] ** 2
    ak = eigvec[:, np.nonzero(con > 0)[0]]
    return np.concatenate((ak, T @ ak)).ravel()


def cart_to_pol(coeffs):
    """

    Convert the cartesian conic coefficients, (a, b, c, d, e, f), to the
    ellipse parameters, where F(x, y) = ax^2 + bxy + cy^2 + dx + ey + f = 0.
    The returned parameters are x0, y0, ap, bp, e, phi, where (x0, y0) is the
    ellipse centre; (ap, bp) are the semi-major and semi-minor axes,
    respectively; e is the eccentricity; and phi is the rotation of the semi-
    major axis from the x-axis.

    """

    # We use the formulas from https://mathworld.wolfram.com/Ellipse.html
    # which assumes a cartesian form ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0.
    # Therefore, rename and scale b, d and f appropriately.
    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2
    g = coeffs[5]

    den = b ** 2 - a * c
    if den > 0:
        raise ValueError('coeffs do not represent an ellipse: b^2 - 4ac must'
                         ' be negative!')

    # The location of the ellipse centre.
    x0, y0 = (c * d - b * f) / den, (a * f - b * d) / den

    num = 2 * (a * f ** 2 + c * d ** 2 + g * b ** 2 - 2 * b * d * f - a * c * g)
    fac = np.sqrt((a - c) ** 2 + 4 * b ** 2)
    # The semi-major and semi-minor axis lengths (these are not sorted).
    ap = np.sqrt(num / den / (fac - a - c))
    bp = np.sqrt(num / den / (-fac - a - c))

    # Sort the semi-major and semi-minor axis lengths but keep track of
    # the original relative magnitudes of width and height.
    width_gt_height = True
    if ap < bp:
        width_gt_height = False
        ap, bp = bp, ap

    # The eccentricity.
    r = (bp / ap) ** 2
    if r > 1:
        r = 1 / r
    e = np.sqrt(1 - r)

    # The angle of anticlockwise rotation of the major-axis from x-axis.
    if b == 0:
        phi = 0 if a < c else np.pi / 2
    else:
        phi = np.arctan((2. * b) / (a - c)) / 2
        if a > c:
            phi += np.pi / 2
    if not width_gt_height:
        # Ensure that phi is the angle to rotate to the semi-major axis.
        phi += np.pi / 2
    phi = phi % np.pi

    return x0, y0, ap, bp, e, phi


def gen_aperture_ref(image, threshold, coord_pix, mask=None, plot=False,
                     axs=None, **segm_kwargs):
    '''
    Generate the reference ellipse parameters for the aperture. The method
    follows Clark et al. (2017).

    Parameters
    ----------

    Notes
    -----
    FIXME: doc
    '''
    segm, cimg = get_image_segmentation(
        image, threshold=threshold, npixels=12, mask=mask, plot=False, **segm_kwargs)

    x, y = coord_pix
    mask = segm == segm.data[int(y), int(x)]
    poly = get_mask_polygons(mask)[0]
    p = shape(poly)
    x, y = p.convex_hull.exterior.coords.xy
    coeffs = fit_ellipse(np.array(x), np.array(y))
    x0, y0, ap, bp, e, phi = cart_to_pol(coeffs)
    aper = EllipticalAperture((x0, y0), ap, bp, phi)

    if plot:
        if axs is None:
            fig, axs = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
            fig.subplots_adjust(wspace=0.03)

        ax = axs[0]
        ax.imshow(segm, origin='lower', cmap=segm.cmap, interpolation='nearest')
        ax.plot(x, y, ls='none', marker='.', ms=5, color='red')

        ax = axs[1]
        ax.imshow(mask, origin='lower', cmap='Greys_r')
        ax.plot(x, y, ls='none', marker='.', ms=5, color='red')

        aper.plot(ax=ax, color='cyan', zorder=3, lw=1.5)
    return aper


def get_ellipse_pts(params, npts=100, tmin=0, tmax=2 * np.pi):
    """
    Return npts points on the ellipse described by the params = x0, y0, ap,
    bp, e, phi for values of the parametric variable t between tmin and tmax.

    """

    x0, y0, ap, bp, e, phi = params
    # A grid of the parametric variable, t.
    t = np.linspace(tmin, tmax, npts)
    x = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
    y = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
    return x, y


def find_aperture_bounds(image, aper_ref, mask=None, naper=10, threshold_snr=2,
                         fracs=[0.5, 3], plot=False, axs=None):
    '''
    Find the bounds of the aperture.

    Parameters
    ----------

    Notes
    -----
    FIXME: doc
    '''
    fmin, fmax = fracs
    aList = np.linspace(fmin * aper_ref.a, fmax * aper_ref.a, naper)
    bList = aList * aper_ref.b / aper_ref.a

    aperList = [EllipticalAperture(aper_ref.positions, a, b, aper_ref.theta) for (a, b) in zip(aList, bList)]
    annuList = [EllipticalAnnulus(aper_ref.positions, a_in=aList[loop], a_out=aList[loop + 1],
                                  b_out=bList[loop + 1], theta=aper_ref.theta) for loop in range(naper - 1)]

    intens = []
    intens_rms = []
    area = []
    for annu in annuList:
        stats = ApertureStats(image, annu, mask=mask, sum_method='center')
        intens.append(stats.mean)
        intens_rms.append(stats.mad_std)
        area.append(stats.sum_aper_area.value)

    intens = np.array(intens)
    intens_rms = np.array(intens_rms) / np.sqrt(area)
    snr = intens / intens_rms

    search_idx = np.where((snr - threshold_snr) < 0)[0]

    if len(search_idx > 0):
        idx = search_idx[0]
        a_in = annuList[idx - 1].a_in
        a_out = annuList[idx].a_out
    else:
        a_in = np.nan
        a_out = np.nan

    if plot:
        if axs is None:
            fig, axs = plt.subplots(1, 2, figsize=(14, 7))

        ax = axs[0]
        norm = simple_norm(image, stretch='asinh', percent=99.99, asinh_a=0.01)
        ax.imshow(image, origin='lower', cmap='Greys_r', norm=norm)

        if mask is not None:
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            plot_mask_contours(mask, ax=ax, color='w', ls='-', lw=0.5)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        for ap in aperList:
            ap.plot(ax=ax, color='C2', ls=':', lw=1.5)

        ax.minorticks_on()
        ax.set_title('Image', fontsize=18)

        ax = axs[1]
        sma = (aList[:-1] + aList[1:]) / 2
        ax.plot(sma, snr, marker='o', color='C3')
        ax.axvspan(xmin=a_in, xmax=a_out, ls='--', edgecolor='gray', facecolor='none', hatch='/')
        ax.axhline(y=threshold_snr, ls='--', color='gray')
        ax.set_xlabel('Semimajor axis (pix)', fontsize=24)
        ax.set_ylabel('SNR', fontsize=24, color='C3')
        ax.minorticks_on()

        axr = ax.twinx()
        axr.plot(sma, intens, marker='o', color='C0')
        axr.set_ylabel('Intensity', fontsize=24, color='C0')
        axr.minorticks_on()

    return a_in, a_out


def gen_aperture_ellipse(image, coord_pix, threshold_segm, psf_fwhm, threshold_snr=2,
                         mask=None, grid_num=8, grid_mask=None,  naper=20, fracs=[1, 5], plot=False, axs=None,
                         **segm_kwargs):
    '''
    Generate elliptical aperture.

    Parameters
    ----------
    image : 2D array
        The image.
    coord_pix : tuple
        Coordinate of the target star, units: pixel.
    threshold_segm : float
        The threshold to detect the segmentation of the target.
    psf_fwhm : float
        The FWHM of the PSF in the units of pixel.
    threshold_snr : float (default: 2)
        The threshold to SNR to determine the aperture.
    mask (optional) : 2D bool array
        A boolean mask, with the same shape as the input data, where True
        values indicate masked pixels. Masked pixels will not be included in
        any source.
    grid_num: int (default:8)
        The number of grids in a row or column of an image, used to estimate large-scale rms.
    grid_mask: 2D bool array
        A boolean mask, with the same shape as the input data, where True
        values indicate masked pixels. Masked pixels will not be included in
        any source.
    naper : int (default: 20)
        The number of apertures sampled in the first step search.
    fracs : list (default: [1, 5])
        The minimum and maximum aperture size as the factors of the reference
        aperture.
    plot : bool (default: False)
        Plot the image and the aperture searching results if True.
    axs : Matplotlib Axes
        The axes to plot. Four panels (2x2) are needed.

    Notes
    -----
    FIXME: doc
    '''
    if plot:
        if axs is None:
            fig, axs = plt.subplots(2, 2, figsize=(14, 14))

    aper_ref = gen_aperture_ref(
        image, threshold=threshold_segm, coord_pix=coord_pix, mask=mask,
        plot=plot, axs=axs[0, :], **segm_kwargs)
    a_in, a_out = find_aperture_bounds(
        image, aper_ref, mask=mask, naper=naper, threshold_snr=threshold_snr, fracs=fracs,
        plot=plot, axs=axs[1, :])

    if ~np.isnan(a_in):
        aList = np.arange(a_in, a_out, psf_fwhm)
        bList = aList * aper_ref.b / aper_ref.a
        annuList = [EllipticalAnnulus(aper_ref.positions, a_in=aList[loop], a_out=aList[loop + 1],
                                      b_out=bList[loop + 1], theta=aper_ref.theta)
                    for loop in range(len(aList) - 1)]

        intens = []
        intens_rms = []
        area = []
        for annu in annuList:
            stats = ApertureStats(image, annu, mask=mask, sum_method='center')
            intens.append(stats.mean)
            intens_rms.append(stats.mad_std)
            area.append(stats.sum_aper_area.value)

        sep_num = grid_num
        x1 = np.shape(image)[0]//sep_num
        x2 = np.shape(image)[1]//sep_num
        mea_box = []
        for row in range(math.floor(sep_num)):
            for col in range(math.floor(sep_num)):
                ma_data = np.ma.array(image[row*x1:(row+1)*x1, col*x2:(col+1)*x2],
                                      mask=grid_mask[row*x1:(row+1)*x1, col*x2:(col+1)*x2])
                pix_num = x1*x2 - grid_mask[row*x1:(row+1)*x1, col*x2:(col+1)*x2].sum()
                if isinstance(ma_data.sum()/pix_num, float) is False:
                    continue
                mea_box.append(ma_data.sum()/pix_num)
        macro_rms = np.std(mea_box)

        intens = np.array(intens)
        intens_rms = np.sqrt((np.array(intens_rms) / np.sqrt(area))**2 + macro_rms**2)
        snr = intens / intens_rms
        snr_sub = (snr - threshold_snr)

        sma = (aList[:-1] + aList[1:]) / 2

        snr_func = interp1d(sma, snr_sub ** 2)
        res = differential_evolution(snr_func, bounds=[(sma.min(), sma.max())])
        ap_r = res.x[0]

        bp_r = ap_r * aper_ref.b / aper_ref.a
        aper = EllipticalAperture(aper_ref.positions, ap_r, bp_r, aper_ref.theta)
    else:
        aper = None

    if plot & (aper is not None):
        ax = axs[1, 0]
        aper.plot(ax=ax, color='C2', lw=2)

        ax = axs[1, 1]
        ax.plot(sma, snr, color='C1', label='Refined SNR')
        ax.axvline(ap_r, color='C2', ls='--', lw=2, label='Refined SMA')
        ax.legend(loc='upper right', fontsize=16)

    return aper


def gen_random_apertures(img, nsample, mask_aper, percent, mask=None,
                         plot=False, axs=None, norm_kwargs=None, **ellipse_kwargs):
    mask_cen = mask_aper.positions
    mask_sma = mask_aper.a
    mask_smb = mask_aper.b
    mask_pa = mask_aper.theta
    counter = 0
    aper_list = []
    shape_x = np.shape(img)[0]
    shape_y = np.shape(img)[1]
    factor = math.sqrt(percent)
    sma = factor * mask_sma
    smb = factor * mask_smb
    pa = mask_pa
    area = np.pi * sma * smb
    mask_master_aper = add_mask_ellipse(np.zeros(np.shape(img), dtype=bool),
                                        mask_cen[0], mask_cen[1], mask_sma, mask_smb,
                                        mask_pa)
    if mask is not None:
        total_mask = mask | mask_master_aper
    else:
        total_mask = mask_master_aper
    aperture_map = np.zeros(np.shape(img), dtype='float')
    while counter <= nsample - 1:
        x_c = random.uniform(max(sma, smb) + 1, shape_x - max(sma, smb) - 1)
        y_c = random.uniform(max(sma, smb) + 1, shape_y - max(sma, smb) - 1)
        if total_mask[int(y_c), int(x_c)] == 1.:
            counter += 1
            continue
        aper_temp = EllipticalAperture((x_c, y_c), sma, smb, pa)
        phot_temp0 = aperture_photometry(total_mask, aper_temp)
        if phot_temp0['aperture_sum'][0] != 0.:
            counter += 1
            continue
        phot_temp1 = aperture_photometry(aperture_map, aper_temp)
        if phot_temp1['aperture_sum'][0] == 0.:
            aperture_map[int(y_c), int(x_c)] = 1
            aper_list.append(aper_temp)
            counter += 1
        else:
            counter += 1

    if plot:
        if axs is None:
            fig, axs = plt.subplots(1, 2, figsize=(14, 7))
            fig.subplots_adjust(wspace=0.25)
            axs = axs.ravel()

        if norm_kwargs is None:
            norm_kwargs = dict(percent=99.99, stretch='asinh', asinh_a=0.0001)
        norm = simple_norm(img, **norm_kwargs)

        ax0 = axs[0]
        ax0.imshow(img, origin='lower', cmap='Greys_r', norm=norm)

        aper_init = EllipticalAperture(mask_cen, mask_sma, mask_smb, mask_pa)
        aper_init.plot(ax=ax0, color='yellow', lw=1.5, label='Photometry aperture')
        ax0.legend(loc='upper left', fontsize=12)

        ax1 = axs[1]
        ax1.imshow(img, origin='lower', cmap='Greys_r', norm=norm)
        aper_init.plot(ax=ax1, color='yellow', lw=1.5, label='Photometry aperture')
        for i in range(len(aper_list)):
            aper = aper_list[i]
            aper.plot(ax=ax1, color='red', lw=1, label='Random apertures')
            if i == 0:
                ax1.legend(loc='upper left', fontsize=12)

    return aper_list


def gen_image_mask(image, threshold, npixels=5, mask=None, connectivity=8,
                   kernel_fwhm=0, deblend=False, nlevels=32, contrast=0.001,
                   mode='linear', nproc=1, progress_bar=False, expand_factor=1.2,
                   bounds: list = None, choose_coord=None, plot=False, fig=None,
                   axs=None, norm_kwargs=None, interactive=False, verbose=True):
    '''
    Generate the mask in a specified region.

    Parameters
    ----------
    image : 2D array
        The image data.
    threshold : float
        Threshold of image segmentation.
    npixels : int (default: 5)
        The minimum number of connected pixels, each greater than threshold,
        that an object must have to be detected. npixels must be a positive
        integer.
    mask (optional) : 2D bool array
        A boolean mask, with the same shape as the input data, where True
        values indicate masked pixels. Masked pixels will not be included in
        any source.
    connectivity : {4, 8} optional
        The type of pixel connectivity used in determining how pixels are
        grouped into a detected source. The options are 4 or 8 (default).
        4-connected pixels touch along their edges. 8-connected pixels touch
        along their edges or corners.
    kernel_fwhm : float (default: 0)
        The kernel FWHM to smooth the image. If kernel_fwhm=0, skip the convolution.
    expand_factor : float (default: 1.2)
        Expand the mask by this factor.
    bounds : list
        The bounds of the box to make the mask, (xmin, xmax, ymin, ymax).
    choose_coord (optional) : tuple of x and y
        The pixel coordinate of the target to generate the mask.
        If provided, the function will only generate the mask for the segment
        containing the input pixel.
    plot : bool (default: False)
        Plot the data and segmentation map if True.
    fig : Matplotlib Figure
        The figure to plot.
    axs : Matplotlib Axes
        The axes to plot. Two panels are needed.
    norm_kwargs (optional) : dict
        The keywords to normalize the data image.
    interactive : bool (default: False)
        Use the interactive plot if True.
    verbose : bool (default: True)
        Show details if True.

    Notes
    -----
    [SGJY added]
    '''
    if bounds is None:
        img = image
    else:
        xmin, xmax, ymin, ymax = bounds
        slice_x = slice(xmin, xmax)
        slice_y = slice(ymin, ymax)
        img = image[slice_y, slice_x]

        if mask is not None:
            mask = mask[slice_y, slice_x]
    
    smap, cdata = get_image_segmentation(img, threshold=threshold,
                                         npixels=npixels, mask=mask,
                                         connectivity=connectivity,
                                         kernel_fwhm=kernel_fwhm,
                                         deblend=deblend, nlevels=nlevels,
                                         contrast=contrast, mode=mode,
                                         nproc=nproc, progress_bar=progress_bar,
                                         plot=False)

    if choose_coord is None:
        mask = smap.data

    else:
        x, y = choose_coord

        if bounds is None:
            mask = smap.data == smap.data[int(y), int(x)]
        else:
            mask = smap.data == smap.data[int(y) - ymin, int(x) - xmin]

    mask_e = scale_mask(mask, factor=expand_factor, connectivity=connectivity)

    if bounds is None:
        mask = mask_e
    else:
        mask = np.zeros_like(image, dtype=bool)
        mask[slice_y, slice_x] = mask_e

    if plot:
        if interactive:
            ipy = get_ipython()
            ipy.run_line_magic('matplotlib', 'tk')

            def on_close(event):
                ipy.run_line_magic('matplotlib', 'inline')

        if axs is None:
            fig, axs = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
            fig.subplots_adjust(wspace=0.05)
        else:
            assert fig is not None, 'Please provide fig together with axs!'

        if interactive:
            fig.canvas.mpl_connect('close_event', on_close)

        if norm_kwargs is None:
            norm_kwargs = dict(percent=99.99, stretch='asinh', asinh_a=0.001)
        norm = simple_norm(image, **norm_kwargs)

        ax = axs[0]
        ax.imshow(image, origin='lower', cmap='Greys_r', norm=norm)
        xlim = ax.get_xlim();
        ylim = ax.get_ylim()
        plot_mask_contours(mask, ax=ax, verbose=verbose, color='cyan', lw=0.5)

        if bounds is not None:
            x = [bounds[0], bounds[0], bounds[1], bounds[1], bounds[0]]
            y = [bounds[2], bounds[3], bounds[3], bounds[2], bounds[2]]
            ax.plot(x, y, ls='--', color='red')

        ax.set_xlim(xlim);
        ax.set_ylim(ylim)
        ax.set_title('Image', fontsize=18)

        ax = axs[1]
        ax.imshow(smap, origin='lower', cmap=smap.cmap, interpolation='nearest')
        xlim = ax.get_xlim();
        ylim = ax.get_ylim()
        plot_mask_contours(mask_e, ax=ax, verbose=verbose, color='cyan', lw=0.5)
        ax.set_xlim(xlim);
        ax.set_ylim(ylim)
        ax.set_title('Segmentation map', fontsize=18)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    return mask, smap, cdata


def gen_images_matched(atlas, psf_fwhm: float, image_size: float,
                       pixel_scale: float = None, progress_bar=False,
                       verbose=False):
    '''
    Generate the matched images.

    Parameters
    ----------
    atlas : Atlas
        The Atlas object.
    psf_fwhm : float
        The PSF FWHM, units: arcsec
    image_size : float
        The size of the output image, units: arcsec
    pixel_scale (optional) : float
        The output pixel scale, units: arcsec. If not provided, half of
        the psf_fwhm will be used to ensure the Nyquist sampling.
    progress_bar : bool (default: False)
        The progress of the processed images.
    verbose : bool (default: False)
        Print details if True.

    Returns
    -------
    images : list
        List of matched images.
    output_wcs : WCS
        The WCS of the output images.

    Notes
    -----
    FIXME: The PSF matching is simplied for now by just using a Gaussian
           convolution. We can implement more rigorous matching method later.
    '''
    if pixel_scale is None:
        pixel_scale = psf_fwhm / 2.  # Nyquist sampling

    image_size_pix = np.ceil(image_size / pixel_scale).astype(int)

    output_wcs = WCS(naxis=2)
    output_wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    output_wcs.wcs.crval = [atlas._ra_deg, atlas._dec_deg]
    output_wcs.wcs.crpix = image_size_pix // 2, image_size_pix // 2
    output_wcs.wcs.cdelt = -1 * pixel_scale / 3600, pixel_scale / 3600
    shape_out = (image_size_pix, image_size_pix)
    output_wcs.pixel_shape = shape_out

    images = []

    if progress_bar:
        imGen = tqdm.tqdm(atlas)
    else:
        imGen = atlas._image_list

    for img in imGen:
        data_clean = img.fetch_temp_image('data_clean')
        assert data_clean is not None, f'[gen_images_matched]: Generate the cleaned image of {img} first!'

        if psf_fwhm > img._psf_fwhm:
            fwhm = np.sqrt(psf_fwhm ** 2 - img._psf_fwhm ** 2)
            sigma = fwhm / img._pxs * gaussian_fwhm_to_sigma
            data_conv = gaussian_filter(data_clean, sigma=sigma)
        else:
            if verbose:
                print(f'[gen_images_matched]: Skip convolution of {img} (pixel scale: {img._psf_fwhm}")!')
            data_conv = data_clean

        data_rebin, _ = reproject_adaptive((data_conv, img._wcs), output_wcs,
                                           shape_out=shape_out, kernel='gaussian',
                                           conserve_flux=True,
                                           boundary_mode='ignore')
        images.append(data_rebin)
        del data_clean, data_conv
    return images, output_wcs


def gen_variance_matched(atlas, psf_fwhm: float, image_size: float,
                         pixel_scale: float = None, progress_bar=False,
                         verbose=False):
    '''
    Generate the matched variance images.

    Parameters
    ----------
    atlas : Atlas
        The Atlas object.
    psf_fwhm : float
        The PSF FWHM, units: arcsec
    image_size : float
        The size of the output image, units: arcsec
    pixel_scale (optional) : float
        The output pixel scale, units: arcsec. If not provided, half of
        the psf_fwhm will be used to ensure the Nyquist sampling.
    progress_bar : bool (default: False)
        The progress of the processed images.
    verbose : bool (default: False)
        Print details if True.

    Returns
    -------
    images : list
        List of matched images.
    output_wcs : WCS
        The WCS of the output images.

    Notes
    -----
    FIXME: The PSF matching is simplied for now by just using a Gaussian
           convolution. We can implement more rigorous matching method later.
    '''
    if pixel_scale is None:
        pixel_scale = psf_fwhm / 2.  # Nyquist sampling

    image_size_pix = np.ceil(image_size / pixel_scale).astype(int)

    output_wcs = WCS(naxis=2)
    output_wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    output_wcs.wcs.crval = [atlas._ra_deg, atlas._dec_deg]
    output_wcs.wcs.crpix = image_size_pix // 2, image_size_pix // 2
    output_wcs.wcs.cdelt = -1 * pixel_scale / 3600, pixel_scale / 3600
    shape_out = (image_size_pix, image_size_pix)
    output_wcs.pixel_shape = shape_out

    varmaps = []

    if progress_bar:
        imGen = tqdm.tqdm(atlas)
    else:
        imGen = atlas._image_list

    for img in imGen:
        data_variance = img.fetch_temp_image('data_variance')
        assert data_variance is not None, '[gen_variance_matched]: Please generate the variance map!'

        if psf_fwhm > img._psf_fwhm:
            fwhm = np.sqrt(psf_fwhm ** 2 - img._psf_fwhm ** 2)
            sigma = fwhm / img._pxs * gaussian_fwhm_to_sigma
            data_conv = gaussian_filter(data_variance, sigma=sigma)
        else:
            if verbose:
                print(f'[gen_variance_matched]: Skip convolution of {img} (pixel scale: {img._psf_fwhm}")!')
            data_conv = data_variance

        data_rebin, _ = reproject_adaptive((data_conv, img._wcs), output_wcs,
                                           shape_out=shape_out, kernel='gaussian',
                                           conserve_flux=True,
                                           boundary_mode='ignore')

        varmaps.append(data_rebin)
        del data_variance, data_conv
    return varmaps, output_wcs


def get_mask_polygons(mask, connectivity=8):
    '''
    Get polygons from the mask.

    Parameters
    ----------
    mask : 2D array
        The mask with masked in region True or 1.
    connectivity : {4, 8} (default: 8)
        The type of pixel connectivity used in determining how pixels are
        grouped into a detected source. The options are 4 or 8 (default).
        4-connected pixels touch along their edges. 8-connected pixels touch
        along their edges or corners.

    Returns
    -------
    pList : list
        List of dict with polygon information.

    Notes
    -----
    [SGJY added]
    '''
    polygons = np.array(list(shapes(mask.astype('int32'), connectivity=connectivity)))
    vals = polygons[:, 1]

    # Collect the polygons that are associated with the mask.
    pList = polygons[vals > 0, 0]
    return pList


def get_image_segmentation(data, threshold, npixels, mask=None, connectivity=8,
                           kernel_fwhm=0, deblend=False, nlevels=32, contrast=0.001,
                           mode='linear', nproc=1, progress_bar=True,
                           plot=False, axs=None, norm_kwargs=None,
                           interactive=False):
    '''
    Get the image segmentation.

    Parameters
    ----------
    data : 2d array
        The image data to be decomposed.
    threshold : float
        The threshold ot detect source with image segmentation.
    npixels : int
        The minimum number of connected pixels, each greater than threshold,
        that an object must have to be detected. npixels must be a positive
        integer.
    mask : 2D bool array
        A boolean mask, with the same shape as the input data, where True
        values indicate masked pixels. Masked pixels will not be included in
        any source.
    connectivity : {4, 8} optional
        The type of pixel connectivity used in determining how pixels are
        grouped into a detected source. The options are 4 or 8 (default).
        4-connected pixels touch along their edges. 8-connected pixels touch
        along their edges or corners.
    kernel_fwhm : float (default: 0)
        The kernel FWHM to smooth the image. If kernel_fwhm=0, skip the convolution.
    plot : bool (default: False)
        Plot the data and segmentation map if True.
    axs : matplotlib axes
        The axes to plot the data and segmentation map. Must be >=2 panels.
    norm_kwargs (optional) : dict
        The keywords to normalize the data image.
    interactive : bool (default: False)
        Use the interactive plot if True.

    Returns
    -------
    segment_map : Photutils Segmentation
        The segmentation.
    convolved_data : 2D array
        The kernel convolved image.

    Notes
    -----
    [SGJY added]
    '''
    if kernel_fwhm == 0:
        convolved_data = data
    elif kernel_fwhm > 0:
        convolved_data = gaussian_filter(data, kernel_fwhm * gaussian_fwhm_to_sigma)
    else:
        raise ValueError(f'The kernel_fwhm ({kernel_fwhm}) has to be >=0!')

    finder = SourceFinder(npixels=npixels, connectivity=connectivity,
                          deblend=deblend, nlevels=nlevels, contrast=contrast,
                          mode=mode, relabel=True, nproc=nproc,
                          progress_bar=progress_bar)
    segment_map = finder(convolved_data, threshold=threshold, mask=mask)

    if plot:
        if interactive:
            ipy = get_ipython()
            ipy.run_line_magic('matplotlib', 'tk')

            def on_close(event):
                ipy.run_line_magic('matplotlib', 'inline')

        if axs is None:
            fig, axs = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)

        if interactive:
            fig.canvas.mpl_connect('close_event', on_close)

        if norm_kwargs is None:
            norm_kwargs = dict(percent=99.99, stretch='asinh', asinh_a=0.001)

        norm = simple_norm(convolved_data, **norm_kwargs)

        ax = axs[0]
        ax.imshow(convolved_data, origin='lower', cmap='Greys_r', norm=norm)
        ax.set_title('Convolved data', fontsize=16)

        ax = axs[1]
        ax.imshow(segment_map, origin='lower', cmap=segment_map.cmap, interpolation='nearest')
        ax.set_title('Segmentation Image', fontsize=16)

    return segment_map, convolved_data


def get_masked_patch(data, mask, coord_pix, factor=1, plot=False, axs=None,
                     norm_kwargs=None):
    '''
    Get the patch of the image that fits the target mask.

    Parameters
    ----------
    data : 2D array
        The image data.
    mask : 2D bool array
        The target mask with only one connected masked region, True for masked region.
    coord_pix : 1D array (x, y)
        The pixel coordinate of the masked target.
    factor : float (default: 1)
        The factor to scale the image patch.
    plot : bool (default: False)
        Plot the results if True.
    axs : matplotlib axes
        The axes to make the plots. Use the first two.
    norm_kwargs (optional) : dict
        The keywords to normalize the data image.

    Returns
    -------
    data_s : 2D array
        The sliced data patch.
    bounds : list (xmin, xmax, ymin, ymax)
        The min and max coordinates of the patch in the original image.

    Notes
    -----
    [SGJY added]
    '''
    pList = get_mask_polygons(mask)
    poly = shape(pList[0])
    xy_poly = np.c_[poly.exterior.xy]

    coord_pix = np.array(coord_pix)
    r = np.sqrt(np.sum((xy_poly - coord_pix[np.newaxis, :]) ** 2, axis=1))

    a_box = r.max() * factor
    x1 = int(coord_pix[0] - a_box)
    x2 = int(coord_pix[0] + a_box)
    y1 = int(coord_pix[1] - a_box)
    y2 = int(coord_pix[1] + a_box)
    bounds = [x1, x2, y1, y2]

    slice_x = slice(x1, x2)
    slice_y = slice(y1, y2)
    data_s = data[slice_y, slice_x]

    if plot:
        if axs is None:
            fig, axs = plt.subplots(1, 2, figsize=(14, 7))

        if norm_kwargs is None:
            norm_kwargs = dict(percent=99.99, stretch='asinh', asinh_a=0.001)
        norm = simple_norm(data, **norm_kwargs)

        ax = axs[0]
        ax.imshow(data, origin='lower', cmap='Greys_r', norm=norm)
        ax.plot(xy_poly[:, 0], xy_poly[:, 1], color='cyan', lw=2)

        ax = axs[1]
        ax.imshow(data_s, origin='lower', cmap='Greys_r', norm=norm)
        ax.plot(xy_poly[:, 0] - x1, xy_poly[:, 1] - y1, color='cyan', lw=2)

    return data_s, bounds


def image_photometry(image, aperture, calibration_uncertainty=None,
                     mask=None, bkgsub=True, rannu_in=1.25, rannu_out=1.60,
                     error=True, nsample=300,
                     area_sample=(0.02, 0.04, 0.08, 0.16, 0.32),
                     plot=False, ax=None, norm_kwargs=None):
    '''
    Aperture photometry on one image.

    Parameters
    ----------
    image : 2D array
        The image data.
    aperture : EllipticalAperture
        The elliptical aperture to measure the image.
    mask (optional) : 2D array
        The mask of the contaminants.
    rannu_in, rannu_out : float (default: 1.25, 1.60)
        The inner and outer annulus semimajor axes, units: pixel. The values
        follow Clark et al. (2017).
    plot : bool (default: False)
        Plot the results.
    ax : Axis
        The axis to plot.
    norm_kwargs (optional) : dict
        The keywords to normalize the data image.

    Returns
    -------
    phot_bkgsub : float
        The background subtracted flux.
    '''
    annulus = EllipticalAnnulus(aperture.positions,
                                a_in=aperture.a * rannu_in,
                                a_out=aperture.a * rannu_out,
                                b_in=aperture.b * rannu_in,
                                b_out=aperture.b * rannu_out,
                                theta=aperture.theta)
    mask_nan = np.isnan(image)
    if mask is not None:
        mask = mask | mask_nan
    else:
        mask = mask_nan
    phot_table = aperture_photometry(image, aperture, mask=mask)
    aperture_area = aperture.area_overlap(image, mask=mask)

    if bkgsub:
        aperstats = ApertureStats(image, annulus)
        total_bkg = aperstats.mean * aperture_area
        phot_bkgsub = phot_table['aperture_sum'][0] - total_bkg
    else:
        phot_bkgsub = phot_table['aperture_sum'][0]

    if error:
        sigma_box = []
        if area_sample is not None:
            for i in range(len(area_sample)):
                sample_box = []
                aper_list = gen_random_apertures(image, nsample=nsample, mask_aper=aperture, mask=mask,
                                                 percent=area_sample[i])
                for j in range(len(aper_list)):
                    phot_table = aperture_photometry(image, aper_list[j])
                    sample_box.append(phot_table['aperture_sum'][0])
                sigma_t = np.std(np.array(sample_box))
                sigma_box.append(sigma_t)
            sigma = error_curve_fit(area_sample, sigma_box)
            if calibration_uncertainty is not None:
                sigma = math.sqrt(sigma ** 2 + (phot_bkgsub * calibration_uncertainty) ** 2)
        else:
            if calibration_uncertainty is not None:
                sigma = phot_bkgsub * calibration_uncertainty

    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))

        if norm_kwargs is None:
            norm_kwargs = dict(percent=99.99, stretch='asinh', asinh_a=0.0001)
        norm = simple_norm(image, **norm_kwargs)

        ax.imshow(image, origin='lower', cmap='Greys_r', norm=norm)

        aperture.plot(ax=ax, color='cyan', lw=1.5, label='Photometry')
        annulus.plot(ax=ax, color='cyan', ls='--', lw=1.0, label='Background')
        ax.legend(loc='upper right', fontsize=16)
        if error:
            if bkgsub:
                ax.text(0.95, 0.05, f'Phot: {phot_bkgsub:.2e}\nBkg: {total_bkg:.2e}\nSigma: {sigma:.2e}',
                        fontsize=14, transform=ax.transAxes, va='bottom', ha='right',
                        bbox=dict(alpha=0.8, color='w'))
            else:
                ax.text(0.95, 0.05, f'Phot: {phot_bkgsub:.2e}\nSigma: {sigma:.2e}',
                        fontsize=14, transform=ax.transAxes, va='bottom', ha='right',
                        bbox=dict(alpha=0.8, color='w'))
        else:
            if bkgsub:
                ax.text(0.95, 0.05, f'Phot: {phot_bkgsub:.2e}\nBkg: {total_bkg:.2e}',
                        fontsize=14, transform=ax.transAxes, va='bottom', ha='right',
                        bbox=dict(alpha=0.8, color='w'))
            else:
                ax.text(0.95, 0.05, f'Phot: {phot_bkgsub:.2e}',
                        fontsize=14, transform=ax.transAxes, va='bottom', ha='right',
                        bbox=dict(alpha=0.8, color='w'))
    if error:
        return phot_bkgsub, sigma
    else:
        return phot_bkgsub

def multi_apertures_photometry(image, apertures, calibration_uncertainty=None,
                     mask=None, bkgsub=True, rannu_in=1.25, rannu_out=1.60,
                     error=True, nsample=300,
                     area_sample=(0.02, 0.04, 0.08, 0.16, 0.32),
                     plot=False, ax=None, norm_kwargs=None):
    '''
    Multi-apertures photometry on one image.

    Parameters
    ----------
    image : 2D array
        The image data.
    apertures : a list of EllipticalAperture
        The elliptical apertures to measure the image. The first element should be master apertures
    mask (optional) : 2D array
        The mask of the contaminants.
    rannu_in, rannu_out : float (default: 1.25, 1.60)
        The inner and outer annulus semimajor axes, units: pixel. The values
        follow Clark et al. (2017).
    plot : bool (default: False)
        Plot the results.
    ax : Axis
        The axis to plot.
    norm_kwargs (optional) : dict
        The keywords to normalize the data image.

    Returns
    -------
    phot_bkgsub, sigma :array
        The flux and the error.
    '''
    aperture = apertures[0]

    annulus = EllipticalAnnulus(aperture.positions,
                                a_in=aperture.a * rannu_in,
                                a_out=aperture.a * rannu_out,
                                b_in=aperture.b * rannu_in,
                                b_out=aperture.b * rannu_out,
                                theta=aperture.theta)
    mask_nan = np.isnan(image)
    if mask is not None:
        mask = mask | mask_nan
    else:
        mask = mask_nan
    phot_tb = []
    for aper in apertures:
        phot_table = aperture_photometry(image, aper, mask=mask)
        phot_tb.append(phot_table['aperture_sum'][0])
    apertures_area = np.array([aper.area_overlap(image, mask=mask) for aper in apertures])

    if bkgsub:
        aperstats = ApertureStats(image, annulus)
        total_bkg = aperstats.mean * apertures_area
        phot_bkgsub = np.array([phot_tb[i] - total_bkg[i] for i in range(len(apertures))])
    else:
        phot_bkgsub = np.array([phot_tb[i] for i in range(len(apertures))])

    if error:
        sigma_box = []
        if area_sample is not None:
            for i in range(len(area_sample)):
                sample_box = []
                aper_list = gen_random_apertures(image, nsample=nsample, mask_aper=aperture, mask=mask,
                                                 percent=area_sample[i])
                for j in range(len(aper_list)):
                    phot_table = aperture_photometry(image, aper_list[j])
                    sample_box.append(phot_table['aperture_sum'][0])
                sigma_t = np.std(np.array(sample_box))
                sigma_box.append(sigma_t)
            sigma = np.array([error_curve_fit(area_sample, sigma_box, standard_area=apertures_area[i]/apertures_area[0])
                              for i in range(len(apertures_area))])
            if calibration_uncertainty is not None:
                sigma = np.array([math.sqrt(i) for i in (sigma ** 2 + (phot_bkgsub * calibration_uncertainty) ** 2)])
        else:
            if calibration_uncertainty is not None:
                sigma = phot_bkgsub * calibration_uncertainty

    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))

        if norm_kwargs is None:
            norm_kwargs = dict(percent=99.99, stretch='asinh', asinh_a=0.0001)
        norm = simple_norm(image, **norm_kwargs)

        ax.imshow(image, origin='lower', cmap='Greys_r', norm=norm)

        apertures.plot(ax=ax, color='cyan', lw=1.5, label='Photometry')
        annulus.plot(ax=ax, color='cyan', ls='--', lw=1.0, label='Background')
        ax.legend(loc='upper right', fontsize=16)

    if error:
        return phot_bkgsub, sigma
    else:
        return phot_bkgsub

def error_curve_fit(area, noise, standard_area=1, plot=False):
    log_area = np.array([math.log10(i) for i in area])
    log_noise = np.array([math.log10(i) for i in noise])
    curve = np.polyfit(log_area, log_noise, 1)
    slope = curve[0]
    intercept = curve[1]

    log_ans = intercept + slope * math.log10(standard_area)
    ans = 10 ** log_ans

    if plot:
        x = np.array(area)
        y = np.array(noise)
        curve = np.polyfit(np.log10(x), np.log10(y), 1)
        slope = curve[0]
        intercept = curve[1]
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(7, 7))
        ax.scatter(np.log10(x), np.log10(y))

        x_line = [min(np.log10(x)), 0.5]
        y_line = [slope * i + intercept for i in x_line]
        ax.plot(x_line, y_line, ls='--')
        ax.scatter(0., slope * 0. + intercept, c='r')
        ax.axhline(y=intercept, c='k', ls='--', xmax=0.85)
        ax.axvline(x=0, c='k', ls='--', ymax=0.85)
        ax.set_xlabel('$log \ mini-aperture \ area \ [\%]$')
        ax.set_ylabel('$log \ sigma \ [Jy]$')
        ax.text(0.95, 0.05, f'Phot Sigma: {10 ** (intercept):.2e}',
                fontsize=14, transform=ax.transAxes, va='bottom', ha='right',
                bbox=dict(alpha=0.8, color='w'))

    return ans


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


def plot_mask_contours(mask, ax=None, verbose=False, **plot_kwargs):
    '''
    Plot the mask in contours.
    '''
    pList = get_mask_polygons(mask)

    if verbose:
        print(f'Found {len(pList)} masks!')
        pList = tqdm.tqdm(pList)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    for p in pList:
        poly = shape(p)
        xy_poly = np.c_[poly.exterior.xy]
        ax.plot(xy_poly[:, 0]-0.5, xy_poly[:, 1]-0.5, **plot_kwargs)
    return ax


def plot_segment_contours(segm, ax=None, connectivity=8, verbose=False, cmap=None, **plot_kwargs):
    '''
    Plot the SegmentionImage in contours.

    Parameters
    ----------
    segm : 2D array
    '''
    polygons = np.array(list(shapes(segm.astype('int32'), connectivity=connectivity)))
    fltr = polygons[:, 1] > 0
    pList = polygons[fltr, 0]
    vList = polygons[fltr, 1]
    vmax = np.max(vList)

    if verbose:
        print(f'Found {len(pList)} masks!')
        pList = tqdm.tqdm(pList)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    for p, v in zip(pList, vList):
        poly = shape(p)
        xy_poly = np.c_[poly.exterior.xy]

        kwargs = plot_kwargs.copy()
        if cmap is not None:
            kwargs['c'] = cmap(v/vmax)

        if ('c' not in kwargs) & ('color' not in kwargs):
            kwargs['c'] = f'C{int(v%10)}'

        x = xy_poly[:, 0] - 0.5
        y = xy_poly[:, 1] - 0.5
        ax.plot(x, y, **kwargs)
    return ax


def polys_to_mask(polys, mask_shape):
    '''
    Convert the polygon list to a mask.
    '''
    sList = [shape(p) for p in polys]
    mask = rasterize(sList, out_shape=mask_shape).astype('bool')
    return mask


def poly_to_xy(poly):
    '''
    Convert polygon to xy array.

    Parameters
    ----------
    poly : dict
        {'type': 'Poly', 'coordinates':[[(x, y), ...]]}

    Returns
    -------
    xy : 2D array
        The coordinates, [m, 2], for m points.
    '''
    xy = np.array(poly['coordinates'][0])
    return xy


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
        c = SkyCoord('{0} {1}'.format(ra, dec), frame='icrs', unit=(units.hourangle, units.deg))
    else:
        c = SkyCoord(ra, dec, frame='icrs', unit='deg')
    return c


def rebin_image(image, factor=10, plot=False, norm_kwargs=None):
    '''
    Rebin image to reduce the image size.

    Parameters
    ----------
    image : 2D array
        The image data.
    factor : int
        The factor to rebin the image.
    plot : bool (default: False)
            Plot the data and segmentation map if True.
    norm_kwargs (optional) : dict
        The keywords to normalize the data image.
    '''
    ny, nx = image.shape
    input_wcs = WCS(naxis=2)
    output_wcs = WCS(naxis=2)

    input_wcs.wcs.crpix = ny // 2, nx // 2
    input_wcs.wcs.cdelt = -1, 1
    output_wcs.wcs.crpix = input_wcs.wcs.crpix / factor
    output_wcs.wcs.cdelt = input_wcs.wcs.cdelt * factor

    shape_out = (ny // factor, nx // factor)
    image_rb, _ = reproject_interp((image, input_wcs), output_wcs, shape_out=shape_out)

    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(14, 7))

        if norm_kwargs is None:
            norm_kwargs = dict(percent=99.99, stretch='asinh', asinh_a=0.001)

        ax = axs[0]
        norm = simple_norm(image, **norm_kwargs)
        ax.imshow(image, origin='lower', norm=norm, cmap='Greys_r')

        ax = axs[1]
        ax.imshow(image_rb, origin='lower', norm=norm, cmap='Greys_r')

    return image_rb


def scale_mask(mask, factor, connectivity=8):
    '''
    Scale the mask.

    Parameters
    ----------
    mask : 2D array
        The mask or segments. The masked region have True or >=1 values.
    factor : float
        The scaling factor of the mask
    connectivity : {4, 8} (default: 8)
        The type of pixel connectivity used in determining how pixels are
        grouped into a detected source. The options are 4 or 8 (default).
        4-connected pixels touch along their edges. 8-connected pixels touch
        along their edges or corners.

    Returns
    -------
    mask_s : 2D array
        The scaled mask with masked region True.

    Notes
    -----
    [SGJY added]
    '''
    pList = get_mask_polygons(mask, connectivity=connectivity)

    sList = []
    for p in pList:
        sList.append(scale(shape(p), xfact=factor, yfact=factor))

    # Scale the polygon
    mask_s = rasterize(sList, out_shape=mask.shape).astype('bool')
    return mask_s


def segment_add(segm, mask, plot=False):
    '''
    Add a masked region in the segmentation.

    Parameters
    ----------
    segm : SegmentationImage
        The input SegmentationImage.
    mask : 2D array
        Mask of the region to be added.
    plot : bool (default: False)
        Plot the results if True.

    Returns
    -------
    segm_o : SegmentationImage
        The output segmentation.
    '''
    data = segm.data.copy()
    data[mask] = segm.nlabels + 1
    segm_o = SegmentationImage(data=data)

    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)

        ax = axs[0]
        ax.imshow(segm, origin='lower', cmap=segm.cmap, interpolation='nearest')
        plot_mask_contours(mask, ax=ax, color='cyan', lw=0.5)
        ax.set_title('Input SegmentationImage', fontsize=16)

        ax = axs[1]
        ax.imshow(segm_o, origin='lower', cmap=segm_o.cmap, interpolation='nearest')
        plot_mask_contours(mask, ax=ax, color='cyan', lw=0.5)
        ax.set_title('Output SegmentationImage', fontsize=16)
    return segm_o


def segment_combine(segm1, segm2, progress_bar=False, plot=False):
    '''
    Combine two segmentations.

    Parameters
    ----------
    segm1, segm2 : SegmentationImage
        The input segmentations.
    progress_bar: bool (default: False)
        Show the progress if True.
    plot : bool (default: False)
        Plot the results if True.

    Returns
    -------
    segm_combined : SegmentationImage
        The combined segmentations.
    '''
    assert segm1.shape == segm2.shape, 'The two segments should have the same shape!'

    # Add the smaller segment to the larger segment to be fast.
    if segm1.nlabels > segm2.nlabels:
        segm_l = segm1
        segm_s = segm2
    else:
        segm_l = segm2
        segm_s = segm1

    data = segm_l.data.copy()

    if progress_bar:
        labels = tqdm.tqdm(segm_s.labels)
    else:
        labels = segm_s.labels

    for l in labels:
        data[segm_s.data == l] = segm_l.nlabels + l

    # Create a SegmentationImage using the data
    segm_combined = SegmentationImage(data=data)

    if plot:
        fig, axs = plt.subplots(1, 3, figsize=(21, 7), sharex=True, sharey=True)

        ax = axs[0]
        ax.imshow(segm1, origin='lower', cmap=segm1.cmap, interpolation='nearest')
        ax.set_title('Segm1', fontsize=16)

        ax = axs[1]
        ax.imshow(segm2, origin='lower', cmap=segm2.cmap, interpolation='nearest')
        ax.set_title('Segm2', fontsize=16)

        ax = axs[2]
        ax.imshow(segm_combined, origin='lower', cmap=segm_combined.cmap, interpolation='nearest')
        ax.set_title('Segm_combined', fontsize=16)

    return segm_combined


def segment_remove(segm, mask, overwrite=False, plot=False):
    '''
    Remove segments in the masked region.

    Parameters
    ----------
    segm : SegmentationImage
        The inpug SegmentationImage.
    mask : 2D array
        Mask of the region to be added.
    plot : bool (default: False)
        Plot the results if True.

    Returns
    -------
    segm_o : SegmentationImage
        The output segmentation.
    '''
    if overwrite:
        segm_o = segm
    else:
        segm_o = deepcopy(segm)

    # Get the center of all the segments
    cat = SourceCatalog(segm_o.data, segm_o, progress_bar=False)
    tb = cat.to_table()
    x_cent = tb['xcentroid']
    y_cent = tb['ycentroid']
    c_cent = np.c_[x_cent, y_cent]
    mpoint = MultiPoint(c_cent)

    # Make the target polygon and select segments
    p_mask = get_mask_polygons(mask)
    s_mask = shape(p_mask[0])

    fltr = np.array([s_mask.contains(p) for p in mpoint.geoms])
    labels = tb[fltr]['label'].data
    segm_o.remove_labels(labels, relabel=True)

    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)

        ax = axs[0]
        ax.imshow(segm, origin='lower', cmap=segm.cmap, interpolation='nearest')
        plot_mask_contours(mask, ax=ax, color='cyan', lw=0.5)
        ax.set_title('Input SegmentationImage', fontsize=16)

        ax = axs[1]
        ax.imshow(segm_o, origin='lower', cmap=segm_o.cmap, interpolation='nearest')
        plot_mask_contours(mask, ax=ax, color='cyan', lw=0.5)
        ax.set_title('Output SegmentationImage', fontsize=16)

    return segm_o


def select_segment_stars(image, segm, wcs, convolved_image=None, mask=None,
                         xmatch_radius=3, plx_snr=3, plot=False,
                         norm_kwargs=None):
    '''
    Select the stars corresponding to the segments.

    Parameters
    ----------
    image : 2D array
        The image data.
    segm : SegmentationImage
        The input SegmentationImage.
    wcs : WCS
        The WCS of the image to get the on-sky coordinates of the sources.
    convolved_image (optional) : 2D array
        The 2D array used to calculate the source centroid and morphological
        properties.
    mask : 2D array
        A boolean mask with the same shape as data where a True value indicates
        the corresponding element of data is masked. Masked data are excluded
        from all calculations.
    xmatch_radius : float (default: 3)
        The cross-matching radius, units: arcsec.
    plx_snr : float
        The SNR of the parallax to select stars.
    plot : bool (default: False)
        Plot the data and segmentation map if True.
    norm_kwargs (optional) : dict
        The keywords to normalize the data image.

    Notes
    -----
    [SGJY added]
    '''
    cat = SourceCatalog(image, segm, convolved_data=convolved_image, mask=mask,
                        wcs=wcs)
    tb = cat.to_table()
    coo = tb['sky_centroid']

    xmTb = Table([tb['label'], tb['xcentroid'], tb['ycentroid'], coo.ra.deg,
                  coo.dec.deg, tb['area'], tb['semimajor_sigma'],
                  tb['semiminor_sigma'], tb['orientation'],
                  tb['eccentricity'], tb['segment_flux']],
                 names=['label', 'x', 'y', 'ra', 'dec',
                        'area', 'semimajor_sigma', 'semiminor_sigma',
                        'orientation', 'eccentricity', 'segment_flux'])
    tb_o = xmatch_gaiadr3(xmTb, radius=xmatch_radius, colRA1='ra', colDec1='dec')

    # Select the stars
    fltr = ~tb_o['Plx'].mask & (tb_o['Plx'] / tb_o['e_Plx'] > plx_snr)
    tb_s = tb_o[fltr]

    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)

        if norm_kwargs is None:
            norm_kwargs = dict(percent=99.99, stretch='asinh', asinh_a=0.001)
        norm = simple_norm(image, **norm_kwargs)

        ax = axs[0]
        ax.imshow(image, origin='lower', cmap='Greys_r', norm=norm)
        ax.plot(tb_s['x'], tb_s['y'], ls='none', marker='o', ms=6,
                mfc='none', mec='red', mew=0.2)

        ax = axs[1]
        ax.imshow(segm, origin='lower', cmap=segm.cmap, interpolation='nearest')
        ax.plot(tb_s['x'], tb_s['y'], ls='none', marker='o', ms=6,
                mfc='none', mec='red', mew=0.2)
    return tb_s


def simplify_mask(mask, connectivity=8):
    '''
    Simplify the mask to the convex hull.

    Parameters
    ----------
    mask : 2D array
        The mask or segments. The masked region have True or >=1 values.
    connectivity : {4, 8} (default: 8)
        The type of pixel connectivity used in determining how pixels are
        grouped into a detected source. The options are 4 or 8 (default).
        4-connected pixels touch along their edges. 8-connected pixels touch
        along their edges or corners.

    Returns
    -------
    mask_s : 2D array
        The simplified mask with masked region True.
    '''
    pList = get_mask_polygons(mask, connectivity=connectivity)

    # Get the convex hull
    sList = [shape(p).convex_hull for p in pList]

    # Convert back to masks
    mask_s = rasterize(sList, out_shape=mask.shape).astype('bool')
    return mask_s


def xmatch_gaiadr3(cat, radius, colRA1='ra', colDec1='dec'):
    '''
    Cross match the catalog with Gaia DR3.

    Parameters
    ----------
    cat : Astropy Table
        A table with the source ra and dec to cross-match with Gaia DR3.
    radius : float
        Cross-match radius, units: arcsec.
    colRA1 : string (default: 'ra')
        The Column of the ra.
    colDec1 : string (default: 'dec')
        The Column of the dec.

    Returns
    -------
    t_o : Astropy Table
        The output table.

    Notes
    -----
    [SGJY added]
    '''
    t_o = XMatch.query(cat1=cat, cat2='vizier:I/355/gaiadr3', max_distance=radius * units.arcsec,
                       colRA1=colRA1, colDec1=colDec1)  # Gaia xmatch.
    return t_o

import tqdm
import warnings
import numpy as np
from copy import deepcopy
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
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

from photutils.segmentation import make_2dgaussian_kernel, detect_sources, deblend_sources
from photutils.segmentation import SourceFinder, SegmentationImage, SourceCatalog
from rasterio.features import rasterize, shapes
from shapely.geometry import shape, MultiPoint
from shapely.affinity import scale
from scipy.ndimage import gaussian_filter

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
    t_o = XMatch.query(cat1=cat, cat2='vizier:I/355/gaiadr3', max_distance=radius*u.arcsec, 
                       colRA1=colRA1, colDec1=colDec1)  # Gaia xmatch.
    return t_o


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
    pList = polygons[vals == 1, 0]
    return pList


def scale_mask(mask, factor, connectivity=8):
    '''
    Scale the mask.

    Parameters
    ----------
    mask : 2D array
        The mask with masked in region True or 1.
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
        sList.append( scale(shape(p), xfact=factor, yfact=factor) )
    
    # Scale the polygon
    mask_s = rasterize(sList, out_shape=mask.shape).astype('bool')
    return mask_s


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

    if plot_kwargs is None:
        plot_kwargs = dict(color='cyan', lw=1)

    for p in pList:
        poly = shape(p)
        xy_poly = np.c_[poly.exterior.xy]
        ax.plot(xy_poly[:, 0], xy_poly[:, 1], **plot_kwargs)
    return ax


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
    r = np.sqrt(np.sum((xy_poly - coord_pix[np.newaxis, :])**2, axis=1))
        
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
        ax.plot(xy_poly[:, 0]-x1, xy_poly[:, 1]-y1, color='cyan', lw=2)
        
    return data_s, bounds


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
        convolved_data = gaussian_filter(data, kernel_fwhm/2.3548)
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


def add_mask_circle(mask, x, y, radius):
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
        
    Returns
    -------
    mask : 2D bool array
        Output mask, True for masked region.
    '''
    nx = np.arange(mask.shape[1])
    ny = np.arange(mask.shape[0])
    xx, yy = np.meshgrid(nx, ny)
    r = np.sqrt((xx - x)**2 + (yy - y)**2)
    mask[r <= radius] = 1
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


def polys_to_mask(polys, mask_shape):
    '''
    Convert the polygon list to a mask.
    '''
    sList = [shape(p) for p in polys]
    mask = rasterize(sList, out_shape=mask_shape).astype('bool')
    return mask


def gen_image_mask(image, threshold, npixels=5, mask=None, connectivity=8, 
                   kernel_fwhm=0, deblend=False, nlevels=32, contrast=0.001, 
                   mode='linear', nproc=1, progress_bar=True, expand_factor=1.2, 
                   bounds:list=None, choose_coord=None, plot=False, fig=None, 
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
    
    smap, cdata = get_image_segmentation(img, threshold=threshold, 
                                         npixels=npixels, mask=mask, 
                                         connectivity=connectivity, 
                                         kernel_fwhm=kernel_fwhm, 
                                         deblend=deblend, nlevels=nlevels, 
                                         contrast=contrast, mode=mode, 
                                         nproc=nproc, progress_bar=progress_bar, 
                                         plot=False)

    if choose_coord is None:
        mask = smap.data > 0
    
    else:
        x, y = choose_coord

        if bounds is None:
            mask = smap.data == smap.data[int(y), int(x)]
        else:
            mask = smap.data == smap.data[int(y)-ymin, int(x)-xmin]

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
        xlim = ax.get_xlim(); ylim = ax.get_ylim()
        plot_mask_contours(mask, ax=ax, verbose=verbose, color='cyan', lw=0.5)
        
        if bounds is not None:
            x = [bounds[0], bounds[0], bounds[1], bounds[1], bounds[0]]
            y = [bounds[2], bounds[3], bounds[3], bounds[2], bounds[2]]
            ax.plot(x, y, ls='--', color='red')
        
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_title('Image', fontsize=18)

        ax = axs[1]
        ax.imshow(smap, origin='lower', cmap=smap.cmap, interpolation='nearest')
        xlim = ax.get_xlim(); ylim = ax.get_ylim()
        plot_mask_contours(mask_e, ax=ax, verbose=verbose, color='cyan', lw=0.5)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_title('Segmentation map', fontsize=18)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    return mask, smap, cdata


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


def segment_add(segm, mask, plot=False):
    '''
    Add a masked region in the segmentation.

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


def detect_source_extended(image:np.array, target_coord:tuple, target_mask:np.array, 
                           threshold_o:float, threshold_i:float, npixels_o=5, 
                           npixels_i=5, nlevels_o=32, nlevel_i=256, contrast_o=0.001, 
                           contrast_i=1e-6, connectivity=8, kernel_fwhm=0, 
                           mode='linear', nproc=1, progress_bar=False,  
                           plot=False, fig=None, axs=None, norm_kwargs=None, 
                           interactive=False, verbose=False):
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
    segm_o, _ = get_image_segmentation(image, threshold_o, npixels=npixels_o, 
                                       mask=target_mask, kernel_fwhm=kernel_fwhm, 
                                       nlevels=nlevels_o, contrast=contrast_o,
                                       connectivity=connectivity, mode=mode,
                                       nproc=nproc, deblend=True, 
                                       progress_bar=progress_bar)

    # Get the segments inside the target mask
    segm_i0 = detect_sources(image, threshold_i, npixels=npixels_i, mask=~target_mask)
    debl_i = deblend_sources(image, segm_i0, npixels=npixels_i, labels=None, 
                               nlevels=nlevel_i, contrast=contrast_i, mode=mode, 
                               connectivity=connectivity, relabel=True, 
                               nproc=nproc, progress_bar=progress_bar)
    # Get the mask of the galaxy innter region
    x_t, y_t = target_coord
    target_mask_i = segm_i0.data == segm_i0.data[int(y_t), int(x_t)]
    # Final segments; merged the segments inside target_mask_i.
    segm_i = segment_add(debl_i, target_mask_i)  
    
    # Combine the segments
    segm_c = segment_combine(segm_o, segm_i, progress_bar=progress_bar)
    
    # Remove the segments in the inner region of the target
    segment_remove(segm_c, target_mask_i, overwrite=True)

    if plot:
        if interactive:
            ipy = get_ipython()
            ipy.run_line_magic('matplotlib', 'tk')

            def on_close(event):
                ipy.run_line_magic('matplotlib', 'inline')

        if axs is None:
            fig, axs = plt.subplots(2, 2, figsize=(14, 14), sharex=True, sharey=True)
            fig.subplots_adjust(hspace=0.07, wspace=0.05)
        else:
            assert fig is not None, 'Please provide fig together with axs!'

        if interactive:
            fig.canvas.mpl_connect('close_event', on_close)
        
        if norm_kwargs is None:
            norm_kwargs = dict(percent=99.99, stretch='asinh', asinh_a=0.001)
        norm = simple_norm(image, **norm_kwargs)

        ax = axs[0, 0]
        ax.imshow(image, origin='lower', cmap='Greys_r', norm=norm)
        ax.plot(target_coord[0], target_coord[1], marker='+', ms=10, color='red')
        xlim = ax.get_xlim(); ylim = ax.get_ylim()
        plot_mask_contours(target_mask, ax=ax, verbose=verbose, color='cyan', lw=0.5)
        plot_mask_contours(target_mask_i, ax=ax, verbose=verbose, color='magenta', lw=0.5)
        
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_title('Image', fontsize=18)

        ax = axs[0, 1]
        ax.imshow(segm_c, origin='lower', cmap=segm_c.cmap, interpolation='nearest')
        xlim = ax.get_xlim(); ylim = ax.get_ylim()
        plot_mask_contours(target_mask, ax=ax, verbose=verbose, color='cyan', lw=0.5)
        plot_mask_contours(target_mask_i, ax=ax, verbose=verbose, color='magenta', lw=0.5)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_title('Combined segmentation', fontsize=18)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax = axs[1, 0]
        ax.imshow(segm_i, origin='lower', cmap=segm_i.cmap, interpolation='nearest')
        xlim = ax.get_xlim(); ylim = ax.get_ylim()
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_title('Inner segmentation', fontsize=18)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax = axs[1, 1]
        ax.imshow(segm_o, origin='lower', cmap=segm_o.cmap, interpolation='nearest')
        xlim = ax.get_xlim(); ylim = ax.get_ylim()
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_title('Outer segmentation', fontsize=18)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    detection_results = {
        'segment_out': segm_o,
        'segment_in': segm_i,
        'segment_combine': segm_c,
        'target_mask_o': target_mask,
        'target_mask_i': target_mask_i,
    }
    return detection_results


def select_segment_stars(image, segm, wcs, convolved_image=None, plx_snr=3, 
                         mask=None, plot=False, norm_kwargs=None):
    '''
    Select
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
    tb_o = xmatch_gaiadr3(xmTb, radius=3, colRA1='ra', colDec1='dec')

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


def cutout_star(image, segm, coord_pix, extract_size, sigma=1, plot=True):
    '''
    Extract the radial profile around the specified coordinate.
    '''
    segm = deepcopy(segm)
    l = segm.data[int(coord_pix[1]), int(coord_pix[0])]
    #segm.remove_label(l)
    mask = segm.data > 0
    
    img_c = Cutout2D(image, position=coord_pix, size=extract_size)
    mask_c = Cutout2D(mask, position=coord_pix, size=extract_size)
    data = img_c.data
    mask = mask_c.data
    
    ny, nx = data.shape
    yy, xx = np.mgrid[:ny, :nx]

    amp = np.percentile(data, 99)
    mean_init = extract_size/2
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
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
from astropy.visualization import AsinhStretch, SqrtStretch, LogStretch
from astropy.visualization import PercentileInterval, simple_norm
from astropy.visualization.mpl_normalize import ImageNormalize
from astroquery.xmatch import XMatch
from astropy.convolution import convolve
import tqdm

from photutils.segmentation import make_2dgaussian_kernel, detect_sources, deblend_sources
from photutils.segmentation import SourceFinder
from rasterio.features import rasterize, shapes
from shapely.geometry import shape
from shapely.affinity import scale

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


def get_masked_patch(data, mask, coord_pix, factor=1, plot=False, axs=None, norm_kwargs=None):
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
                           kernel_fwhm=2, deblend=False, nlevels=32, contrast=0.001, 
                           mode='linear', nproc=1, progress_bar=True, 
                           plot=False, axs=None, norm_kwargs=None):
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
    kernel_fwhm : float (default: 2)
        The kernel FWHM to smooth the image. If kernel_fwhm=0, skip the convolution.
    plot : bool (default: False)
        Plot the data and segmentation map if True.
    axs : matplotlib axes
        The axes to plot the data and segmentation map. Must be >=2 panels.
    norm_kwargs (optional) : dict
        The keywords to normalize the data image.

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
        kernel = make_2dgaussian_kernel(fwhm=kernel_fwhm, size=(2*kernel_fwhm+1))
        convolved_data = convolve(data, kernel)
    else:
        raise ValueError(f'The kernel_fwhm ({kernel_fwhm}) has to be >=0!')

    #segment_map = detect_sources(convolved_data, threshold=threshold, npixels=npixels, 
    #                             connectivity=connectivity, mask=mask)
    finder = SourceFinder(npixels=npixels, connectivity=connectivity, 
                          deblend=deblend, nlevels=nlevels, contrast=contrast, 
                          mode=mode, relabel=True, nproc=nproc, 
                          progress_bar=progress_bar)
    segment_map = finder(convolved_data, threshold=threshold, mask=mask)

    if plot:
        if not axs:
            fig, axs = plt.subplots(1, 2, figsize=(14, 7))
            plain = False
        else:
            plain = True
        if norm_kwargs is None:
            norm_kwargs = dict(percent=99.99, stretch='asinh', asinh_a=0.001)

        norm = simple_norm(convolved_data, **norm_kwargs)

        ax = axs[0]
        ax.imshow(convolved_data, origin='lower', cmap='Greys_r', norm=norm)
        ax = axs[1]
        ax.imshow(segment_map, origin='lower', cmap=segment_map.cmap)

        if not plain:
            axs[0].set_title('Convolved data')
            axs[1].set_title('Segmentation Image')
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


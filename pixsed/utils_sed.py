import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy import units
from astropy.coordinates import SkyCoord
from astropy.visualization import simple_norm
from vorbin.voronoi_2d_binning import voronoi_2d_binning
from astroquery.ipac.irsa.irsa_dust import IrsaDust
from photutils.segmentation import SegmentationImage

from .utils import plot_segment_contours, plot_mask_contours


def binmap_voronoi(image, error, mask, target_sn, pixelsize=1, cvt=True, 
                   wvt=True, sn_func=None, plot=False, fig=None, axs=None, 
                   norm_kwargs=None, label_kwargs=None, interactive=False, 
                   verbose=False):
    '''
    Generate the Voronoi binning with vorbin.

    Parameters
    ----------
    image : 2D array
        The image array.
    error : 2D array
        The error array.
    mask : 2D array
        The mask of the target.
    pixelsize : float (default: 1)
        The pixel size.
    target_sn : float
        The target SNR of the voronoi binning.
    cvt : bool (default: True)
        Use the Centroidal Voronoi Tessellation (CVT) algorithm if True.  
        See the doc of vorbin.
    wvt : bool (default: True)
        Use the Weighted Voronoi Tessellation method to modify the binning. 
        See the doc of vorbin.
    sn_func (optional) : callable
        The function to calculate SNR. See the doc of vorbin.
    plot : bool (default: False)
        Plot the results if True.
    fig : Matplotlib Figure
        The figure to plot.
    axs : Matplotlib Axes
        The axes to plot. Two panels are needed.
    norm_kwargs (optional) : dict
        The keywords to normalize the data image.
    label_kwargs (optional) : dict
        Add the label to the voronoi bins if specified.
    interactive : bool (default: False)
        Use the interactive plot if True.
    verbose : bool (default: True)
        Show details if True.

    Returns
    -------
    bin_info : dict
        image, noise, mask, target_sn : input parameters
        segm : 2D array
            The segmentation map of the binning.
    axs : list of Axes
        The plot axes.

    Notes
    -----
    The generated segm is starts from 1, so the values are (bin_num + 1). 
    '''
    # Prepare the input of vorbin
    xi = np.arange(0, image.shape[1])
    yi = np.arange(0, image.shape[0])
    x = xi - np.mean(xi)
    y = yi - np.mean(yi)
    xx, yy = np.meshgrid(x, y)
    xList = xx[mask]
    yList = yy[mask]
    signal = image[mask]
    noise = error[mask]

    results = voronoi_2d_binning(
        xList, yList, signal, noise, target_sn, cvt=cvt, wvt=wvt, 
        sn_func=sn_func, pixelsize=1, plot=False, quiet=~verbose)
    
    xmin = x.min()
    ymin = y.min()
    x_bar = results[3] - xmin
    y_bar = results[4] - ymin
    x_index = (xList - xmin).astype(int)
    y_index = (yList - ymin).astype(int)

    nbins = np.max(results[0])+1
    segm = np.zeros_like(image, dtype=int)
    segm[mask] = results[0] + 1
    cmap = rand_cmap(nbins+1, type='soft', first_color_white=True, verbose=False)

    bin_info = {
        'image': image,
        'noise': noise,
        'mask': mask,
        'segm': segm,
        'cmap': cmap,
        'target_sn': target_sn,
        'nbins': nbins,
        'bin_num': results[0],
        'x_index': x_index,
        'y_index': y_index,
        'x_bar': x_bar,
        'y_bar': y_bar,
        'sn': results[5],
        'nPixels': results[6],
        'scale': results[7]
    }
    
    if plot:
        if interactive:
            ipy = get_ipython()
            ipy.run_line_magic('matplotlib', 'tk')

            def on_close(event):
                ipy.run_line_magic('matplotlib', 'inline')

        if axs is None:
            fig = plt.figure(figsize=(16, 12))
            ax0 = fig.add_axes([0.05, 0.5, 0.45, 0.45])
            ax1 = fig.add_axes([0.5, 0.5, 0.45, 0.45])
            ax2 = fig.add_axes([0.05, 0.05, 0.9, 0.30])
            axs = [ax0, ax1, ax2]
        
        if interactive:
            fig.canvas.mpl_connect('close_event', on_close)

        if norm_kwargs is None:
            norm_kwargs = dict(percent=99.99, stretch='asinh', asinh_a=0.001)
        norm = simple_norm(image, **norm_kwargs)

        ax = axs[0]
        ax.imshow(image, origin='lower', cmap='Greys_r', norm=norm)
        plot_segment_contours(segm, cmap=cmap, ax=ax, alpha=0.5)
        ax.set_xlabel(r'$X$ (pixel)', fontsize=24)
        ax.set_ylabel(r'$Y$ (pixel)', fontsize=24)

        ax = axs[1]
        ax.imshow(segm, origin='lower', cmap=cmap)
        plot_mask_contours(mask, ax=ax, color='k', verbose=verbose)
        ax.set_xlabel(r'$X$ (pixel)', fontsize=24)
        ax.set_ylabel(r'$Y$ (pixel)', fontsize=24)

        if label_kwargs is not None:
            if 'fontsize' not in label_kwargs:
                label_kwargs['fontsize'] = 6

            if 'color' not in label_kwargs:
                label_kwargs['color'] = 'k'

            for xb, yb in zip(x_bar, y_bar):
                v = segm[int(yb), int(xb)] - 1
                ax.text(xb, yb, f'{v}', transform=ax.transData, ha='center', 
                        va='center', **label_kwargs)

        ax = axs[2]
        r_input = np.sqrt(xList**2 + yList**2) * pixelsize
        snr_input = signal / noise
        ax.scatter(r_input, snr_input, color='k', s=20, label='Input S/N')

        r_output = np.sqrt(results[3]**2 + results[4]**2) * pixelsize
        snr_output = results[5]

        fltr = results[6] == 1
        if np.sum(fltr) > 0:
            ax.scatter(r_output[fltr], snr_output[fltr], marker='x', color='b', s=50, 
                       label='Not binned')
        ax.scatter(r_output[~fltr], snr_output[~fltr], color='r', s=50, 
                   label='Voronoi bins')
        ax.axhline(y=target_sn, ls='--', color='C0', label='Target S/N')
        ax.set_xlim([0, r_output.max()])
        ax.legend(loc='upper right', fontsize=16)
        ax.minorticks_on()
        ax.set_xlabel(r'Radius (arcsec)', fontsize=24)
        ax.set_ylabel(r'S/N', fontsize=24)
        
    return bin_info, axs


def get_Galactic_Av(ra, dec, band='CTIO V', ref='A_SandF'):
    '''
    Get the Milky Way extinction, Av, from IRSA Dust Extinction Service.

    https://astroquery.readthedocs.io/en/latest/ipac/irsa/irsa_dust/irsa_dust.html

    Parameters
    ----------
    ra, dec : floats
        The coordinate of the target, units: degree.
    band : string (default: 'CTIO V')
        The optical band name.
    ref : string (default: 'A_SandF')
        The measurement work. SandF stands for Schlafly & Finkbeiner (2011).
        Refer to the full table to check the available data.
        Return the full table if `ref=None`.
    
    Returns
    -------
    If `ref=None`, return the full querried table, otherwise, return the A_V value.
    '''
    c = SkyCoord(ra=ra*units.degree, dec=dec*units.degree)
    tb = IrsaDust.get_extinction_table(c)

    if ref is None:
        return tb
    else:
        idx = np.where(tb['Filter_name'] == band)[0]

        if len(idx) == 0:
            raise KeyError(f'Cannot find the band ({band})!')
        elif len(idx) > 1:
            print('More than 1 value found. Please check the entire table!')
            return tb
        else:
            Av = tb[ref][idx[0]]
            return Av


def get_Galactic_Alambda(wavelength, A_V, model='F99', Rv='3.1'):
    '''
    Get the wavelength dependent extinction of the Milky Way.

    Parameters
    ----------
    wavelength : Quantity
        The wavelength with units e.g. Angstrom or micron.
    A_V : float
        The extinction in V band. It can be obtained by `get_Galactic_Av()`.
    model : {'F99' or 'CCM89'} (default: 'F99')
        The extinction model.
    Rv : string (default: '3.1')
        The Rv of the extinction model.

    Returns
    -------
    A_lambda : floats
        The extinction of the corresponding wavelengths.

    Notes
    -----
    f_corr = f * 10**(0.4 * A_lambda)
    '''
    if model == 'F99':
        from dust_extinction.parameter_averages import F99
        ext_model = F99(Rv=Rv)
    elif model == 'CCM89':
        from dust_extinction.parameter_averages import CCM89
        ext_model = CCM89(Rv=Rv)
    else:
        raise KeyError(f'Cannot recognize the model ({model})!')
    
    w_range = 1 / np.array(ext_model.x_range) * units.micron

    fltr = wavelength < w_range[1]
    if np.sum(fltr) > 0:
        raise ValueError(f'The short wavelength data ({wavelength[fltr]}) are out of the working range!')

    fltr = (wavelength < w_range[0])
    A_lambda = np.zeros(len(wavelength))
    A_lambda[fltr] = ext_model(wavelength[fltr]) * A_V
    return A_lambda


def rand_cmap(nlabels, type='bright', first_color_white=True, last_color_white=False, verbose=True):
    """
    From : https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib

    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_white: Option to use first color as white, True or False
    :param last_color_white: Option to use last color as white, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """
    from matplotlib.colors import LinearSegmentedColormap
    import colorsys
    import numpy as np


    if type not in ('bright', 'soft'):
        print ('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_white:
            randRGBcolors[0] = [1, 1, 1]

        if last_color_white:
            randRGBcolors[-1] = [1, 1, 1]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        from matplotlib import colors, colorbar
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                   boundaries=bounds, format='%1i', orientation=u'horizontal')

    return random_colormap
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
from astropy import units
from astropy.coordinates import SkyCoord
from astropy.visualization import simple_norm
from vorbin.voronoi_2d_binning import voronoi_2d_binning
from astroquery.ipac.irsa.irsa_dust import IrsaDust
import corner
import dill as pickle

from sedpy.observate import load_filters
from prospect.utils.obsutils import fix_obs
from prospect.models.templates import TemplateLibrary
from prospect.models import SpecModel
from prospect.sources import CSPSpecBasis
from prospect.fitting import lnprobfn, fit_model
from prospect.io import read_results as reader

from .utils import plot_segment_contours, plot_mask_contours


import matplotlib as mpl
mpl.rc("xtick", direction="in", labelsize=16)
mpl.rc("ytick", direction="in", labelsize=16)
mpl.rc("xtick.major", width=1., size=8)
mpl.rc("ytick.major", width=1., size=8)
mpl.rc("xtick.minor", width=1., size=5)
mpl.rc("ytick.minor", width=1., size=5)


def binmap_voronoi(image, error, mask, target_sn, pixelsize=1, cvt=True, 
                   wvt=True, sn_func=None, plot=False, fig=None, axs=None, 
                   norm_kwargs=None, label_kwargs={}, interactive=False, 
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

        plot_bin_image(bin_info, fig=fig, ax=axs[0], norm_kwargs=norm_kwargs)
        plot_bin_segm(bin_info, fig=fig, ax=axs[1], label_kwargs=label_kwargs)
        axs[1].text(0.05, 0.95, f'Total bins: {nbins}', fontsize=16, 
                    transform=axs[1].transAxes, ha='left', va='top')

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


def fit_SED_Prospector(bands, maggies, maggies_unc, redshift, lumdist=None, 
                       model_params=None, noise_model=None, sps=None, 
                       optimize=False, emcee=False, dynesty=True, 
                       fitting_kwargs=None, print_progress=True):
    '''
    Fit an SED with Prospector.

    Parameters
    ----------
    bands : list
        A list of the band names. It should follow the rule of sedpy.
    maggies : 1D array
        The SED flux in the unit of maggies (3631 Jy).
    maggies_unc : 1D array
        The SED flux uncertainties in the unit of maggies (3631 Jy).
    redshift : float
        The redshift of the source.
    lumdist (optional) : float
        The luminosity distance of the source, units: Mpc. If not provided, 
        Prospector will use the redshift to calculate the lumdist.
    '''
    # Observation
    filters = load_filters([f'{b}' for b in bands])
    obs = dict(wavelength=None, spectrum=None, unc=None, redshift=redshift,
               maggies=maggies, maggies_unc=maggies_unc, filters=filters)
    obs = fix_obs(obs)

    # Model
    if model_params is None:
        model_params = TemplateLibrary["parametric_sfh"]
        model_params.update(TemplateLibrary["dust_emission"])
        model_params['duste_umin']['isfree'] = True
        model_params['duste_qpah']['isfree'] = True
        model_params['duste_gamma']['isfree'] = True

    if lumdist is not None:
        model_params["lumdist"] = {"N": 1, "isfree": False, "init": lumdist, 
                                   "units":"Mpc"}

    model_params["zred"]["init"] = obs["redshift"]
    model = SpecModel(model_params)

    if noise_model is None:
        noise_model = (None, None)
        
    if sps is None:
        sps = CSPSpecBasis(zcontinuous=1)

    if fitting_kwargs is None:
        fitting_kwargs = dict(nlive_init=400, nested_sample="rwalk", 
                              nested_target_n_effective=1000, 
                              nested_dlogz_init=0.05)
    fitting_kwargs['print_progress'] = print_progress

    output = fit_model(obs, model, sps, optimize=optimize, emcee=emcee, 
                       dynesty=dynesty, lnprobfn=lnprobfn, noise=noise_model, **fitting_kwargs)

    return output, obs, model, sps


def gen_image_phys(parname, bin_info, bin_phys):
    '''
    Generate the map with the physical parameters from the SED fitting.
    '''
    bin_num = bin_info['bin_num']
    x_index = bin_info['x_index']
    y_index = bin_info['y_index']
    phypar = bin_phys[parname]

    image = np.full_like(bin_info['image'], np.nan)
    for b, x, y in zip(bin_num, x_index, y_index):
        image[y, x] = phypar[b]

    return image


def gen_image_density(parname, bin_info, bin_phys, pixel_to_area):
    '''
    Generate the density map with the physical parameters from the SED fitting.
    '''
    from collections import Counter

    bin_num = bin_info['bin_num']
    bin_area = Counter(bin_info['bin_num'])
    x_index = bin_info['x_index']
    y_index = bin_info['y_index']
    phypar = bin_phys[parname]

    image = np.full_like(bin_info['image'], np.nan)
    area = np.zeros_like(bin_info['image'])
    for b, x, y in zip(bin_num, x_index, y_index):
        image[y, x] = phypar[b]
        area[y, x] = bin_area[b]
    
    if parname in ['logmstar', 'logmdust']:
        density = image - np.log10(area * pixel_to_area)
    else:
        density = image / (area * pixel_to_area)

    return density


def get_BestFit_Prospector(output, model=None):
    '''
    Get the best-fit parameters of the Prospector results.

    Parameters
    ----------
    output : dict
    model (optional) : SpecModel
        Prospector SpecModel that was used in the fitting.
    '''
    if output['sampling'][0] is not None:
        results = output['sampling'][0]

        if 'logvol' in results:
            lnprob = results['logl'] + model.prior_product(results.samples, nested=True),
        else:
            raise NotImplementedError('To be implemented for the emcee results!')

        theta = results.samples[np.argmax(lnprob), :]
    else:
        results = output['optimization'][0]
        raise NotImplementedError('To be implemented for optimization!')

    return theta


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


def get_Params_Prospector(theta, model, sps, sfr_dt=0.1):
    '''
    Get the physical parameters from the Prospector fitting.
    '''
    params = {}

    model.set_parameters(theta)
    sps.update(**model.params)
    mass = np.atleast_1d(sps.params['mass']).copy()

    mstar, sfr, mdust = [], [], []
    # Loop over mass components
    for i, m in enumerate(mass):
        sps.update_component(i)
        mstar.append(sps.ssp.stellar_mass)
        sfr.append(sps.ssp.sfr_avg(times=model.params['tage'][0], dt=sfr_dt))
        mdust.append(sps.ssp.dust_mass)
    
    params['logmstar'] = np.log10(np.dot(mass, mstar))
    params['sfr'] = np.dot(mass, sfr)
    params['logmdust'] = np.log10(np.dot(mass, mdust))
    params['A_V'] = model.params['dust2'][0] * 2.5 * np.log10(np.e)
    return params


def get_Samples_Prospector(output, model, sps, sfr_dt=0.1, show_progress=False):
    '''
    Get the samples of the physical parameters from the Prospector fitting.
    '''
    results = output['sampling'][0]
    assert results is not None, 'The output should be from emcee or dynesty sampling!'

    if 'logvol' in results:
        s = results.samples_equal()
    else:
        raise NotImplementedError('To be implemented to work with emcee results!')
    
    if show_progress:
        sRange = tqdm(range(s.shape[0]))
    else:
        sRange = range(s.shape[0])
    
    samples = []
    for loop in sRange:
        pDict = get_Params_Prospector(s[loop, :], model, sps, sfr_dt=sfr_dt)
        samples.append(list(pDict.values()))
    samples = np.array(samples)
    parnames = list(pDict.keys())
    return samples, parnames


def plot_bin_image(bin_info, highlight_index=None, fig=None, ax=None, 
                   norm_kwargs=None, imshow_kwargs=None):
    '''
    Plot the image and the bin bounaries.

    Parameters
    ----------
    bin_info : dict
        The `bin_info` generated by `binmap_voronoi()`.
    highlight_index (optional) : int
        The bin index to be marked with a cross.
    '''
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    img = bin_info['image']
    segm = bin_info['segm']
    cmap = bin_info['cmap']

    if norm_kwargs is None:
        norm_kwargs = dict(percent=99.99, stretch='asinh', asinh_a=0.001)
    norm = simple_norm(img, **norm_kwargs)

    if imshow_kwargs is None:
        imshow_kwargs = dict(origin='lower', cmap='Greys_r', norm=norm)
    else:
        if 'origin' not in imshow_kwargs:
            imshow_kwargs['origin'] = 'lower'
        
        if 'cmap' not in imshow_kwargs:
            imshow_kwargs['cmap'] = 'Greys_r'

        if 'norm' not in imshow_kwargs:
            imshow_kwargs['norm'] = norm

    ax.imshow(img, **imshow_kwargs)
    plot_segment_contours(segm, cmap=cmap, ax=ax, verbose=False)

    if highlight_index is not None:
        x, y = bin_info['x_bar'][highlight_index], bin_info['y_bar'][highlight_index]
        ax.plot(x, y, marker='x', ms=8, color='r')
        ax.text(0.05, 0.95, f'Bin {highlight_index}', fontsize=16, color='r',
                transform=ax.transAxes, ha='left', va='top')

    ax.set_xlabel(r'$X$ (pixel)', fontsize=24)
    ax.set_ylabel(r'$Y$ (pixel)', fontsize=24)
    return fig, ax


def plot_bin_segm(bin_info, highlight_index=None, fig=None, ax=None, 
                  label_kwargs=None):
    '''
    Plot the bin segmentation maps.
    '''
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    segm = bin_info['segm']
    mask = bin_info['mask']
    cmap = bin_info['cmap']
    x_bar = bin_info['x_bar']
    y_bar = bin_info['y_bar']

    ax.imshow(segm, origin='lower', cmap=cmap)
    plot_mask_contours(mask, ax=ax, color='k', verbose=False)
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
    
    if highlight_index is not None:
        ax.plot(x_bar[highlight_index], y_bar[highlight_index], marker='x', 
                ms=8, color='r')
        ax.text(0.05, 0.95, f'Bin {highlight_index}', fontsize=16, color='r',
                transform=ax.transAxes, ha='left', va='top')

    return fig, ax


def plot_fit_output(bin_info, fit_output, fig=None, axs=None, norm_kwargs=None, 
                    units_x='micron'):
    '''
    Plot the fit output.
    '''
    with open(fit_output['output_name'], 'rb') as f:
        pd = pickle.load(f)
    obs = pd['obs']

    if axs is None:
        fig = plt.figure(figsize=(14, 7))
        ax0 = fig.add_axes([0.1, 0.1, 0.4, 0.8])
        ax1 = fig.add_axes([0.58, 0.35, 0.35, 0.60])
        ax2 = fig.add_axes([0.58, 0.05, 0.35, 0.30])
        ax2.sharex(ax1)

    plot_bin_image(bin_info, highlight_index=fit_output['bin_index'], fig=fig, 
                   ax=ax0, norm_kwargs=norm_kwargs)
    plot_Prospector_SED(fit_output, obs, fig=fig, axs=[ax1, ax2], 
                        units_x=units_x)

    return fig, [ax0, ax1, ax2]


def plot_phys_image(image, bin_info, fig=None, ax=None, norm_kwargs=None, 
                    **imshow_kwargs):
    '''
    Plot the image with physical quantities.
    '''
    segm = bin_info['segm']
    cmap = bin_info['cmap']

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    
    if norm_kwargs is None:
        norm_kwargs = dict(percent=95, stretch='linear')
    norm = simple_norm(image, **norm_kwargs)

    if imshow_kwargs is None:
        imshow_kwargs = dict(origin='lower', cmap='RdYlBu_r', norm=norm)
    else:
        if 'origin' not in imshow_kwargs:
            imshow_kwargs['origin'] = 'lower'
        
        if 'cmap' not in imshow_kwargs:
            imshow_kwargs['cmap'] = 'RdYlBu_r'

        if 'norm' not in imshow_kwargs:
            imshow_kwargs['norm'] = norm

    mp = ax.imshow(image, **imshow_kwargs)
    cbar = fig.colorbar(mp)
    plot_segment_contours(segm, ax=ax, color='gray', verbose=False)

    ax.set_xlabel(r'$X$ (pixel)', fontsize=24)
    ax.set_ylabel(r'$Y$ (pixel)', fontsize=24)
    return fig, ax, cbar


def plot_Prospector_SED(fit_output, obs, fig=None, axs=None, units_x='micron'):
    '''
    Plot the SED of the Prospector fitting results.
    '''
    if axs is None:
        fig = plt.figure(figsize=(7, 7))
        ax1 = fig.add_axes([0.05, 0.3, 0.9, 0.75])
        ax2 = fig.add_axes([0.05, 0.05, 0.9, 0.25])
        ax2.sharex(ax1)
        axs = [ax1, ax2]

    ax = axs[0]
    y_d = obs['maggies']
    y_d_err = obs['maggies_unc']

    phot = fit_output['phot_best']
    spec = fit_output['spec_best']
    x_p = (fit_output['pwave'] * units.angstrom).to(units_x).value
    x_s = (fit_output['swave'] * units.angstrom).to(units_x).value

    ax.plot(x_p, y_d, marker='o', ls='none', color='k', label='Data')
    ax.errorbar(x_p, y_d, yerr=y_d_err, marker='o', ls='none', color='k')
    ax.plot(x_p, phot, linestyle='', marker='s', markersize=10, 
            mec='orange', mew=2, mfc='none', alpha=0.5, label='Model')
    ax.plot(x_s, spec, color='C3', label='Best fit')

    ax.set_xlim(x_p.min() * 0.1, x_p.max() * 5)
    ax.set_ylim(y_d.min() * 0.1, y_d.max() * 5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(r'Flux (maggies)', fontsize=24)
    ax.legend(loc='upper left', fontsize=16, ncols=1, handlelength=1)

    ax = axs[1]
    chi = (y_d - phot) / y_d_err
    ax.plot(x_p, chi, marker='o', ls='none', color='k')
    ax.axhline(y=0, ls='--', color='k')
    ax.set_xlabel(r'Wavelength (micron)', fontsize=24)
    ax.set_ylabel(r'Res.', fontsize=24)
    ax.text(0.03, 0.95, '(data - model) / unc.', fontsize=16, 
            transform=ax.transAxes, va='top', ha='left')
    ax.set_ylim([-4.5, 4.5])
    ax.set_yticks([-3, 0, 3])
    ax.minorticks_on()

    return fig, axs


def plot_Prospector_corner(output, model, sps, show_progress=False, **kwargs):
    '''
    Make the corner plot for the physical parameters from Prospector results.
    '''
    samples, parnames = get_Samples_Prospector(output, model, sps, 
                                               show_progress=show_progress)
    
    if 'labels' not in kwargs:
        kwargs['labels'] = [label_params[k] for k in parnames]
    
    return corner.corner(samples, **kwargs)


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


label_params = {'logmstar': r'$\log\,(M_*/M_\odot)$', 
                'sfr': r'SFR ($M_\odot\,\mathrm{yr}^{-1}$)', 
                'logmdust': r'$\log\,(M_\mathrm{dust}/M_\odot)$', 
                'A_V' : r'$A_V$'
                }
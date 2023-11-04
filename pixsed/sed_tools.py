import numpy as np
import matplotlib as mpl
from tqdm import tqdm
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.units as units
from astropy.visualization import simple_norm
from photutils.segmentation import SegmentationImage

from sedpy.observate import load_filters
from prospect.utils.obsutils import fix_obs
from prospect.models.templates import TemplateLibrary
from prospect.models import SpecModel
from prospect.fitting import lnprobfn, fit_model


from .utils import plot_segment_contours
from .utils_sed import binmap_voronoi, get_Galactic_Alambda

class SED_cube(object):
    '''
    The class of a data cube.
    '''

    def __init__(self, filename) -> None:
        '''
        Parameters
        ----------
        filename : string
            The file name of the SED data cube.

        Notes
        -----
        FIXME: doc
        '''
        self._header = fits.getheader(filename, ext=0)
        self._image = fits.getdata(filename, extname='image')
        self._error = fits.getdata(filename, extname='error')
        self._mask_galaxy = fits.getdata(filename, extname='mask_galaxy').astype(bool)
        
        # Useful information in the header
        self._ra = self._header.get('RA', None)
        self._dec = self._header.get('DEC', None)
        self._pxs = self._header['PSCALE']  # units: arcsec
        
        info = fits.getdata(filename, extname='info')
        info_header = fits.getheader(filename, extname='info')
        self._band = list(info['BAND'])
        self._wavelength = np.array(info['WAVELENGTH']) * units.Unit(info_header['TUNIT2'])
        self._nband = len(self._band)

    def binmap_voronoi(self, target_sn, bin_ref, cvt=True, wvt=True, 
                       sn_func=None, plot=False, fig=None, axs=None, 
                       norm_kwargs=None, label_kwargs=None, interactive=False, 
                       verbose=False):
        '''
        Bin the image with the voronoi method.

        Parameters
        ----------
        ref_band : string
            The name of the reference band.
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
        '''
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
                ax2 = fig.add_axes([0.05, 0.05, 0.9, 0.38])
                axs = [ax0, ax1, ax2]
        
            if interactive:
                fig.canvas.mpl_connect('close_event', on_close)
        
        if isinstance(bin_ref, str):
            assert bin_ref in self._band, f'Cannot find the band ({bin_ref})!'
            idx = self.get_band_index(bin_ref)
            img = self._image[idx]
            err = self._error[idx]
        else:
            assert len(bin_ref) == 2, 'Require to input the image and noise!'
            img, err = bin_ref

        mask = self._mask_galaxy

        self._bin_info, axs = binmap_voronoi(
            image=img, error=err, mask=mask, target_sn=target_sn, 
            pixelsize=self._pxs, cvt=cvt, wvt=wvt, sn_func=sn_func, plot=plot, 
            fig=fig, axs=axs, norm_kwargs=norm_kwargs, 
            label_kwargs=label_kwargs, interactive=False, verbose=verbose)
        
        if plot:
            return axs

    def collect_sed(self, calib_error=None, A_V=0, model='F99', Rv='3.1', 
                    verbose=True):
        '''
        Collect the SEDs. It generates _seds=dict(flux, error). 
        The flux and error have 2 dimensions of (nbands, nbins).

        Parameters
        ----------
        calib_error (optional) : 1D array
            The fraction of flux that will be added in quadrature to the error.
        A_V : float (default: 0)
            The A_V of the target.
        model : {'F99' or 'CCM89'} (default: 'F99')
            The extinction model.
        Rv : string (default: '3.1')
            The Rv of the extinction model.
        '''
        assert getattr(self, '_bin_info', None) is not None, 'Please generate the binned map first!'
        bin_num = self._bin_info['bin_num']
        x_index = self._bin_info['x_index']
        y_index = self._bin_info['y_index']

        flux = np.zeros([self._nband, self._bin_info['nbins']])
        vars = np.zeros([self._nband, self._bin_info['nbins']])
        for b, x, y in zip(bin_num, x_index, y_index):
            flux[:, b] += self._image[:, y, x]
            vars[:, b] += self._error[:, y, x]**2
        error = np.sqrt(vars)

        if calib_error is not None:
            assert len(calib_error) == self._nband, f'The calib_error should have {self._nband} elements!'
            error = np.sqrt(error**2 + (calib_error[:, np.newaxis] * flux)**2)

        if A_V == 0:
            if verbose:
                print('No extinction correction is applied!')
        elif A_V > 0:
            A_lambda = get_Galactic_Alambda(self._wavelength, A_V, model=model, Rv=Rv)
            f_corr = 10**(0.4 * A_lambda[:, np.newaxis])
            flux *= f_corr
            error *= f_corr
        else:
            raise ValueError('A negative A_V is not allowed!')

        self._seds = {
            'flux': flux,
            'error': error
        }

    def gen_averaged_map(self, indices:list, plot=False, ax=None, 
                         norm_kwargs=None):
        '''
        Generate the weight averaged map.
        '''
        imgs = self._image[np.array(indices), :, :]
        ivw = self._error[np.array(indices), :, :]**(-2)

        img_ave, w = np.average(imgs, axis=0, weights=ivw, returned=True)
        err_ave = w**(-0.5)

        if plot:
            if ax is None:
                _, ax = plt.subplots(figsize=(7, 7))

            if norm_kwargs is None:
                norm_kwargs = dict(percent=99.99, stretch='asinh', asinh_a=0.001)
            norm = simple_norm(img_ave, **norm_kwargs)
            ax.imshow(img_ave, origin='lower', cmap='Greys_r', norm=norm)

        return img_ave, err_ave

    def get_band_index(self, band):
        '''
        Get the index of the band in the cube.

        Parameters
        ----------
        band : string
            The band name.
        
        Returns
        -------
        idx : int
            The index of the band.
        '''
        idx = self._band.index(band)
        return idx

    def fit_allsed_prospector(self, bands, redshift, unc_add=0.1, 
                              model_params=None, noise_model=None, sps=None, 
                              optimize=False, emcee=False, dynesty=True, 
                              fitting_kwargs=None, ncpu=1, print_progress=True): 
        '''
        Fit all the binned SEDs.
        '''
        if print_progress:
            rIndex = tqdm(range(self._bin_info['nbins']))
        else:
            rIndex = range(self._bin_info['nbins'])

        func = lambda x : self.fit_sed_prospector(
            x, bands=bands, redshift=redshift, unc_add=unc_add, 
            model_params=model_params, noise_model=noise_model, sps=sps, 
            optimize=optimize, emcee=emcee, dynesty=dynesty, 
            fitting_kwargs=fitting_kwargs, skip_fit=False, print_progress=False, 
            plot=False, verbose=False)

        if ncpu > 1:
            from multiprocess import Pool
            pool = Pool(ncpu)
            pool.map(func, rIndex)
        else:
            for idx in rIndex:
                func(idx)

    def fit_sed_prospector(self, index, bands, redshift, unc_add=0.1, 
                           model_params=None, noise_model=None, sps=None,
                           optimize=False, emcee=False, dynesty=True, 
                           fitting_kwargs=None, skip_fit=False, 
                           print_progress=True, plot=False, fig=None, axs=None, 
                           norm_kwargs=None, verbose=False): 
        '''
        Fit one SED with Prospector.
        '''
        assert getattr(self, '_seds', None) is not None, 'Please run collect_sed first!'

        # Prepare the observation
        filters = load_filters([f'{b}' for b in bands])
        maggies = self._seds['flux'][:, index] / 3631  # converted to maggies
        maggies_unc = self._seds['error'][:, index] / 3631
        maggies_unc = np.sqrt(maggies_unc**2 + (unc_add * maggies)**2)
        obs = dict(wavelength=None, spectrum=None, unc=None, redshift=redshift,
                   maggies=maggies, maggies_unc=maggies_unc, filters=filters)
        obs = fix_obs(obs)

        # Prepare the model
        if model_params is None:
            model_params = TemplateLibrary["parametric_sfh"]
            model_params['mass']['prior'].update(mini=1e4, maxi=1e11)
        model_params["zred"]["init"] = obs["redshift"]  # Always needed?
        model = SpecModel(model_params)

        if noise_model is None:
            noise_model = (None, None)
        
        if sps is None:
            from prospect.sources import CSPSpecBasis
            sps = CSPSpecBasis(zcontinuous=1)

        if skip_fit:
            assert getattr(self, '_fit_output', None) is not None, 'Cannot skip the fit without the _fit_output!'
        else:
            if fitting_kwargs is None:
                fitting_kwargs = dict(nlive_init=400, nested_sample="rwalk", 
                                      nested_target_n_effective=1000, 
                                      nested_dlogz_init=0.05)
            fitting_kwargs['print_progress'] = print_progress

            output = fit_model(
                obs, model, sps, optimize=optimize, emcee=emcee, dynesty=dynesty, 
                lnprobfn=lnprobfn, noise=noise_model, **fitting_kwargs)
            
            results = output['sampling'][0]
            lnprob = results['logl'] + model.prior_product(results.samples, nested=True),
            btheta = results.samples[np.argmax(lnprob), :]
            spec, phot, _ = model.predict(btheta, obs=obs, sps=sps)

            swave = sps.wavelengths * (1 + redshift)
            pwave = np.array([f.wave_effective for f in obs["filters"]])

            if 'add_agn_dust' in model.params:
                theta_noagn = btheta.copy()
                theta_noagn[model.free_params.index('fagn')] = 0
                spec_noagn, _, _ = model.predict(theta_noagn, obs=obs, sps=sps)
                spec_agn = spec - spec_noagn
            else:
                spec_agn = None

            if getattr(self, '_fit_output', None) is None:
                self._fit_output = {}

            self._fit_output[f'bin{index}'] = dict(
                obs = obs,
                sps = sps,
                model = model,
                results = results, 
                best_fit = dict(
                    theta = btheta, 
                    spec = (swave, spec),
                    phot = (pwave, phot),
                    spec_agn = spec_agn
                )
            )

        if plot:
            if axs is None:
                fig = plt.figure(figsize=(14, 7))
                ax0 = fig.add_axes([0.1, 0.1, 0.4, 0.8])
                ax1 = fig.add_axes([0.58, 0.35, 0.35, 0.60])
                ax2 = fig.add_axes([0.58, 0.05, 0.35, 0.30])
                axs = [ax0, ax1, ax2]
                ax2.sharex(ax1)

            ax = axs[0]
            img = self._bin_info['image']
            x = self._bin_info['x_bar'][index]
            y = self._bin_info['y_bar'][index]

            if norm_kwargs is None:
                norm_kwargs = dict(percent=99.99, stretch='asinh', asinh_a=0.001)
            norm = simple_norm(img, **norm_kwargs)

            ax.imshow(img, origin='lower', cmap='Greys_r', norm=norm)
            plot_segment_contours(
                self._bin_info['segm'], cmap=self._bin_info['cmap'], ax=ax, 
                verbose=verbose)
            ax.plot(x, y, marker='x', ms=8, color='r')
            ax.set_xlabel(r'$X$ (pixel)', fontsize=24)
            ax.set_ylabel(r'$Y$ (pixel)', fontsize=24)
            ax.text(0.05, 0.95, f'Bin: {index}', fontsize=18, color='k', 
                    transform=ax.transAxes, va='top', ha='left')

            ax = axs[1]
            pwave, phot = self._fit_output[f'bin{index}']['best_fit']['phot']
            swave, spec = self._fit_output[f'bin{index}']['best_fit']['spec']
            pwave = pwave / 1e4
            swave = swave / 1e4
            ax.plot(pwave, maggies, marker='o', ls='none', color='k', label='Data')
            ax.errorbar(pwave, maggies, yerr=maggies_unc, marker='o', ls='none', 
                        color='k')
            ax.plot(pwave, phot, linestyle='', marker='s', markersize=10, 
                    mec='orange', mew=2, mfc='none', alpha=0.5, label='Model')
            ax.plot(swave, spec, color='C3', label='Best fit')
            
            spec_agn = self._fit_output[f'bin{index}']['best_fit']['spec_agn']
            if spec_agn is not None:
                ax.plot(swave, spec_agn, color='gray', ls='--', label='Torus')

            ax.set_xlim(pwave.min() * 0.1, pwave.max() * 5)
            ax.set_ylim(maggies.min() * 0.1, maggies.max() * 5)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_ylabel(r'Flux (maggies)', fontsize=24)
            ax.legend(loc='upper left', fontsize=16, ncols=1, handlelength=1)

            ax = axs[2]
            chi = (maggies - phot) / maggies_unc
            ax.plot(pwave, chi, marker='o', ls='none', color='k')
            ax.axhline(y=0, ls='--', color='k')
            ax.set_xlabel(r'Wavelength (micron)', fontsize=24)
            ax.set_ylabel(r'Res. ($\sigma$)', fontsize=24)
            ax.set_ylim([-4.5, 4.5])
            ax.set_yticks([-3, 0, 3])
            ax.minorticks_on()

    def plot_sed(self, index, fig=None, axs=None, norm_kwargs=None, 
                 units_w='micron', units_f='Jy', verbose=False):
        '''
        Plot the SED of a selected bin.

        Parameters
        ----------
        index : int
            The bin index to plot.
        fig : Matplotlib Figure
            The figure to plot.
        axs : Matplotlib Axes
            The axes to plot. Two panels are needed.
        norm_kwargs (optional) : dict
            The keywords to normalize the data image.
        units_w : string (default: 'micron')
            The units of the wavelength.
        units_f : string (default: 'Jy')
            The units of the flux.
        verbose : bool (default: False)
            Print information if True.
        '''
        flux = self._seds['flux'][:, index]
        error = self._seds['error'][:, index]
        x = self._bin_info['x_bar'][index]
        y = self._bin_info['y_bar'][index]

        if axs is None:
            fig, axs = plt.subplots(1, 2, figsize=(14, 7))
            fig.subplots_adjust(wspace=0.2)

        ax = axs[0]
        img = self._bin_info['image']
        if norm_kwargs is None:
            norm_kwargs = dict(percent=99.99, stretch='asinh', asinh_a=0.001)
        norm = simple_norm(img, **norm_kwargs)
        ax.imshow(img, origin='lower', cmap='Greys_r', norm=norm)
        plot_segment_contours(
            self._bin_info['segm'], cmap=self._bin_info['cmap'], ax=ax, 
            verbose=verbose)
        ax.plot(x, y, marker='x', ms=8, color='r')
        ax.set_xlabel(r'$X$ (pixel)', fontsize=24)
        ax.set_ylabel(r'$Y$ (pixel)', fontsize=24)
        
        ax = axs[1]
        w = self._wavelength.to(units_w).value
        ax.errorbar(w, flux, yerr=error, marker='o', mec='k', mfc='none', color='k')
        ax.text(0.05, 0.95, f'Bin: {index}', fontsize=18, color='k', 
                transform=ax.transAxes, va='top', ha='left')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(f'Wavelength ({units_w})', fontsize=24)
        ax.set_ylabel(f'Flux ({units_f})', fontsize=24)

    def plot_seds(self, indices:list, fig=None, axs=None, norm_kwargs=None, 
                  units_w='micron', units_f='Jy', verbose=False):
        '''
        Plot a list of SEDs.
        '''
        nseds = len(indices)
        nbins = self._bin_info['nbins']

        if axs is None:
            fig, axs = plt.subplots(nseds, 2, figsize=(10, nseds*5))
            fig.subplots_adjust(wspace=0.3, hspace=0.02)
        assert axs.shape == (nseds, 2), f'Incorrect axs shape ({axs.shape})!'

        ylimList = []
        for loop, idx in enumerate(indices):
            index = idx % nbins
            self.plot_sed(index, fig=fig, axs=axs[loop, :], norm_kwargs=norm_kwargs, 
                          units_w=units_w, units_f=units_f, verbose=verbose)
            ylimList.append(axs[loop, 1].get_ylim())

            if loop > 0:
                axs[loop, 1].sharex(axs[0, 1])
                axs[loop, 1].sharey(axs[0, 1])
            if loop < nseds-1:
                axs[loop, 0].set_xlabel('')
                axs[loop, 1].set_xlabel('')
                axs[loop, 0].set_xticklabels([])
                axs[loop, 1].set_xticklabels([])
        axs[0, 1].set_ylim(np.min(ylimList), np.max(ylimList))
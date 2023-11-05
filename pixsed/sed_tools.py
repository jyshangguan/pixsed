import numpy as np
import matplotlib as mpl
from tqdm import tqdm
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.units as units
from astropy.visualization import simple_norm
from photutils.segmentation import SegmentationImage
import dill as pickle
from pathlib import Path

from sedpy.observate import load_filters
from prospect.utils.obsutils import fix_obs
from prospect.models.templates import TemplateLibrary
from prospect.models import SpecModel
from prospect.fitting import lnprobfn, fit_model


from .utils import plot_segment_contours
from .utils_sed import (plot_bin_image, plot_bin_segm, plot_Prospector_SED, plot_fit_output)
from .utils_sed import (binmap_voronoi, get_Galactic_Alambda, 
                        fit_SED_Prospector, get_Params_Prospector, 
                        get_Samples_Prospector, get_BestFit_Prospector, 
                        gen_image_phys, gen_image_density)
#from prospect.io import write_results as writer


class SED_cube(object):
    '''
    The class of a data cube.
    '''

    def __init__(self, filename, temp_path='./temp') -> None:
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

        # Create the temp path
        self._temp_path = temp_path
        Path(f'{temp_path}').mkdir(parents=True, exist_ok=True)
        
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
                       norm_kwargs=None, label_kwargs={}, interactive=False, 
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

    def gen_bin_phys(self):
        '''
        Generate the list of physical parameter for the bins.
        '''
        nbins = self._bin_info['nbins']
        best_phys = self._fit_outputs['bin0']['phy_params']
        parnames = list(best_phys.keys())
        bin_phys = {}

        for loop in range(nbins):
            if loop == 0:
                for pn in parnames:
                    bin_phys[pn] = [best_phys[pn]]
            else:
                best_phys = self._fit_outputs[f'bin{loop}']['phy_params']
                for pn in parnames:
                    bin_phys[pn].append(best_phys[pn])

        self._bin_phys = bin_phys

    def gen_image_density(self, pixel_to_area=None):
        '''
        Generate the density map with the physical parameters from the SED 
        fitting.
        '''
        if pixel_to_area is None:
            raise NotImplementedError('To be done...')
        
        self._image_density = {}
        for pn in ['logmstar', 'sfr', 'logmdust']:
            self._image_density[pn] = gen_image_density(
                pn, self._bin_info, self._bin_phys, pixel_to_area)

    def gen_image_phys(self):
        '''
        Generate the maps with the physical parameters from the SED fitting.
        '''
        self._image_phys = {}

        for pn in self._bin_phys.keys():
            self._image_phys[pn] = gen_image_phys(pn, self._bin_info, self._bin_phys)

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

    def fit_allsed_prospector(self, bands, redshift, lumdist=None, unc_add=0.1, 
                              model_params=None, noise_model=None, sps=None, 
                              optimize=False, emcee=False, dynesty=True, 
                              fitting_kwargs=None, ncpu=1, print_progress=True,
                              verbose=True, debug=False): 
        '''
        Fit all the binned SEDs.
        '''
        nbins = self._bin_info['nbins']
        if print_progress:
            rIndex = tqdm(range(nbins))
        else:
            rIndex = range(nbins)

        func = lambda x : self.fit_sed_prospector(
            x, bands=bands, redshift=redshift, lumdist=lumdist, unc_add=unc_add, 
            model_params=model_params, noise_model=noise_model, sps=sps, 
            optimize=optimize, emcee=emcee, dynesty=dynesty, 
            fitting_kwargs=fitting_kwargs, print_progress=False, 
            plot=False, verbose=verbose, debug=debug)

        if ncpu > 1:
            from multiprocess import Pool
            pool = Pool(ncpu)
            results = pool.map(func, rIndex)
        else:
            results = [func(idx) for idx in rIndex]

        self._fit_outputs = dict(zip([f'bin{idx}' for idx in range(nbins)], results))

    def fit_sed_prospector(self, index, bands, redshift, lumdist=None, unc_add=0.1, 
                           model_params=None, noise_model=None, sps=None,
                           optimize=False, emcee=False, dynesty=True, 
                           fitting_kwargs=None, print_progress=True, plot=False, 
                           fig=None, axs=None, norm_kwargs=None, 
                           units_x='micron', verbose=False, debug=False): 
        '''
        Fit one SED with Prospector.

        Parameters
        ----------
        index : int
            The bin index.
        bands : list of str
            The bands of the SED.
        redshift : float
            The redshift of the source.
        lumdist (optional) : float
            The luminosity distance of the source, units: Mpc. If not provided, 
            Prospector will use the redshift to calculate the lumdist.
        unc_add : float (default: 0.1)
            The 
        '''
        assert getattr(self, '_seds', None) is not None, 'Please run collect_sed first!'

        if verbose:
            print(f'Fit bin {index}!')

        if debug:
            return index
        
        maggies = self._seds['flux'][:, index] / 3631  # converted to maggies
        maggies_unc = self._seds['error'][:, index] / 3631
        maggies_unc = np.sqrt(maggies_unc**2 + (unc_add * maggies)**2)
        output, obs, model, sps = fit_SED_Prospector(
            bands, maggies=maggies, maggies_unc=maggies_unc, redshift=redshift, 
            lumdist=lumdist, model_params=model_params, noise_model=noise_model, 
            sps=sps, optimize=optimize, emcee=emcee, dynesty=dynesty, 
            fitting_kwargs=fitting_kwargs, print_progress=print_progress)

        btheta = get_BestFit_Prospector(output, model=model)
        spec, phot, _ = model.predict(btheta, obs=obs, sps=sps)

        swave = sps.wavelengths * (1 + redshift)
        pwave = np.array([f.wave_effective for f in obs["filters"]])

        phy_params = get_Params_Prospector(btheta, model=model, sps=sps)
        phy_samples = get_Samples_Prospector(output, model=model, sps=sps)

        # Save fitting results
        output_name = f'{self._temp_path}/temp_bin{index}.dill'
        pd = dict(obs=obs, sps=sps, model=model, output=output, phy_samples=phy_samples)
        with open(output_name, 'wb') as f:
            pickle.dump(pd, f)

        fit_output = dict(
            bin_index = index,
            pwave = pwave,
            swave = swave,
            spec_best = spec,
            phot_best = phot,
            fit_params = dict(zip(model.free_params, btheta)),
            phy_params = phy_params,
            output_name = output_name, 
        )

        if plot:
            plot_fit_output(self._bin_info, fit_output, fig=fig, axs=axs, 
                            norm_kwargs=norm_kwargs, units_x=units_x)
        
        return fit_output

    def plot_sed(self, index, fig=None, axs=None, units_w='micron', 
                 units_f='Jy', label_kwargs={}):
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
            fig.subplots_adjust(wspace=0.25)

        plot_bin_segm(self._bin_info, highlight_index=index, fig=fig, 
                       ax=axs[0], label_kwargs=label_kwargs)
        
        ax = axs[1]
        w = self._wavelength.to(units_w).value
        ax.errorbar(w, flux, yerr=error, marker='o', mec='k', mfc='none', color='k')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(f'Wavelength ({units_w})', fontsize=24)
        ax.set_ylabel(f'Flux ({units_f})', fontsize=24)

    def plot_seds(self, indices:list, fig=None, axs=None, units_w='micron', 
                  units_f='Jy', label_kwargs={}):
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
            self.plot_sed(index, fig=fig, axs=axs[loop, :], units_w=units_w, 
                          units_f=units_f, label_kwargs=label_kwargs)
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


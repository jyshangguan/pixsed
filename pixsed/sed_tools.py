import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.units as units
from astropy.visualization import simple_norm
from photutils.segmentation import SegmentationImage
from .utils import plot_segment_contours, binmap_voronoi

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
        self._pxs = self._header['PSCALE']  # units: arcsec
        
        info = fits.getdata(filename, extname='info')
        info_header = fits.getheader(filename, extname='info')
        self._band = list(info['BAND'])
        self._wavelength = np.array(info['WAVELENGTH']) * units.Unit(info_header['TUNIT2'])
        self._nband = len(self._band)

    def binmap_voronoi(self, ref_band, target_sn, cvt=True, wvt=True, 
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
                fig, axs = plt.subplots(1, 2, figsize=(14, 7))
        
            if interactive:
                fig.canvas.mpl_connect('close_event', on_close)
        
        idx = self.get_band_index(ref_band)
        img = self._image[idx]
        err = self._error[idx]
        mask = self._mask_galaxy

        self._segm, self._bin_info = binmap_voronoi(
            image=img, error=err, mask=mask, target_sn=target_sn, 
            pixelsize=self._pxs, cvt=cvt, wvt=wvt, sn_func=sn_func, plot=plot, 
            fig=fig, axs=axs, norm_kwargs=norm_kwargs, 
            label_kwargs=label_kwargs, interactive=False, verbose=verbose)

        axs[0].text(0.05, 0.95, ref_band, fontsize=18, color='w', 
                    transform=axs[0].transAxes, va='top', ha='left')

    def collect_sed(self, calib_error=None):
        '''
        Collect the SEDs. It generates _seds=dict(flux, error). 
        The flux and error have 2 dimensions of (nbands, nbins).

        Parameters
        ----------
        calib_error (optional) : 1D array
            The fraction of flux that will be added in quadrature to the error.
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

        self._seds = {
            'flux': flux,
            'error': error
        }

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
        ref_band = self._bin_info['ref_band']
        idx = self.get_band_index(ref_band)
        img = self._image[idx]
        if norm_kwargs is None:
            norm_kwargs = dict(percent=99.99, stretch='asinh', asinh_a=0.001)
        norm = simple_norm(img, **norm_kwargs)
        ax.imshow(img, origin='lower', cmap='Greys_r', norm=norm)
        plot_segment_contours(self._segm, ax=ax, verbose=verbose)
        ax.plot(x, y, marker='x', ms=8, color='r')
        ax.set_xlabel(r'$X$ (pixel)', fontsize=24)
        ax.set_ylabel(r'$Y$ (pixel)', fontsize=24)
        ax.text(0.05, 0.95, ref_band, fontsize=18, color='w', 
                transform=ax.transAxes, va='top', ha='left')
        
        ax = axs[1]
        w = self._wavelength.to(units_w).value
        ax.errorbar(w, flux, yerr=error, marker='o', mec='k', mfc='none', color='k')
        ax.text(0.05, 0.95, f'Bin: {index}', fontsize=18, color='k', 
                transform=ax.transAxes, va='top', ha='left')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(f'Wavelength ({units_w})', fontsize=24)
        ax.set_ylabel(f'Flux ({units_f})', fontsize=24)

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
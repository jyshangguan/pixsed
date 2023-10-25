import math

import tqdm
import random
from math import log, sqrt, ceil, log10
import os
from copy import deepcopy
from datetime import datetime

import astropy.units as units
import extinction
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.convolution import convolve
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.table import Table
from astropy.visualization import simple_norm
from astropy.wcs import WCS
from astropy.modeling import models, fitting
from astropy.io import fits
from astropy.nddata import NDData
from astroquery.xmatch import XMatch
from photutils.psf import EPSFBuilder, extract_stars
from photutils.aperture import CircularAperture
from photutils.background import Background2D, MedianBackground, MeanBackground, SExtractorBackground
from photutils.detection import DAOStarFinder
from photutils.profiles import RadialProfile, CurveOfGrowth
from photutils.segmentation import (detect_sources, make_2dgaussian_kernel,
                                    deblend_sources, SourceCatalog, SourceFinder,
                                    SegmentationImage)
from photutils.aperture import (EllipticalAperture, EllipticalAnnulus,
                                ApertureStats, aperture_photometry)

from reproject import reproject_interp, reproject_adaptive
from scipy import interpolate

from shapely.geometry import shape
from .utils import read_coordinate, plot_image, circular_error_estimate
from .utils import xmatch_gaiadr3, plot_mask_contours
from .utils import get_image_segmentation, gen_image_mask, detect_source_extended
from .utils import scale_mask, get_mask_polygons, add_mask_circle, adapt_mask, adapt_segmentation
from .utils import select_segment_stars, segment_combine, segment_remove
from .utils import clean_header_string, add_mask_circle, add_mask_rect
from .utils import gen_images_matched, image_photometry
from .utils import gen_random_apertures, gen_aperture_ellipse
from .utils_interactive import MaskBuilder_segment, MaskBuilder_draw

mpl.rc("xtick", direction="in", labelsize=16)
mpl.rc("ytick", direction="in", labelsize=16)
mpl.rc("xtick.major", width=1., size=8)
mpl.rc("ytick.major", width=1., size=8)
mpl.rc("xtick.minor", width=1., size=5)
mpl.rc("ytick.minor", width=1., size=5)


class Image(object):
    """
    The class of an image. For the moment, we only assume that there is one
    science target in the image.
    """

    def __init__(self, filename=None, data=None, header=None, coord_sky=None,
                 coord_pix=None, pixel_scale=None, target=None, telescope=None,
                 band=None, verbose=True):
        """
        Parameters
        ----------
        data : numpy 2D array
            The 2D image data.
        header : astropy fits header
            The header of the image data.
        psf_fwhm : float.
            The fwhm of the PSF. asec
        target_coordinate : Tuple. Contain two floats.
            The Ra and Dec of the target galaxy.
        wavelength: float.
            The wavelength of this image. um
        filter_name: string.
            The filter name of this image.
        telescope_name: string.
            The telescope name of this image.
        """
        if filename is None:
            self._data = data.copy()
            self._header = header.copy()

            if pixel_scale is None:
                self._pxs = np.abs(WCS(header).wcs.cdelt[0]) * 3600

        else:
            hdul = fits.open(filename)
            header = hdul[0].header
            self._header = header
            extList = [h.name.lower() for h in hdul]

            if header.get('PIXSED', False):
                self._ra_deg, self._dec_deg = header['RA'], header['DEC']
                self._coord_pix = (header['XCOORD'], header['YCOORD'])
                self._pxs = header['PSCALE']

                if verbose:
                    print('Load reduced data!')

                for ext in extList:
                    if ext == 'psf_data':
                        continue

                    d = hdul[ext].data

                    if 'mask' in ext:
                        d = d.astype('bool')

                    if 'segm' in ext:
                        d = SegmentationImage(d.astype(int))

                    setattr(self, f'_{ext}', d)
                    if verbose:
                        print(f'[__init__] Load {ext}')

                if 'psf_data' in extList:
                    self._psf_data = hdul['psf_data'].data
                    self._psf_fwhm = hdul['psf_data'].header['FWHM']
                    self._psf_fwhm_pix = self._psf_fwhm / self._pxs
                    self._psf_enclose_radius = hdul['psf_data'].header['ENRADIUS']
                    self._psf_enclose_radius_pix = self._psf_enclose_radius / self._pxs
                    self._psf_oversample = hdul['psf_data'].header['OVERSAMP']

            else:
                self._data = hdul[0].data

                if pixel_scale is None:
                    self._pxs = np.abs(WCS(header).wcs.cdelt[0]) * 3600

        if hasattr(self, '_data'):
            self._shape = self._data.shape
        else:
            self._shape = self._data_clean.shape
        self._wcs = WCS(header)

        if target is None:
            self._target = clean_header_string(self._header.get('TARGET', None))
        else:
            self._target = target

        if telescope is None:
            self._telescope = clean_header_string(self._header.get('TELESCOP', None))
        else:
            self._telescope = telescope

        if band is None:
            self._band = clean_header_string(self._header.get('FILTER', None))
        else:
            self._band = band

        # Set coordinates
        if coord_sky is not None:
            c_sky = read_coordinate(coord_sky[0], coord_sky[1])
            self._ra_deg = c_sky.ra.deg
            self._dec_deg = c_sky.dec.deg

        if coord_pix is not None:
            self._coord_pix = tuple(coord_pix)

        if not hasattr(self, '_coord_pix'):
            if (getattr(self, '_ra_deg', None) is not None):
                c_sky = read_coordinate(self._ra_deg, self._dec_deg)
                ra_pix, dec_pix = self._wcs.world_to_pixel(c_sky)
                self._coord_pix = (float(ra_pix), float(dec_pix))
            else:
                if verbose:
                    print('[__init__] Please specify the target position!')

    def adapt_segment(self, segm_name, segm=None, filename=None, extension=1,
                      plot=False, fig=None, axs=None, norm_kwargs=None,
                      interactive=False, verbose=False):
        '''
        Add a segment to the segment attribute of the object. The segment can be directly
        input or read from a FITS file.

        Parameters
        ----------
        segm_name : string
            The segm name of the object.
        segm (optional) : SegmentationImage
            The added segmentation.
        filename (optional) : string
            The FITS file name to load the data.
        extension : int (default: 1)
            The extension that saves the mask.
        plot : bool (default: False)
            Plot the image and cleaned image if True.
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
        '''
        snList = ['segment_inner', 'segment_edge', 'segment_outer',
                  'segm_background']
        assert segm_name in snList, f'Cannot recognize the mask name ({segm_name})!'

        shape_out = self._data.shape

        if segm is None:
            assert filename is not None, 'The input segment is lacking!'

            hdul = fits.open(filename)
            data = hdul[extension].data
            input_wcs = WCS(hdul[extension].header)

            segm = adapt_segmentation(SegmentationImage(data),
                                      input_wcs=input_wcs, output_wcs=self._wcs,
                                      shape_out=shape_out)
        else:
            assert (segm.shape[0] == shape_out[0]) & (segm.shape[1] == shape_out[1])

        # Update the mask
        setattr(self, f'_{segm_name}', segm)

        if plot:
            if interactive:
                ipy = get_ipython()
                ipy.run_line_magic('matplotlib', 'tk')

                def on_close(event):
                    ipy.run_line_magic('matplotlib', 'inline')

            if axs is None:
                fig, axs = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
                fig.subplots_adjust(wspace=0.05, hspace=0.1)
            else:
                assert fig is not None, 'Please provide fig together with axs!'

            if interactive:
                fig.canvas.mpl_connect('close_event', on_close)

            if norm_kwargs is None:
                norm_kwargs = dict(percent=99.99, stretch='asinh', asinh_a=0.001)

            ax = axs[0]
            norm = simple_norm(self._data, **norm_kwargs)
            ax.imshow(self._data, origin='lower', cmap='Greys_r', norm=norm)
            xlim = ax.get_xlim();
            ylim = ax.get_ylim()

            mask = segm.data > 0
            plot_mask_contours(mask, verbose=verbose, ax=ax, color='cyan', lw=0.5)
            ax.set_xlim(xlim);
            ax.set_ylim(ylim)
            ax.set_title('(a) Original image', fontsize=18)

            ax = axs[1]
            ax.imshow(segm, origin='lower', cmap=segm.cmap, interpolation='nearest')
            ax.set_title('(b) Segmentation', fontsize=18)

    def adapt_to(self, other, plot=False, fig=None, axs=None, norm_kwargs=None,
                 interactive=False, verbose=False):
        '''
        Adapt the masks and segmentations from another Image object.

        Parameters
        ----------

        Notes
        -----
        FIXME: doc!
        '''
        input_wcs = other._wcs
        output_wcs = self._wcs
        shape_out = self._shape

        if hasattr(other, '_mask_background'):
            self._mask_background = adapt_mask(other._mask_background,
                                               input_wcs=input_wcs,
                                               output_wcs=output_wcs,
                                               shape_out=shape_out)

            if verbose:
                print('Adapted the background mask')

        if hasattr(other, '_mask_contaminant'):
            self._mask_contaminant = adapt_mask(other._mask_contaminant,
                                                input_wcs=input_wcs,
                                                output_wcs=output_wcs,
                                                shape_out=shape_out)

            if verbose:
                print('Adapted the contaminant mask')

        if hasattr(other, '_mask_galaxy'):
            self._mask_galaxy = adapt_mask(other._mask_galaxy,
                                           input_wcs=input_wcs,
                                           output_wcs=output_wcs,
                                           shape_out=shape_out)

            if verbose:
                print('Adapted the galaxy outer mask')

        if hasattr(other, '_mask_galaxy_inner'):
            self._mask_galaxy_inner = adapt_mask(other._mask_galaxy_inner,
                                                 input_wcs=input_wcs,
                                                 output_wcs=output_wcs,
                                                 shape_out=shape_out)

            if verbose:
                print('Adapted the galaxy inner mask')
        
        if hasattr(other, '_mask_manual'):
            self._mask_manual = adapt_mask(other._mask_manual,
                                           input_wcs=input_wcs,
                                           output_wcs=output_wcs,
                                           shape_out=shape_out)

            if verbose:
                print('Adapted the manual mask')

        if hasattr(other, '_segment_inner'):
            self._segment_inner = adapt_segmentation(other._segment_inner,
                                                     input_wcs=input_wcs,
                                                     output_wcs=output_wcs,
                                                     shape_out=shape_out)

            if verbose:
                print('Adapted the inner segmentation')

        if hasattr(other, '_segment_edge'):
            self._segment_edge = adapt_segmentation(other._segment_edge,
                                                    input_wcs=input_wcs,
                                                    output_wcs=output_wcs,
                                                    shape_out=shape_out)

            if verbose:
                print('Adapted the edge segmentation')

        if hasattr(other, '_segment_outer'):
            self._segment_outer = adapt_segmentation(other._segment_outer,
                                                     input_wcs=input_wcs,
                                                     output_wcs=output_wcs,
                                                     shape_out=shape_out)

            if verbose:
                print('Adapted the outer segmentation')

        if plot:
            self.plot_summary(fig=fig, axs=axs, norm_kwargs=norm_kwargs,
                              interactive=interactive, verbose=verbose)

    def add_mask(self, mask_name, mask_a=None, expand_factor=1, filename=None,
                 extension=1, plot=False, fig=None, axs=None, norm_kwargs=None,
                 interactive=False, verbose=False):
        '''
        Add a mask to the mask attribute of the object. The mask can be directly
        input or read from a FITS file.

        Parameters
        ----------
        mask_name : string
            The mask name of the object.
        mask_a (optional) : 2D array
            The added mask.
        filename (optional) : string
            The FITS file name to load the data.
        extension : int (default: 1)
            The extension that saves the mask.
        plot : bool (default: False)
            Plot the image and cleaned image if True.
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
        '''
        mnList = ['mask_background', 'mask_contaminant', 'mask_galaxy',
                  'mask_galaxy_inner', 'mask_inner', 'mask_edge',
                  'mask_outer']
        assert mask_name in mnList, f'Cannot recognize the mask name ({mask_name})!'

        if not hasattr(self, mask_name):
            mask_o = np.zeros_like(self._data, dtype=bool)
        else:
            mask_o = getattr(self, f'_{mask_name}')

        if mask_a is None:
            assert filename is not None, 'The input mask is lacking!'

            hdul = fits.open(filename)
            data = hdul[extension].data
            input_wcs = WCS(hdul[extension].header)

            mask_a = adapt_mask(data, input_wcs=input_wcs, output_wcs=self._wcs,
                                shape_out=mask_o.shape)

        if expand_factor > 1:
            mask_a = scale_mask(mask_a, expand_factor)

        # Update the mask
        mask = mask_o | mask_a
        setattr(self, f'_{mask_name}', mask)

        if plot:
            if interactive:
                ipy = get_ipython()
                ipy.run_line_magic('matplotlib', 'tk')

                def on_close(event):
                    ipy.run_line_magic('matplotlib', 'inline')

            if axs is None:
                fig, axs = plt.subplots(1, 3, figsize=(21, 7), sharex=True, sharey=True)
                fig.subplots_adjust(wspace=0.05, hspace=0.1)
            else:
                assert fig is not None, 'Please provide fig together with axs!'

            if interactive:
                fig.canvas.mpl_connect('close_event', on_close)

            if norm_kwargs is None:
                norm_kwargs = dict(percent=99.99, stretch='asinh', asinh_a=0.001)

            ax = axs[0]
            norm = simple_norm(self._data, **norm_kwargs)
            ax.imshow(self._data, origin='lower', cmap='Greys_r', norm=norm)
            xlim = ax.get_xlim();
            ylim = ax.get_ylim()

            plot_mask_contours(mask, verbose=verbose, ax=ax, color='cyan', lw=0.5)
            ax.set_xlim(xlim);
            ax.set_ylim(ylim)
            ax.set_title('(a) Original image', fontsize=18)

            ax = axs[1]
            ax.imshow(mask_o, origin='lower', cmap='Greys_r')
            ax.set_title('(b) Original mask', fontsize=18)

            ax = axs[2]
            ax.imshow(mask_a, origin='lower', cmap='Greys_r')
            ax.set_title('(c) Added contaminant mask', fontsize=18)

    def background_properties(self, mask_type='quick', sigma=3, maxiters=5, f_sample=1, **kwargs):
        '''
        Calculate the mean, median, and std of the background using sigma clip.

        Parameters
        ----------
        mask_type : string (default: 'quick')
            If 'quick', directly use the sigma clip to get a quick statistics.
            If 'segmentation', use the mask generated by image segmentation to mask the source
            emssion. This is a quick method, so the function will
            continue even if the mask is None.
            If 'full', use the carefully generated masks of the target and
            background. The masks must be generated in advance.
        sigma : float (default: 3)
            Sigma clip threshold.
        maxiters : int (default: 5)
            Maximum iterations of the sigma clip.
        f_sample : float (default: -1)
            Use a fraction of the unmasked data to calculate the sigma clip
        **kwargs : Other parameters of sigma_clipped_stats()

        Returns
        -------
        mean, median, stddev : floats
            The returns of sigma_clipped_stats().
        '''
        if mask_type == 'quick':
            if hasattr(self, '_mask_field'):
                data = self._data[~self._mask_coverage]
            else:
                data = self._data.flatten()

        elif mask_type == 'background':
            if not hasattr(self, '_mask_background'):
                raise ValueError(
                    'The background mask (_mask_background) is not generated! Please run mask_background()!')
            mask = ~self._mask_background

            if hasattr(self, '_mask_coverage'):
                mask &= ~self._mask_coverage

            data = self._data[mask]

        elif mask_type == 'subbkg':
            if not hasattr(self, '_mask_background'):
                raise ValueError(
                    'The background mask (_mask_background) is not generated! Please run mask_background()!')

            if not hasattr(self, '_data_subbkg'):
                raise ValueError(
                    'The background-subtracted data (_data_subbkg) is not generated! Please run background_subtract2()!')
            mask = ~self._mask_background

            if hasattr(self, '_mask_coverage'):
                mask &= ~self._mask_coverage

            data = self._data_subbkg[mask]

        else:
            raise ValueError(f'The mask_type ({mask_type}) is not recognized!')

        if f_sample < 1:
            assert f_sample > 0, f'f_sample ({f_sample}) should be 0 to 1!'
            data = np.random.choice(data, size=int(f_sample * len(data)), replace=False)

        res = sigma_clipped_stats(data, sigma=sigma, maxiters=maxiters, **kwargs)
        self._bkg_mean, self._bkg_median, self._bkg_std = res
        return self._bkg_mean, self._bkg_median, self._bkg_std

    def background_subtract2(self, plot=False, norm_kwargs=None):
        '''
        Remove the background of the image data.
        Parameter
        ----------
        plot : bool (default: False)
            Plot the data and segmentation map if True.
        norm_kwargs (optional) : dict
            The keywords to normalize the data image.

        Notes
        -----
        [SGJY added]
        '''
        assert hasattr(self, '_model_background'), 'Please run background_model() first!'
        self._data_subbkg = self._data - self._model_background

        if plot:
            fig, axs = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)

            if norm_kwargs is None:
                norm_kwargs = dict(percent=90, stretch='asinh', asinh_a=0.1)

            ax = axs[0]
            norm = simple_norm(self._data, **norm_kwargs)
            ax.imshow(self._data, cmap='Greys_r', origin='lower', norm=norm)
            ax.set_title('Original image', fontsize=16)

            ax = axs[1]
            norm = simple_norm(self._data_subbkg, **norm_kwargs)
            ax.imshow(self._data_subbkg, cmap='Greys_r', origin='lower', norm=norm)
            ax.set_title('Background subtracted', fontsize=16)

    def detect_source_outer_and_middle(self, threshold_o: float, threshold_i: float, threshold_inner_galaxy: float,
                                       npixels_o=5, npixels_i=5, nlevels_o=32,
                                       nlevel_i=256, contrast_o=0.001, contrast_i=1e-6,
                                       connectivity=8, kernel_fwhm=0, mode='linear',
                                       nproc=1, progress_bar=False, plot=False,
                                       fig=None, axs=None, norm_kwargs=None,
                                       interactive=False, verbose=False):
        '''
        Detect the image sources for the extended target. This function get
        the segmentations of the image in two steps, one inside the target_mask
        and one outside the target_mask.

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
            Threshold to generate the segmentation between the target mask and inner target mask.
        threshold_inner_galaxy: float
            Threshold to generate the segmentation of inner galaxy.
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
        if hasattr(self, '_data_convolved'):
            image = self._data_convolved
            image_name = 'Convolved data'
        elif hasattr(self, '_data_subbkg'):
            image = self._data_subbkg
            image_name = 'Background subtracted data'
        else:
            raise ValueError('Cannot find neither _data_convolved nor _data_subbkg!')

        assert hasattr(self, '_mask_galaxy'), 'The target mask is lacking!'

        # if hasattr(self, '_mask_field'):
        #    coverage_mask = ~self._mask_field
        # else:
        #    coverage_mask = None
        coverage_mask = getattr(self, '_mask_coverage', None)

        res = detect_source_extended(image, self._coord_pix,
                                     self._mask_galaxy, threshold_o=threshold_o,
                                     threshold_i=threshold_inner_galaxy,
                                     npixels_o=npixels_o,
                                     npixels_i=12, nlevels_o=nlevels_o,
                                     nlevel_i=1, contrast_o=contrast_o,
                                     contrast_i=1,
                                     coverage_mask=coverage_mask,
                                     connectivity=connectivity,
                                     kernel_fwhm=kernel_fwhm, mode=mode,
                                     nproc=nproc, progress_bar=progress_bar,
                                     plot=False, verbose=verbose)
        new_mask = res['mask_in'] | ~self._mask_galaxy
        self._mask_galaxy_inner = res['mask_in']
        self._segment_outer = res['segment_out']
        new_res = get_image_segmentation(image, threshold=threshold_i, npixels=npixels_i, mask=new_mask, connectivity=8,
                                         kernel_fwhm=kernel_fwhm, deblend=True, nlevels=nlevel_i, contrast=contrast_i,
                                         mode='linear', nproc=1, progress_bar=True,
                                         plot=False, axs=None, norm_kwargs=None,
                                         interactive=False)
        self._segment_edge = new_res[0]

        # Remove the segments in the inner region of the target
        segment_remove(self._segment_edge, self._mask_galaxy_inner, overwrite=True)

        if plot:
            if interactive:
                ipy = get_ipython()
                ipy.run_line_magic('matplotlib', 'tk')

                def on_close(event):
                    ipy.run_line_magic('matplotlib', 'inline')

            if axs is None:
                fig, axs = plt.subplots(1, 3, figsize=(21, 17), sharex=True, sharey=True)
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
            ax.plot(self._coord_pix[0], self._coord_pix[1], marker='+', ms=10, color='red')
            xlim = ax.get_xlim();
            ylim = ax.get_ylim()
            plot_mask_contours(self._mask_galaxy, ax=ax, verbose=verbose, color='cyan', lw=0.5)
            plot_mask_contours(self._mask_galaxy_inner, ax=ax, verbose=verbose, color='magenta', lw=0.5)

            ax.set_xlim(xlim);
            ax.set_ylim(ylim)
            ax.set_title(image_name, fontsize=18)

            ax = axs[1]
            ax.imshow(self._segment_edge, origin='lower', cmap=self._segment_edge.cmap, interpolation='nearest')
            xlim = ax.get_xlim();
            ylim = ax.get_ylim()
            ax.set_xlim(xlim);
            ax.set_ylim(ylim)
            ax.set_title('Middle segmentation', fontsize=18)
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            ax = axs[2]
            ax.imshow(self._segment_outer, origin='lower', cmap=self._segment_outer.cmap, interpolation='nearest')
            xlim = ax.get_xlim();
            ylim = ax.get_ylim()
            ax.set_xlim(xlim);
            ax.set_ylim(ylim)
            ax.set_title('Outer segmentation', fontsize=18)
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    def detect_source_extended(self, threshold_o:float, threshold_i:float, 
                               npixels_o=5, npixels_i=5, nlevels_o=32, 
                               nlevel_i=256, contrast_o=0.001, contrast_i=1e-6, 
                               connectivity=8, kernel_fwhm=0, mode='linear', 
                               nproc=1, progress_bar=False, plot=False, 
                               fig=None, axs=None, norm_kwargs=None, 
                               interactive=False, verbose=False):
        '''
        Detect the image sources for the extended target. This function get 
        the segmentations of the image in two steps, one inside the target_mask 
        and one outside the target_mask.

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
        if hasattr(self, '_data_convolved'):
            image = self._data_convolved
            image_name = 'Convolved data'
        elif hasattr(self, '_data_subbkg'):
            image = self._data_subbkg
            image_name = 'Background subtracted data'
        else:
            raise ValueError('Cannot find neither _data_convolved nor _data_subbkg!')

        assert hasattr(self, '_mask_galaxy'), 'The target mask is lacking!'

        #if hasattr(self, '_mask_field'):
        #    coverage_mask = ~self._mask_field
        #else:
        #    coverage_mask = None
        coverage_mask = getattr(self, '_mask_coverage', None)

        res = detect_source_extended(image, self._coord_pix, 
                                     self._mask_galaxy, threshold_o=threshold_o, 
                                     threshold_i=threshold_i, 
                                     npixels_o=npixels_o, 
                                     npixels_i=npixels_i, nlevels_o=nlevels_o, 
                                     nlevel_i=nlevel_i, contrast_o=contrast_o, 
                                     contrast_i=contrast_i, 
                                     coverage_mask=coverage_mask,
                                     connectivity=connectivity, 
                                     kernel_fwhm=kernel_fwhm, mode=mode, 
                                     nproc=nproc, progress_bar=progress_bar, 
                                     plot=False, verbose=verbose)
        self._mask_galaxy_inner = res['mask_in']
        self._segment_outer = res['segment_out']
        self._segment_edge = res['segment_in']
        
        # Remove the segments in the inner region of the target
        segment_remove(self._segment_edge, self._mask_galaxy_inner, overwrite=True)
    
        if plot:
            if interactive:
                ipy = get_ipython()
                ipy.run_line_magic('matplotlib', 'tk')

                def on_close(event):
                    ipy.run_line_magic('matplotlib', 'inline')

            if axs is None:
                fig, axs = plt.subplots(1, 3, figsize=(21, 17), sharex=True, sharey=True)
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
            ax.plot(self._coord_pix[0], self._coord_pix[1], marker='+', ms=10, color='red')
            xlim = ax.get_xlim(); ylim = ax.get_ylim()
            plot_mask_contours(self._mask_galaxy, ax=ax, verbose=verbose, color='cyan', lw=0.5)
            plot_mask_contours(self._mask_galaxy_inner, ax=ax, verbose=verbose, color='magenta', lw=0.5)

            ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.set_title(image_name, fontsize=18)

            ax = axs[1]
            ax.imshow(self._segment_edge, origin='lower', cmap=self._segment_edge.cmap, interpolation='nearest')
            xlim = ax.get_xlim(); ylim = ax.get_ylim()
            ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.set_title('Middle segmentation', fontsize=18)
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            ax = axs[2]
            ax.imshow(self._segment_outer, origin='lower', cmap=self._segment_outer.cmap, interpolation='nearest')
            xlim = ax.get_xlim(); ylim = ax.get_ylim()
            ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.set_title('Outer segmentation', fontsize=18)
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    def gen_image_clean(self, plot=False, fig=None, axs=None, norm_kwargs=None,
                        interactive=False, zoom=None, verbose=False):
        '''
        Replace the masked region with the galaxy model.

        Parameters
        ----------
        plot : bool (default: False)
            Plot the image and cleaned image if True.
        galaxy_model: bool (default: True)
            Use galaxy model to do a better interpolation.
        fig : Matplotlib Figure
            The figure to plot.
        axs : Matplotlib Axes
            The axes to plot. Two panels are needed.
        norm_kwargs (optional) : dict
            The keywords to normalize the data image.
        interactive : bool (default: False)
            Use the interactive plot if True.
        zoom : list
            The range to zoom in the figure, xmin, xmax, ymin, ymax.
        verbose : bool (default: True)
            Show details if True.
        '''
        self._data_clean = self._data_subbkg.copy()
        mask_in = self._mask_inner | self._mask_edge
        mask_galaxy = self._mask_galaxy
        mask_out = self._mask_outer ^ (mask_galaxy & self._mask_outer)

        self._data_clean[mask_in] = self._model_galaxy[mask_in] + \
                                    self._model_galaxy_rms[mask_in] * \
                                    np.random.randn(np.sum(mask_in))
        self._data_clean[mask_out] = np.zeros(np.shape(self._data_clean))[mask_out] + \
                                     self._model_galaxy_rms[mask_out] * \
                                     np.random.randn(np.sum(mask_out))

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
            norm = simple_norm(self._data_subbkg, **norm_kwargs)

            ax = axs[0]
            ax.imshow(self._data_subbkg, origin='lower', cmap='Greys_r', norm=norm)
            xlim = ax.get_xlim();
            ylim = ax.get_ylim()
            plot_mask_contours(self._mask_contaminant, ax=ax, verbose=verbose, color='cyan', lw=0.5)
            ax.set_xlim(xlim);
            ax.set_ylim(ylim)
            ax.set_title('Image', fontsize=18)

            ax = axs[1]
            ax.imshow(self._data_clean, origin='lower', cmap='Greys_r', norm=norm)
            xlim = ax.get_xlim();
            ylim = ax.get_ylim()
            ax.set_xlim(xlim);
            ax.set_ylim(ylim)
            ax.set_title('Cleaned image', fontsize=18)

            if zoom is not None:
                ax.set_xlim([zoom[0], zoom[1]])
                ax.set_ylim([zoom[2], zoom[3]])

    def gen_mask_background(self, threshold, npixels=12, mask=None, connectivity=8,
                            kernel_fwhm=0, deblend=False, nlevels=32,
                            contrast=0.001, mode='linear', nproc=1,
                            progress_bar=False, expand_factor=1.2, plot=False,
                            fig=None, axs=None, norm_kwargs=None,
                            interactive=False, verbose=True):
        '''
        Generate the mask of the background.

        Parameters
        ----------
        threshold : float
            Threshold of image segmentation.
        npixels : int
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
        # _, med, std = sigma_clipped_stats(self._data, mask=mask, sigma=sigma)
        assert self._bkg_median is not None, 'Please run background_properties() first to get _bkg_median!'
        assert self._bkg_std is not None, 'Please run background_properties() first to get _bkg_std!'

        if hasattr(self, '_mask_coverage'):
            if mask is None:
                mask = self._mask_coverage
            else:
                mask &= self._mask_coverage

        img_sub = self._data - self._bkg_median
        mask, segm, _ = gen_image_mask(img_sub, threshold, npixels=npixels,
                                       mask=mask, connectivity=connectivity,
                                       kernel_fwhm=kernel_fwhm, deblend=deblend,
                                       nlevels=nlevels, contrast=contrast,
                                       mode=mode, nproc=nproc,
                                       progress_bar=progress_bar,
                                       expand_factor=expand_factor, bounds=None,
                                       choose_coord=None, plot=plot, fig=fig,
                                       axs=axs, norm_kwargs=norm_kwargs,
                                       interactive=interactive, verbose=verbose)
        self._mask_background = mask
        self._segm_background = segm

    def gen_mask_contaminant(self, expand_inner=1, expand_edge=1.2,
                             expand_outer=1.2, expand_manual=1., plot=False, 
                             fig=None, axs=None, norm_kwargs=None, 
                             interactive=False, verbose=False):
        '''
        Generate the mask of all contaminants.

        Parameters
        ----------
        expand_inner : float (default: 1)
            The expand_factor of the inner mask.
        expand_edge : float (default: 1.2)
            The expand_factor of the edge mask.
        expand_outer : float (default: 1.2)
            The expand_factor of the outer mask.
        expand_manual : float (default: 1)
            The expand_factor of the manual mask.
        plot : bool (default: False)
            Plot the image and mask if True.
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
        self.gen_mask_inner(expand_factor=expand_inner, plot=False, verbose=verbose)
        self.gen_mask_edge(expand_factor=expand_edge, plot=False, verbose=verbose)
        self.gen_mask_outer(expand_factor=expand_outer, plot=False, verbose=verbose)

        self._mask_contaminant = self._mask_inner | self._mask_edge | self._mask_outer

        # Include the manual mask if it exists
        mask_manual = getattr(self, '_mask_manual', None)
        if mask_manual is not None:
            if expand_manual > 1:
                mask_manual = scale_mask(mask_manual, factor=expand_manual)
            self._mask_contaminant |= mask_manual 

        if plot:
            if interactive:
                ipy = get_ipython()
                ipy.run_line_magic('matplotlib', 'tk')

                def on_close(event):
                    ipy.run_line_magic('matplotlib', 'inline')

            if axs is None:
                fig, axs = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
                fig.subplots_adjust(wspace=0.05, hspace=0.1)
            else:
                assert fig is not None, 'Please provide fig together with axs!'

            if interactive:
                fig.canvas.mpl_connect('close_event', on_close)

            if norm_kwargs is None:
                norm_kwargs = dict(percent=99.99, stretch='asinh', asinh_a=0.001)
            norm = simple_norm(self._data_subbkg, **norm_kwargs)

            ax = axs[0]
            ax.imshow(self._data_subbkg, origin='lower', cmap='Greys_r', norm=norm)
            ax.set_title('Image', fontsize=18)
            xlim = ax.get_xlim();
            ylim = ax.get_ylim()
            plot_mask_contours(self._mask_contaminant, ax=ax, verbose=verbose, color='cyan', lw=0.5)

            ax = axs[1]
            ax.imshow(self._mask_contaminant, origin='lower', cmap='Greys_r')
            ax.set_title('Contaminant mask', fontsize=18)
            plot_mask_contours(self._mask_outer, ax=ax, verbose=verbose, color='C0', lw=0.5)
            plot_mask_contours(self._mask_edge, ax=ax, verbose=verbose, color='C1', lw=0.5)
            plot_mask_contours(self._mask_inner, ax=ax, verbose=verbose, color='C2', lw=0.5)
            ax.set_xlim(xlim);
            ax.set_ylim(ylim)

    def gen_mask_edge(self, expand_factor=1.2, plot=False, fig=None,
                      axs=None, norm_kwargs=None, interactive=False,
                      verbose=False):
        '''
        Generate the mask of the sources on the edge of the target galaxy.
        It takes the segments outside the inner mask of the target.

        Parameters
        ----------
        expand_factor : float (default: 1)
            The kernel FWHM to smooth the galaxy mask if expand_factor>1.
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
        assert hasattr(self, '_segment_edge'), 'Please run detect_source_extended() first!'

        if expand_factor != 1:
            mask = scale_mask(self._segment_edge.data, factor=expand_factor)
        else:
            mask = self._segment_edge.data > 0

        self._mask_edge = mask

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
            norm = simple_norm(self._data_subbkg, **norm_kwargs)

            ax = axs[0]
            ax.imshow(self._data_subbkg, origin='lower', cmap='Greys_r', norm=norm)
            xlim = ax.get_xlim();
            ylim = ax.get_ylim()
            plot_mask_contours(mask, ax=ax, verbose=verbose, color='cyan', lw=0.5)
            ax.set_xlim(xlim);
            ax.set_ylim(ylim)
            ax.set_title('Image', fontsize=18)

            ax = axs[1]
            ax.imshow(self._segment_edge, origin='lower',
                      cmap=self._segment_edge.cmap, interpolation='nearest')
            xlim = ax.get_xlim();
            ylim = ax.get_ylim()
            plot_mask_contours(mask, ax=ax, verbose=verbose, color='cyan', lw=0.5)
            ax.set_xlim(xlim);
            ax.set_ylim(ylim)
            ax.set_title('Edge segmentation', fontsize=18)

    def gen_mask_inner(self, expand_factor=1, plot=False, fig=None, axs=None,
                       norm_kwargs=None, interactive=False, verbose=False):
        '''
        Generate the mask of the sources overlapping with the target. It takes
        the segments overlapping the target.

        Parameters
        ----------
        expand_factor : float (default: 1)
            The kernel FWHM to smooth the galaxy mask if expand_factor>1.
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
        assert hasattr(self, '_segment_inner'), 'Please run detect_source_extended() first!'

        if expand_factor != 1:
            mask = scale_mask(self._segment_inner.data, factor=expand_factor)
        else:
            mask = self._segment_inner.data > 0

        self._mask_inner = mask

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
            norm = simple_norm(self._data_subbkg, **norm_kwargs)

            ax = axs[0]
            ax.imshow(self._data_subbkg, origin='lower', cmap='Greys_r', norm=norm)
            xlim = ax.get_xlim();
            ylim = ax.get_ylim()
            plot_mask_contours(mask, ax=ax, verbose=verbose, color='cyan', lw=0.5)
            ax.set_xlim(xlim);
            ax.set_ylim(ylim)
            ax.set_title('Image', fontsize=18)

            ax = axs[1]
            ax.imshow(self._segment_inner, origin='lower',
                      cmap=self._segment_inner.cmap, interpolation='nearest')
            xlim = ax.get_xlim();
            ylim = ax.get_ylim()
            plot_mask_contours(mask, ax=ax, verbose=verbose, color='cyan', lw=0.5)
            ax.set_xlim(xlim);
            ax.set_ylim(ylim)
            ax.set_title('Inner segmentation', fontsize=18)

    def gen_mask_manual(self, mode='draw', verbose=True):
        '''
        Generate a manual mask.  A pop-up figure will be generated for
        interactive operations.

        Check the operation manual of the two modes in MaskBuilder_draw and
        MaskBuilder_segm in the utils_interactive module.

        Parameters
        ----------
        mode : {'draw', 'segm'} optional
            Choose the working mode.
            draw : create the mask by drawing polygons.
            segm : create the mask by selecting segmentations.
        verbose : bool (default: False)
            Output details if True.

        Notes
        -----
        [SGJY added]
        '''
        # Change the matplotlib backend
        ipy = get_ipython()
        ipy.run_line_magic('matplotlib', 'tk')

        # Prepare the event functions
        def on_click(event):
            mb.on_click(event)

        def on_press(event):
            mb.on_press(event)

        def on_close(event):
            mb.on_close(event)

        # Start to work
        mask = getattr(self, '_mask_contaminant', None)
        if mask is None:
            mask = np.zeros_like(self._data_subbkg, dtype=bool)

        if getattr(self, '_mask_manual', None) is None:
            self._mask_manual = np.zeros_like(mask, dtype=bool)

        fig, axs = plt.subplots(2, 2, figsize=(14, 14), sharex=True, sharey=True)
        fig.subplots_adjust(wspace=0.05, hspace=0.1)

        if mode == 'draw':
            mb = MaskBuilder_draw(self._data_subbkg, mask, mask_manual=self._mask_manual,
                                  ipy=ipy, fig=fig, axs=axs, verbose=verbose)
            fig.canvas.mpl_connect('button_press_event', on_click)
            fig.canvas.mpl_connect('key_press_event', on_press)
            fig.canvas.mpl_connect('close_event', on_close)
            plt.show()

        elif mode == 'segm':
            mb = MaskBuilder_segment(self._data_subbkg, mask, self._segment_map,
                                     mask_manual=self._mask_manual, ipy=ipy, fig=fig,
                                     axs=axs, verbose=verbose)
            fig.canvas.mpl_connect('button_press_event', on_click)
            fig.canvas.mpl_connect('close_event', on_close)
            plt.show()

    def gen_mask_outer(self, expand_factor=1.2, plot=False, fig=None,
                       axs=None, norm_kwargs=None, interactive=False,
                       verbose=False):
        '''
        Generate the mask of the sources outside the target galaxy.

        Parameters
        ----------
        expand_factor : float (default: 1)
            The kernel FWHM to smooth the galaxy mask if expand_factor>1.
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
        assert hasattr(self, '_segment_outer'), 'Please run detect_source_extended() first!'

        if expand_factor != 1:
            mask = scale_mask(self._segment_outer.data, factor=expand_factor)
        else:
            mask = self._segment_outer.data > 0

        self._mask_outer = mask

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
            norm = simple_norm(self._data_subbkg, **norm_kwargs)

            ax = axs[0]
            ax.imshow(self._data_subbkg, origin='lower', cmap='Greys_r', norm=norm)
            xlim = ax.get_xlim();
            ylim = ax.get_ylim()
            plot_mask_contours(mask, ax=ax, verbose=verbose, color='cyan', lw=0.5)
            ax.set_xlim(xlim);
            ax.set_ylim(ylim)
            ax.set_title('Image', fontsize=18)

            ax = axs[1]
            ax.imshow(self._segment_outer, origin='lower',
                      cmap=self._segment_outer.cmap, interpolation='nearest')
            xlim = ax.get_xlim();
            ylim = ax.get_ylim()
            plot_mask_contours(mask, ax=ax, verbose=verbose, color='cyan', lw=0.5)
            ax.set_xlim(xlim);
            ax.set_ylim(ylim)
            ax.set_title('Outer segmentation', fontsize=18)

    def gen_model_background(self, box_size=None, filter_size=5, plot=False,
                             norm_kwargs=None, show_mask=False):
        '''
        Generate the background model.Using median background method.
        Parameter
        ----------
        box_size: int or tuple (ny, nx)
            The size used to calculate the local median.
            If None, the default size is 1/30 of the image size.
            It is better not to use a too small box size, otherwise there is
            a high risk to remove the source emission.
        filter_size: int or tuple (ny, nx) (default: 5)
            The kernel size used to smooth the background model.
        plot : bool (default: False)
            Plot the data and segmentation map if True.
        norm_kwargs (optional) : dict
            The keywords to normalize the data image.

        Notes
        -----
        [SGJY added]
        '''
        assert hasattr(self, '_mask_background'), 'Please run gen_mask_background() first!'

        ny, nx = self._data.shape

        if not box_size:
            box_size = (ny // 30, nx // 30)

        coverage_mask = getattr(self, '_mask_coverage', None)

        sigma_clip = SigmaClip(sigma=3.)
        bkg_estimator = MeanBackground()
        bkg = Background2D(self._data, box_size, mask=self._mask_background,
                           coverage_mask=coverage_mask, filter_size=filter_size,
                           sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
        self._model_background = bkg.background
        self._model_background_rms = bkg.background_rms

        if plot:
            fig, axs = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)

            if norm_kwargs is None:
                norm_kwargs = dict(percent=80, stretch='asinh', asinh_a=0.1)
            norm = simple_norm(self._data, **norm_kwargs)

            ax = axs[0]
            ax.imshow(self._data, cmap='Greys_r', origin='lower', norm=norm)
            ax.set_title('Data', fontsize=16)

            if show_mask:
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                plot_mask_contours(self._mask_background, ax=ax, verbose=False, color='cyan', lw=0.5)

            ax = axs[1]
            ax.imshow(self._model_background, cmap='Greys_r', origin='lower', norm=norm)
            ax.set_title('Background model', fontsize=16)

    def gen_model_galaxy(self, box_size=None, filter_size=1, plot=False,
                         fig=None, axs=None, norm_kwargs=None,
                         interactive=False, verbose=False):
        '''
        Generate the galaxy model with Photutils Background2D.

        Parameters
        ----------
        box_size : int or array_like (int)
            The box size along each axis. If box_size is a scalar then a square
            box of size box_size will be used. If box_size has two elements,
            they must be in (ny, nx) order.
        filter_size : int or array_like (int), optional
            The window size of the 2D median filter to apply to
            the low-resolution background map. If filter_size is a scalar then a
            square box of size filter_size will be used. If filter_size has two
            elements, they must be in (ny, nx) order. filter_size must be odd
            along both axes. A filter size of 1 (or (1, 1)) means no filtering.
        plot : bool (default: False)
            Plot the image and galaxy model if True.
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
        '''
        assert hasattr(self, '_mask_contaminant'), 'Please run gen_mask_contaminant() first!'

        ny, nx = self._data_subbkg.shape

        if not box_size:
            # 1 times the PSF enclosed size
            bsize = int(self._psf_enclose_radius_pix * 2)
            box_size = (bsize, bsize)

        coverage_mask = getattr(self, '_mask_coverage', None)

        sigma_clip = SigmaClip(sigma=3.)
        bkg_estimator = MeanBackground()
        model = Background2D(self._data_subbkg, box_size,
                             mask=self._mask_contaminant,
                             coverage_mask=coverage_mask,
                             filter_size=filter_size, sigma_clip=sigma_clip,
                             bkg_estimator=bkg_estimator)
        self._model_galaxy = model.background
        self._model_galaxy_rms = model.background_rms

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
            norm = simple_norm(self._data_subbkg, **norm_kwargs)

            ax = axs[0]
            ax.imshow(self._data_subbkg, origin='lower', cmap='Greys_r', norm=norm)
            xlim = ax.get_xlim();
            ylim = ax.get_ylim()
            plot_mask_contours(self._mask_contaminant, ax=ax, verbose=verbose, color='cyan', lw=0.5)
            ax.set_xlim(xlim);
            ax.set_ylim(ylim)
            ax.set_title('Image', fontsize=18)

            ax = axs[1]
            ax.imshow(self._model_galaxy, origin='lower', cmap='Greys_r', norm=norm)
            xlim = ax.get_xlim();
            ylim = ax.get_ylim()
            ax.set_xlim(xlim);
            ax.set_ylim(ylim)
            ax.set_title('Galaxy model', fontsize=18)

    def gen_psf_model(self, extract_size=25, xmatch_radius=3, plx_snr=3,
                      threshold_flux=None, threshold_eccentricity=0.15,
                      num_lim=54, mask=None, oversampling=4,
                      smoothing_kernel='quadratic', maxiters=3,
                      progress_bar=False, skip_psf_model=False, plot=False,
                      fig=None, axs1=None, axs2=None, norm_kwargs=None, nrows=6,
                      ncols=9, verbose=False):
        '''
        Select the good stars and generate the PSF model.

        Parameters
        ----------

        Notes
        -----
        FIXME: We can add more selection by whether there are more than one segements
        in extracted image of the star.

        [SGJY added]
        '''
        assert hasattr(self, '_data'), 'The _data is lacking!'
        assert hasattr(self, '_segment_outer'), 'Please run detect_source_extended() first!'
        assert hasattr(self, '_wcs'), 'The _wcs is lacking!'

        tb = select_segment_stars(self._data, self._segment_outer, self._wcs,
                                  convolved_image=None,
                                  mask=mask, xmatch_radius=xmatch_radius,
                                  plx_snr=plx_snr, plot=False)

        tb.sort('segment_flux', reverse=True)

        if verbose:
            print(f'Found {len(tb)} stars')

        fltr = tb['eccentricity'] < threshold_eccentricity

        if threshold_flux is not None:
            fltr &= tb['segment_flux'] < threshold_flux

        tb_sel = tb[fltr]

        if verbose:
            print(f'Selected {len(tb_sel)}/{len(tb)} stars')

        # Cut the number of the star table
        if len(tb_sel) > num_lim:
            tb_sel = tb_sel[:num_lim]
        self._psf_table = tb_sel

        # Extract the stars
        nddata = NDData(data=self._data_subbkg)
        self._psf_stars = extract_stars(nddata, tb_sel, size=extract_size)
        nstars = len(self._psf_stars)

        if skip_psf_model:
            self._psf_model = None
            self._psf_data = None
            self._psf_oversample = None
        else:
            epsf_builder = EPSFBuilder(oversampling=oversampling,
                                       smoothing_kernel=smoothing_kernel,
                                       maxiters=maxiters, progress_bar=progress_bar)
            epsf, fitted_stars = epsf_builder(self._psf_stars)
            self._psf_model = epsf
            self._psf_data = epsf.data
            self._psf_oversample = oversampling

        if plot:
            if fig is None:
                fig = plt.figure(figsize=(21, 21))

                panel_gap = 0.01
                panel_size = 0.25
                ax1 = fig.add_axes([0.05, 0.68, panel_size, panel_size])
                ax2 = fig.add_axes([0.35, 0.68, panel_size, panel_size])
                ax3 = fig.add_axes([0.05, 0.94, 0.55, 0.02])
                ax4 = fig.add_axes([0.63, 0.68, panel_size, panel_size])
                ax5 = fig.add_axes([0.89, 0.68, 0.02, panel_size])
                ax6 = fig.add_axes([0.05, 0.05, 0.95, 0.55])
                ax6.set_ylabel('Selected PSF stars', fontsize=24, labelpad=15)
                ax6.spines['top'].set_visible(False)
                ax6.spines['right'].set_visible(False)
                ax6.spines['bottom'].set_visible(False)
                ax6.spines['left'].set_visible(False)
                ax6.set_xticklabels([])
                ax6.set_yticklabels([])
                ax6.tick_params(which='both', length=0)

                grid_gap = panel_gap / 2
                size_x = 0.9 / ncols
                size_y = 0.6 / nrows
                grid_start = [0.05, 0.65 - size_y - panel_gap]

                axs2 = []
                for l_x in range(ncols):
                    axs_y = []
                    axs2.append(axs_y)
                    for l_y in range(nrows):
                        x0 = grid_start[0] + l_x * (size_x + grid_gap)
                        y0 = grid_start[1] - l_y * (size_y + grid_gap)
                        ax = fig.add_axes([x0, y0, size_x, size_y])
                        ax.set_xticklabels([])
                        ax.set_yticklabels([])
                        ax.tick_params(which='both', length=0)
                        axs_y.append(ax)
                axs2 = np.array(axs2)  # From top down
            else:
                assert axs1 is not None, 'Need to provide three axes!'
                assert axs2 is not None, 'Need to provide the axes grid for the stars'
                ax1, ax2, ax3 = axs1

            # Plot the star properties -- flux v.s. area
            mp = ax1.scatter(np.log10(tb['segment_flux']), np.log10(tb['area']),
                             c=tb['Gmag'])

            ax1.plot(np.log10(tb_sel['segment_flux']), np.log10(tb_sel['area']),
                     ls='none', marker='.', color='red', label='Selected')

            if threshold_flux is not None:
                ax1.axvline(x=np.log10(threshold_flux), ls='--', lw=1.5, color='red', alpha=0.8)

            ax1.legend(loc='upper left', fontsize=18, handlelength=1)
            ax1.set_xlabel(r'$\log\,Flux$', fontsize=24)
            ax1.set_ylabel(r'$\log\,(Area / \mathrm{pixel}^2)$', fontsize=24)
            ax1.minorticks_on()

            # Plot the star properties -- flux v.s. eccentricity
            mp = ax2.scatter(np.log10(tb['segment_flux']), tb['eccentricity'], c=tb['Gmag'])

            ax2.plot(np.log10(tb_sel['segment_flux']), tb_sel['eccentricity'],
                     ls='none', marker='.', color='red')

            ax2.axhline(y=threshold_eccentricity, ls='--', lw=1.5, color='red', alpha=0.8, label='Criteria')
            if threshold_flux is not None:
                ax2.axvline(x=np.log10(threshold_flux), ls='--', lw=1.5, color='red', alpha=0.8)

            ax2.legend(loc='upper left', fontsize=18)
            ax2.set_xlim(ax1.get_xlim())
            ax2.set_xlabel(r'$\log\,Flux$', fontsize=24)
            ax2.set_ylabel(r'Eccentricity', fontsize=24)
            ax2.minorticks_on()

            # Add colorbar of Gmag
            cb = plt.colorbar(mp, cax=ax3, orientation='horizontal')
            cb.set_label(r'$G$ (mag)', fontsize=24)
            ax3.xaxis.set_label_position('top')
            ax3.xaxis.tick_top()
            cb.minorticks_on()

            if norm_kwargs is None:
                norm_kwargs = dict(percent=99, stretch='asinh', asinh_a=0.001)

            # Plot the PSF model
            if skip_psf_model:
                ax4.text(0.5, 0.5, 'Skipped the PSF model...', fontsize=24,
                         transform=ax4.transAxes, ha='center', va='center')
                ax4.axis('off')
                ax5.axis('off')
            else:
                norm = simple_norm(self._psf_data, **norm_kwargs)
                mp = ax4.imshow(self._psf_data, norm=norm, origin='lower',
                                cmap='viridis')
                ax4.set_title('PSF model', fontsize=24)
                plt.colorbar(mp, cax=ax5, orientation='vertical')

            # Plot all the stars
            for l_y in range(nrows):
                for l_x in range(ncols):
                    ax = axs2[l_x, l_y]
                    idx = l_x + l_y * ncols

                    if idx < nstars:
                        star = self._psf_stars[idx]
                        norm = simple_norm(star, **norm_kwargs)
                        ax.imshow(star, norm=norm, origin='lower', cmap='viridis')
                        ax.text(0.05, 0.05, f'{idx}', fontsize=18, color='white',
                                transform=ax.transAxes, ha='left', va='bottom')
                    else:
                        ax.axis('off')

    def get_psf_profile(self, enclosed_energy=0.99, plot=False, axs=None,
                        xscale='linear', yscale='linear'):
        '''
        Get the PSF profile and FWHM of the PSF.

        Parameters
        ----------
        enclosed_energy : float (0-1; default: 0.99)
            The enclosed energy to define the radius of the entire PSF.
        plot : bool (default: False)
            Plot the PSF profiles if True.
        axs : Matplotlib Axes
            The axes to plot. Two panels are needed.
        xscale, yscale : string (default: linear)
            The scale of the x and y axes.
        '''
        from photutils.centroids import centroid_quadratic

        data = self._psf_data
        ny, nx = data.shape
        xpeak = nx // 2
        ypeak = ny // 2
        xycen = centroid_quadratic(data, xpeak=xpeak, ypeak=ypeak)

        edge_radii = np.arange(ny)
        rp = RadialProfile(data, xycen, edge_radii, error=None, mask=None)

        fltr = ~np.isnan(rp.profile)
        rp_norm = rp.profile[fltr] / np.max(rp.profile[fltr])
        r_rp_pix = rp.radius[fltr] / self._psf_oversample
        f_rp = interpolate.interp1d(rp_norm, r_rp_pix)
        fwhm_pix = f_rp(0.5) * 2
        fwhm_as = fwhm_pix * self._pxs

        assert (enclosed_energy > 0) & (enclosed_energy < 1), 'Out of range!'

        radii = np.arange(1, ny)
        cog = CurveOfGrowth(data, xycen, radii, error=None, mask=None)
        cog_norm = cog.profile / np.max(cog.profile)
        r_cog_pix = cog.radius / self._psf_oversample
        f_cog = interpolate.interp1d(cog_norm, r_cog_pix)
        r_env_pix = float(f_cog(enclosed_energy))
        r_env_as = r_env_pix * self._pxs

        self._psf_fwhm = fwhm_pix * self._pxs
        self._psf_fwhm_pix = fwhm_pix
        self._psf_enclose_radius = r_env_pix * self._pxs
        self._psf_enclose_radius_pix = r_env_pix
        self._rp = (r_rp_pix, rp_norm)
        self._cog = (r_cog_pix, cog_norm)

        if plot:
            if axs is None:
                fig, axs = plt.subplots(1, 2, figsize=(14, 7))
                fig.subplots_adjust(wspace=0.05)

            locator = plt.MaxNLocator(nbins=5)

            ax = axs[0]
            ax.plot(r_rp_pix, rp_norm, color='k')
            ax.axvline(fwhm_pix / 2, ls='--', color='r')
            ax.axhline(y=0.5, ls='--', color='r')

            ax.set_xlabel('Radius (pixel)', fontsize=24)
            ax.set_ylabel('Normalized flux', fontsize=24)
            ax.minorticks_on()
            ax.xaxis.set_major_locator(locator)
            ax.text(0.95, 0.95, f'FWHM={fwhm_as:.2f}"', fontsize=18,
                    transform=ax.transAxes, ha='right', va='top')
            ax.set_xscale(xscale)
            ax.set_yscale(yscale)

            axu = ax.twiny()
            xlim = np.array(ax.get_xlim()) * self._pxs
            axu.set_xlim(xlim)
            axu.set_xlabel('Radius (arcsec)', fontsize=24)
            axu.minorticks_on()
            axu.set_xscale(xscale)

            ax = axs[1]
            ax.plot(r_cog_pix, cog_norm, color='k')
            ax.axhline(y=enclosed_energy, ls='--', color='r')
            ax.axvline(x=r_env_pix, ls='--', color='r')
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
            ax.set_xlabel('Radius (pixel)', fontsize=24)
            ax.set_ylabel('Curve of growth', fontsize=24)
            ax.minorticks_on()
            ax.xaxis.set_major_locator(locator)
            ax.text(0.95, 0.05, f'R({enclosed_energy * 100:.1f}%)={r_env_as:.2f}"', fontsize=18,
                    transform=ax.transAxes, ha='right', va='bottom')
            ax.set_xscale(xscale)
            ax.set_yscale(yscale)

            axu = ax.twiny()
            xlim = np.array(ax.get_xlim()) * self._pxs
            axu.set_xlim(xlim)
            axu.set_xlabel('Radius (arcsec)', fontsize=24)
            axu.minorticks_on()
            axu.set_xscale(xscale)

    def gen_segment_inner(self, detect_thres=5., xmatch_radius=3.,
                          threshold_plx=2, threshold_gmag=None,
                          mask_radius=None, center_radius=5.,
                          plot=False, fig=None, axs=None, norm_kwargs=None,
                          interactive=False, verbose=False):
        '''
        Get the segmentation of the foreground stars that are overlapping with
        the target galaxy in the inner region.

        Parameter
        -----------
        detect_thres: float
            The threshold for DAOFinder.
        xmatch_radius: float
            The xmatch radius. arcsec.
        threshold_plx (optional): float
            The threshold to select stars according to Gaia parallax SNR.
        threshold_gmag: float
            The max G band magnitude of the stars which will be picked out.
        plot : bool (default: False)
            Plot the image and mask if True.
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
        assert hasattr(self, '_data_subbkg'), 'Please run background_subtract() first!'
        assert hasattr(self, '_mask_galaxy_inner'), 'Please run detect_source_extended() first!'
        assert hasattr(self, '_psf_fwhm_pix'), 'Please run get_PSF_profile() first!'

        # initial detection
        daofind = DAOStarFinder(threshold=detect_thres * self._bkg_std, fwhm=self._psf_fwhm_pix)
        sources = daofind(self._data_subbkg, mask=~self._mask_galaxy_inner)

        segm_data = np.zeros_like(self._data_subbkg, dtype=int)

        # Gaia Xmatch
        w = self._wcs
        if sources:
            c = w.pixel_to_world(sources['xcentroid'], sources['ycentroid'])
            sources_world = Table([c.ra.deg, c.dec.deg, sources['xcentroid'],
                                   sources['ycentroid']],
                                  names=['ra', 'dec', 'x', 'y'])
            t_o = xmatch_gaiadr3(sources_world, xmatch_radius, colRA1='ra', colDec1='dec')  # Gaia xmatch.

            dist = np.sqrt((t_o['x'] - self._coord_pix[0]) ** 2 + (t_o['y'] - self._coord_pix[1]) ** 2)
            fltr = dist > center_radius

            plx_snr = (t_o['Plx'] / t_o['e_Plx'])
            fltr &= (plx_snr > threshold_plx)
            if hasattr(t_o['Plx'], 'mask'):
                fltr &= ~t_o['Plx'].mask

            if threshold_gmag is not None:
                fltr &= (t_o['Gmag'] < threshold_gmag)

            t_s = t_o[fltr]
            self._table_overlap_stars = t_s

            if mask_radius is None:
                mask_radius = self._psf_enclose_radius_pix

            if verbose:
                print(f'Found {len(t_s)} stars!')
                coord_m = tqdm.tqdm(zip(t_s['x'], t_s['y']))
            else:
                coord_m = zip(t_s['x'], t_s['y'])

            for loop, (x, y) in enumerate(coord_m):
                add_mask_circle(segm_data, x, y, mask_radius, int(loop + 1))

        self._segment_inner = SegmentationImage(segm_data)
        mask = segm_data > 0

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
            norm = simple_norm(self._data_subbkg, **norm_kwargs)

            ax = axs[0]
            ax.imshow(self._data_subbkg, origin='lower', cmap='Greys_r', norm=norm)
            xlim = ax.get_xlim();
            ylim = ax.get_ylim()

            ax.plot(sources['xcentroid'], sources['ycentroid'], ls='none',
                    marker='+', ms=5, mfc='none', mec='orange', mew=0.5, alpha=1)
            plot_mask_contours(mask, ax=ax, verbose=verbose, color='cyan', lw=0.5)
            ax.set_xlim(xlim);
            ax.set_ylim(ylim)
            ax.set_title('Image', fontsize=18)

            ax = axs[1]
            ax.imshow(self._segment_inner, origin='lower',
                      cmap=self._segment_inner.cmap, interpolation='nearest')
            xlim = ax.get_xlim();
            ylim = ax.get_ylim()
            plot_mask_contours(mask, ax=ax, verbose=verbose, color='cyan', lw=0.5)
            ax.set_xlim(xlim);
            ax.set_ylim(ylim)
            ax.set_title('Inner segmentation', fontsize=18)

    def gen_target_mask(self, threshold, npixels=12, mask=None, connectivity=8,
                        kernel_fwhm=3, expand_factor=1, bounds: list = None,
                        plot=False, fig=None, axs=None, norm_kwargs=None,
                        interactive=False, verbose=False):
        '''
        Get the mask of the target galaxy.
        Assuming that there is only one target object on the background.
        If the coord of the object is None, then the target which has the largest
        area will be taken into consideration.

        Parameter
        ----------
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
        connectivity : {4, 8} (default: 8)
            The type of pixel connectivity used in determining how pixels are
            grouped into a detected source. The options are 4 or 8 (default).
            4-connected pixels touch along their edges. 8-connected pixels touch
            along their edges or corners.
        kernel_fwhm : float (default: 2)
            The kernel FWHM to smooth the image.
        expand_factor : float (default: 1)
            The kernel FWHM to smooth the galaxy mask if expand_factor>1.
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
        err_text = 'Lacking the background subtracted data! ' + \
                   'Run gen_background_model() and background_subtract2() first.'
        assert hasattr(self, '_data_subbkg'), err_text

        if hasattr(self, '_mask_coverage'):
            if mask is None:
                mask = self._mask_coverage
            else:
                mask |= self._mask_coverage

        mask, smap, cdata = gen_image_mask(self._data_subbkg, threshold,
                                           npixels=npixels, mask=mask,
                                           connectivity=connectivity,
                                           kernel_fwhm=kernel_fwhm,
                                           expand_factor=expand_factor,
                                           bounds=bounds,
                                           choose_coord=self._coord_pix,
                                           plot=plot, fig=fig, axs=axs,
                                           norm_kwargs=norm_kwargs,
                                           interactive=interactive,
                                           verbose=verbose)

        self._mask_galaxy = mask
        self._segment_map = smap
        self._data_convolved = cdata

    def load_psf_data(self, filename, oversample=None, extension=0):
        '''
        Load the PSF data from a FITS file.

        Parameters
        ----------
        filename : string
            The PSF file name.
        oversample (optional) : float
            The oversampling of the PSF image. If it is not provided, the code
            will look for PSCALE in the FITS header to calculate the oversampling
            by itself.
        extension : int (default: 0)
            The extenstion saving the PSF data and header.
        '''
        hdul = fits.open(filename)
        header = hdul[extension].header
        self._psf_data = hdul[extension].data

        if oversample is None:
            pscale = header.get('PSCALE', None)

            if pscale is not None:
                oversample = self._pxs / pscale

        self._psf_oversample = oversample

    def plot_psf(self, fig=None, axs=None, norm_kwargs=None, xscale='linear',
                 yscale='linear'):
        '''
        Plot the PSF info.

        Parameters
        ----------
        fig : Matplotlib Figure
            The figure to plot.
        axs : Matplotlib Axes
            The axes to plot. Two panels are needed.
        norm_kwargs (optional) : dict
            The keywords to normalize the data image.
        xscale, yscale : string (default: linear)
            The scale of the x and y axes.
        '''
        assert hasattr(self, '_psf_data'), 'No PSF data!'

        if axs is None:
            fig, axs = plt.subplots(1, 3, figsize=(21, 7))
            fig.subplots_adjust(wspace=0.1)

        locator = plt.MaxNLocator(nbins=5)

        ax = axs[0]
        if norm_kwargs is None:
            norm_kwargs = dict(percent=99, stretch='asinh', asinh_a=0.001)
        norm = simple_norm(self._psf_data, **norm_kwargs)
        ax.imshow(self._psf_data, origin='lower', norm=norm)
        ax.set_title('(a) PSF data', fontsize=18)

        ax = axs[1]
        if hasattr(self, '_rp'):
            ax.plot(self._rp[0] * self._pxs, self._rp[1], color='k')
            ax.axhline(y=0.5, ls='--', color='r')
            ax.xaxis.set_major_locator(locator)
            ax.set_xscale(xscale)
            ax.set_yscale(yscale)
            ax.minorticks_on()

        if hasattr(self, '_psf_fwhm'):
            ax.axvline(self._psf_fwhm / 2, ls='--', color='r')
            ax.text(0.95, 0.95, f'FWHM={self._psf_fwhm_pix:.2f}"', fontsize=18,
                    transform=ax.transAxes, ha='right', va='top')
        else:
            ax.text(0.95, 0.95, f'FWHM not available', fontsize=18,
                    transform=ax.transAxes, ha='right', va='top')

        ax.set_xlabel('Radius (arcsec)', fontsize=24)
        ax.set_title('Radial profile', fontsize=18)

        ax = axs[2]
        if hasattr(self, '_cog'):
            ax.plot(self._cog[0] * self._pxs, self._cog[1], color='k')
            ax.xaxis.set_major_locator(locator)
            ax.set_xscale(xscale)
            ax.set_yscale(yscale)
            ax.minorticks_on()

        if hasattr(self, '_psf_enclose_radius'):
            ax.axvline(self._psf_enclose_radius, ls='--', color='r')
            ax.text(0.95, 0.05, f'PSF enclose radius: {self._psf_enclose_radius:.2f}"', fontsize=18,
                    transform=ax.transAxes, ha='right', va='bottom')
        else:
            ax.text(0.95, 0.05, f'PSF enclose radius not available', fontsize=18,
                    transform=ax.transAxes, ha='right', va='bottom')

        ax.yaxis.set_label_position("right")
        ax.set_xlabel('Radius (arcsec)', fontsize=24)
        ax.set_title('Curve of growth', fontsize=18)

    def plot_summary(self, fig=None, axs=None, norm_kwargs=None,
                     interactive=False, verbose=False):
        '''
        Plot the data and mask:
        (a) Original data, (b) background subtracted data, (c) galaxy model,
        (d) background mask, (e) contaminant mask, (f) combined segmentation

        Parameters
        ----------
        plot : bool (default: False)
            Plot the image and cleaned image if True.
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
        '''
        # Combine all segments
        if (hasattr(self, '_segment_inner') & hasattr(self, '_segment_edge') &
                hasattr(self, '_segment_outer')):
            segm = segment_combine(self._segment_outer, self._segment_edge)
            segm = segment_combine(segm, self._segment_inner)
        else:
            segm = None

        if interactive:
            ipy = get_ipython()
            ipy.run_line_magic('matplotlib', 'tk')

            def on_close(event):
                ipy.run_line_magic('matplotlib', 'inline')

        if axs is None:
            fig, axs = plt.subplots(2, 3, figsize=(21, 14), sharex=True, sharey=True)
            fig.subplots_adjust(wspace=0.05, hspace=0.1)
        else:
            assert fig is not None, 'Please provide fig together with axs!'

        if interactive:
            fig.canvas.mpl_connect('close_event', on_close)

        if norm_kwargs is None:
            norm_kwargs = dict(percent=99.99, stretch='asinh', asinh_a=0.001)

        ax = axs[0, 0]
        if hasattr(self, '_data'):
            norm = simple_norm(self._data, **norm_kwargs)
            ax.imshow(self._data, origin='lower', cmap='Greys_r', norm=norm)
        else:
            ax.text(0.5, 0.5, 'No data', fontsize=24, transform=ax.transAxes,
                    ha='center', va='center')
        ax.set_title('(a) Original image', fontsize=18)

        ax = axs[0, 1]
        if hasattr(self, '_data_subbkg'):
            norm = simple_norm(self._data_subbkg, **norm_kwargs)
            ax.imshow(self._data_subbkg, origin='lower', cmap='Greys_r', norm=norm)
        else:
            ax.text(0.5, 0.5, 'No data', fontsize=24, transform=ax.transAxes,
                    ha='center', va='center')
        ax.set_title('(b) Background subtracted image', fontsize=18)

        ax = axs[0, 2]
        if hasattr(self, '_model_galaxy'):
            ax.imshow(self._model_galaxy, origin='lower', cmap='Greys_r', norm=norm)
        else:
            ax.text(0.5, 0.5, 'No data', fontsize=24, transform=ax.transAxes,
                    ha='center', va='center')
        ax.set_title('(c) Galaxy model', fontsize=18)

        ax = axs[1, 0]
        if hasattr(self, '_mask_background'):
            ax.imshow(self._mask_background, origin='lower', cmap='Greys_r')
        else:
            ax.text(0.5, 0.5, 'No data', fontsize=24, transform=ax.transAxes,
                    ha='center', va='center')
        ax.set_title('(d) Background mask', fontsize=18)

        ax = axs[1, 1]
        if hasattr(self, '_mask_contaminant'):
            ax.imshow(self._mask_contaminant, origin='lower', cmap='Greys_r')
            xlim = ax.get_xlim();
            ylim = ax.get_ylim()
        else:
            ax.text(0.5, 0.5, 'No data', fontsize=24, transform=ax.transAxes,
                    ha='center', va='center')
            xlim = None;
            ylim = None

        if hasattr(self, '_mask_galaxy'):
            plot_mask_contours(self._mask_galaxy, ax=ax, verbose=verbose, color='cyan', lw=0.5)

        if hasattr(self, '_mask_galaxy_inner'):
            plot_mask_contours(self._mask_galaxy_inner, ax=ax, verbose=verbose, color='magenta', lw=0.5)

        if xlim is not None:
            ax.set_xlim(xlim);
            ax.set_ylim(ylim)

        ax.set_title('(e) Contaminant mask', fontsize=18)

        ax = axs[1, 2]
        if segm is None:
            ax.text(0.5, 0.5, 'No data', fontsize=24, transform=ax.transAxes,
                    ha='center', va='center')
        else:
            ax.imshow(segm, origin='lower', cmap=segm.cmap, interpolation='nearest')

        ax.set_title('(f) Combined segmentation', fontsize=18)

    def gen_photometry_aperture(self, threshold_segm, threshold_snr=2, naper=10, fracs=[0.5, 3],
                                plot=False, axs=None, **segm_kwargs):
        aper = gen_aperture_ellipse(self._data_clean, self._coord_pix, threshold_segm=threshold_segm,
                                    threshold_snr=threshold_snr, psf_fwhm=self._psf_fwhm_pix,
                                    mask=self._mask_contaminant, naper=naper, fracs=fracs,
                                    plot=plot, axs=axs, **segm_kwargs)
        self._phot_aper = aper
        return aper

    def rebin_image(self, factor=10, plot=False, norm_kwargs=None):
        '''
        Rebin the image to reduce the image size.

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
        rebin_wcs = deepcopy(self._wcs)
        rebin_wcs.wcs.crpix = self._wcs.wcs.crpix / factor
        rebin_wcs.wcs.cdelt = self._wcs.wcs.cdelt * factor

        shape_out = (rebin_wcs.pixel_shape[0] // factor, rebin_wcs.pixel_shape[1] // factor)
        rebin_wcs.pixel_shape = shape_out

        self._data_rebin, _ = reproject_interp((self._data, self._wcs), rebin_wcs, shape_out=shape_out)
        self._rebin_wcs = rebin_wcs

        if plot:
            fig, axs = plt.subplots(1, 2, figsize=(14, 7))

            if norm_kwargs is None:
                norm_kwargs = dict(percent=99.99, stretch='asinh', asinh_a=0.001)

            ax = axs[0]
            norm = simple_norm(self._data, **norm_kwargs)
            ax.imshow(self._data, origin='lower', norm=norm)

            ax = axs[1]
            norm = simple_norm(self._data_rebin, **norm_kwargs)
            ax.imshow(self._data_rebin, origin='lower', norm=norm)

    def refine_segment_overlap(self, threshold_i=4, threshold_e=4,
                               kernel_fwhm_i=0, kernel_fwhm_e=0, npixels_i=12,
                               npixels_e=12, deblend_i=True, deblend_e=True,
                               contrast_i=1e-6, contrast_e=1e-6, nlevels_i=256,
                               nlevels_e=256, threshold_gmag=None,
                               connectivity=8, plot=False, fig=None,
                               axs=None, norm_kwargs=None, interactive=False,
                               verbose=False):
        '''
        Refine the segmentations that are overlapping with the target galaxy.
        This means the segment_edge and segment_inner.

        Parameters
        ----------
        threshold_i, threshold_e : float
            The threshold ot detect source in the galaxy inner and edge region.
        kernel_fwhm_i, kernel_fwhm_e : float (default: 2)
            The kernel FWHM to smooth the image.
        npixels_i, npixels_e : int
            The minimum number of connected pixels, each greater than threshold,
            that an object must have to be detected. npixels must be a positive
            integer.
        deblend_i, deblend_e : bool (default: True)
            Deblend the segmentation if True.
        contrast_i, contrast_e : float (default: 1e-6)
            The fraction of the total source flux that a local peak must have (at
            any one of the multi-thresholds) to be deblended as a separate object.
            contrast must be between 0 and 1, inclusive. If contrast=0 then every
            local peak will be made a separate object (maximum deblending).
            If contrast=1 then no deblending will occur. The default is 0.001, which
            will deblend sources with a 15 magnitude difference.
        nlevels_i, nlevels_e : int (default: 32)
            The number of multi-thresholding levels to use for deblending. Each
            source will be re-thresholded at nlevels levels spaced between its
            minimum and maximum values (non-inclusive). The mode keyword determines
            how the levels are spaced.
        threshold_gmag (optoinal) : float
            The threshold to select bright stars.
        connectivity : {4, 8} (default: 8)
            The type of pixel connectivity used in determining how pixels are
            grouped into a detected source. The options are 4 or 8 (default).
            4-connected pixels touch along their edges. 8-connected pixels touch
            along their edges or corners.
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
        '''
        assert hasattr(self, '_data_subbkg'), 'Please run background_subtract2() first!'
        assert hasattr(self, '_model_galaxy'), 'Please run gen_model_galaxy() first!'
        assert hasattr(self, '_mask_galaxy_inner'), 'Please run detect_source_extended() first!'

        image = self._data_subbkg - self._model_galaxy
        mea, med, std = sigma_clipped_stats(image)
        image -= med

        # Inner segmentation
        if threshold_i is not None:
            segm_i, _ = get_image_segmentation(image, threshold_i * std, npixels=npixels_i,
                                               mask=~self._mask_galaxy_inner,
                                               connectivity=connectivity,
                                               kernel_fwhm=kernel_fwhm_i,
                                               deblend=deblend_i, contrast=contrast_i,
                                               nlevels=nlevels_i, plot=False)

            segm_data = np.zeros_like(image, dtype=int)
            tb = self._table_overlap_stars

            if threshold_gmag is not None:
                tb = tb[tb['Gmag'] < threshold_gmag]

            for x, y in zip(tb['x'], tb['y']):
                l = segm_i.data[int(y), int(x)]
                segm_data[segm_i.data == l] = l

            self._segment_inner = SegmentationImage(segm_data)
            mask_i = self._segment_inner.data > 0
        else:
            mask_i = None

        # Edge segmentation
        if threshold_e is not None:
            segm_e, _ = get_image_segmentation(image, threshold_e * std, npixels=npixels_e,
                                               mask=(self._mask_galaxy_inner | (~self._mask_galaxy)),
                                               connectivity=connectivity,
                                               kernel_fwhm=kernel_fwhm_e,
                                               deblend=deblend_e, contrast=contrast_e,
                                               nlevels=nlevels_e, plot=False)
            self._segment_edge = segm_e
            mask_e = segm_e.data > 0
        else:
            mask_e = None

        if plot:
            if interactive:
                ipy = get_ipython()
                ipy.run_line_magic('matplotlib', 'tk')

                def on_close(event):
                    ipy.run_line_magic('matplotlib', 'inline')

            if axs is None:
                fig, axs = plt.subplots(2, 2, figsize=(14, 14), sharex=True, sharey=True)
                fig.subplots_adjust(wspace=0.05, hspace=0.08)
            else:
                assert fig is not None, 'Please provide fig together with axs!'

            if interactive:
                fig.canvas.mpl_connect('close_event', on_close)

            if norm_kwargs is None:
                norm_kwargs = dict(percent=99.99, stretch='asinh', asinh_a=0.001)
            norm = simple_norm(self._data_subbkg, **norm_kwargs)

            ax = axs[0, 0]
            ax.imshow(self._data_subbkg, origin='lower', cmap='Greys_r', norm=norm)
            xlim = ax.get_xlim();
            ylim = ax.get_ylim()

            if mask_i is not None:
                plot_mask_contours(mask_i, ax=ax, verbose=verbose, color='cyan', lw=0.5)

            if mask_e is not None:
                plot_mask_contours(mask_e, ax=ax, verbose=verbose, color='magenta', lw=0.5)

            ax.set_xlim(xlim);
            ax.set_ylim(ylim)
            ax.set_title('Image', fontsize=18)

            ax = axs[0, 1]
            ax.imshow(image, origin='lower', cmap='Greys_r', norm=norm)
            xlim = ax.get_xlim();
            ylim = ax.get_ylim()

            ax.plot(tb['x'], tb['y'], marker='+', color='r', ls='none')

            if mask_i is not None:
                plot_mask_contours(mask_i, ax=ax, verbose=verbose, color='cyan', lw=0.5)

            if mask_e is not None:
                plot_mask_contours(mask_e, ax=ax, verbose=verbose, color='magenta', lw=0.5)

            ax.set_xlim(xlim);
            ax.set_ylim(ylim)
            ax.set_title('Galaxy removed image', fontsize=18)

            ax = axs[1, 0]
            ax.imshow(self._segment_inner, origin='lower', cmap=self._segment_inner.cmap, interpolation='nearest')

            if mask_i is None:
                ax.set_title('Inner segmentation (no change)', fontsize=18)
            else:
                ax.set_title('Inner segmentation (updated)', fontsize=18)

            ax = axs[1, 1]
            ax.imshow(self._segment_edge, origin='lower',
                      cmap=self._segment_edge.cmap, interpolation='nearest')

            if mask_e is None:
                ax.set_title('Edge segmentation (no change)', fontsize=18)
            else:
                ax.set_title('Edge segmentation (updated)', fontsize=18)

    def reset_mask_manual(self):
        '''
        Set the mask_manual to None.

        Notes
        -----
        [SGJY added]
        '''
        self._mask_manual = None

    def scale_rebin_mask(self, mask, plot=False, norm_kwargs=None, verbose=False):
        '''
        Scale the mask generated from the rebinned image back to the original
        pixel scale of the image.

        Parameters
        ----------
        mask : 2D array
            The mask of the rebinned image, to be scaled up.
        plot : bool (default: False)
            Plot the data and segmentation map if True.
        norm_kwargs (optional) : dict
            The keywords to normalize the data image.
        verbose : bool (default: True)

        Notes
        -----
        [SGJY added]
        '''
        mask_s, _ = reproject_interp((mask, self._rebin_wcs), self._wcs, shape_out=self._wcs.pixel_shape)

        if plot:
            fig, axs = plt.subplots(1, 2, figsize=(14, 7))

            if norm_kwargs is None:
                norm_kwargs = dict(percent=90, stretch='asinh', asinh_a=0.1)

            ax = axs[0]
            norm = simple_norm(self._data, **norm_kwargs)
            ax.imshow(self._data, origin='lower', cmap='Greys_r', norm=norm)
            plot_mask_contours(mask_s, ax=ax, color='cyan', verbose=verbose)

            ax = axs[1]
            ax.imshow(mask_s, origin='lower', cmap='Greys_r')
        return mask_s

    def save(self, filename, comprehensive=False, overwrite=False):
        '''
        Save the data into fits file.
        '''
        # Set the header information
        now = datetime.now()
        self._header['PIXSED'] = True
        self._header['RA'] = self._ra_deg
        self._header['DEC'] = self._dec_deg
        self._header['XCOORD'] = self._coord_pix[0]
        self._header['YCOORD'] = self._coord_pix[1]
        self._header['PSCALE'] = self._pxs
        self._header['COMMENT'] = f'Reduced by PIXSED on {now.strftime("%d/%m/%YT%H:%M:%S")}'

        if hasattr(self, '_phot_aper'):
            self._header['APER_SMA'] = self._phot_aper.a
            self._header['APER_SMB'] = self._phot_aper.b
            self._header['APER_PA'] = self._phot_aper.theta

        # Primary extension
        hduList = [fits.PrimaryHDU(header=self._header)]

        header_img = self._wcs.to_header()
        hduList.append(fits.ImageHDU(self._data_clean, header=header_img, name='data_clean'))
        hduList.append(fits.ImageHDU(self._mask_background.astype(int), header=header_img, name='mask_background'))
        hduList.append(fits.ImageHDU(self._mask_contaminant.astype(int), header=header_img, name='mask_contaminant'))
        hduList.append(fits.ImageHDU(self._mask_galaxy.astype(int), header=header_img, name='mask_galaxy'))
        hduList.append(fits.ImageHDU(self._mask_galaxy_inner.astype(int), header=header_img, name='mask_galaxy_inner'))
        hduList.append(fits.ImageHDU(self._segment_inner.data, header=header_img, name='segment_inner'))
        hduList.append(fits.ImageHDU(self._segment_edge.data, header=header_img, name='segment_edge'))
        hduList.append(fits.ImageHDU(self._segment_outer.data, header=header_img, name='segment_outer'))

        mask_manual = getattr(self, '_mask_manual', None)
        if mask_manual is not None:
            hduList.append(fits.ImageHDU(mask_manual.astype(int), header=header_img, name='mask_manual'))

        if hasattr(self, '_psf_data'):
            header_psf = fits.Header()
            header_psf['FWHM'] = self._psf_fwhm
            header_psf['ENRADIUS'] = self._psf_enclose_radius
            header_psf['OVERSAMP'] = self._psf_oversample
            hduList.append(fits.ImageHDU(self._psf_data, name='psf_data', header=header_psf))

        if comprehensive:
            hduList.append(fits.ImageHDU(self._data_subbkg, header=header_img, name='data_subbkg'))
            hduList.append(fits.ImageHDU(self._data, header=header_img, name='data'))
            hduList.append(fits.ImageHDU(self._model_background, header=header_img, name='model_background'))
            hduList.append(fits.ImageHDU(self._model_background_rms, header=header_img, name='model_background_rms'))
            hduList.append(fits.ImageHDU(self._model_galaxy, header=header_img, name='model_galaxy'))
            hduList.append(fits.ImageHDU(self._model_galaxy_rms, header=header_img, name='model_galaxy_rms'))
            hduList.append(fits.ImageHDU(self._mask_inner.astype(int), header=header_img, name='mask_inner'))
            hduList.append(fits.ImageHDU(self._mask_edge.astype(int), header=header_img, name='mask_edge'))
            hduList.append(fits.ImageHDU(self._mask_outer.astype(int), header=header_img, name='mask_outer'))

            mask_coverage = getattr(self, '_mask_coverage', None)
            if mask_coverage is not None:
                hduList.append(fits.ImageHDU(mask_coverage.astype(int), header=header_img, name='mask_coverage'))

        hdul = fits.HDUList(hduList)
        hdul.writeto(filename, overwrite=overwrite)

    def save_mask(self, filename, mask_name, overwrite=False):
        '''
        Save the masks.

        Parameters
        ----------
        filename : string
            The file name to save the mask.
        mask_name : string
            The name of the mask attribute, e.g. _mask_background, _mask_contaminant
        overwrite : bool (default: False)
            Overwrite the existing FITS file if True.
        '''
        assert hasattr(self, mask_name), f'Cannot find the mask name ({mask_name})!'
        mask = getattr(self, mask_name).astype(int)

        hduList = [fits.PrimaryHDU(header=self._header)]
        hduList.append(fits.ImageHDU(mask, header=self._wcs.to_header(),
                                     name=mask_name))

        hdul = fits.HDUList(hduList)
        hdul.writeto(filename, overwrite=overwrite)

    def set_mask_coverage(self, mask=None, shape='rect', mask_kwargs=None, 
                          fill_value=None, plot=False, fig=None, axs=None, 
                          norm_kwargs=None, interactive=False, verbose=False):
        '''
        Set the field mask where the used region

        Parameters
        ----------
        mask (optional) : bool
            The coverage mask with True indicate pixels to be discarded. One can
            manually define the coverage mask with this input.
        shape : string (default: 'rect')
            The shape of the predefined mask with True indicate pixels to be
            discarded. The predefined shapes are circular ('circ') and
            rectangular ('rect').
        mask_kwargs (optional) : dict
            The parameters of the predefined shape function.
            'circ' : x, y, radius
                x, y : The center pixel coordinate of the circle.
                radius : The radius of the circle, units: pixel.
            'rect' : xmin, xmax, ymin, ymax
                The ranges of x and y axes.
        plot : bool (default: False)
            Plot the image and the mask if True.
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
        '''
        if mask is None:
            mask = np.zeros_like(self._data, dtype=bool)
            if shape == 'rect':
                add_mask_rect(mask, value=True, **mask_kwargs)
            elif shape == 'circle':
                add_mask_circle(mask, value=True, **mask_kwargs)
            else:
                raise KeyError(f'Cannot recognize the shape ({shape})')
            mask = ~mask

        self._mask_coverage = mask

        if fill_value is not None:
            self._data[mask] = fill_value

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
                norm_kwargs = dict(percent=90., stretch='asinh', asinh_a=0.5)
            norm = simple_norm(self._data, **norm_kwargs)

            ax = axs[0]
            ax.imshow(self._data, origin='lower', cmap='Greys_r', norm=norm)
            xlim = ax.get_xlim();
            ylim = ax.get_ylim()
            plot_mask_contours(mask, ax=ax, verbose=verbose, color='cyan', lw=0.5)
            ax.set_xlim(xlim);
            ax.set_ylim(ylim)
            ax.set_title('Image', fontsize=18)

            ax = axs[1]
            ax.imshow(mask, origin='lower', cmap='Greys_r')
            ax.set_title('Coverage mask', fontsize=18)

    def set_psf_sizes(self, fwhm: float = None, enclose_radius: float = None):
        '''
        Set the PSF FWHM and the radius to enclose the PSF.

        Parameters
        ----------
        fwhm (optional) : float
            The FWHM of the PSF in arcsec.
        enclose_radius (optional) : float
            The radius to enclose the PSF, in arcsec.
        '''
        if fwhm is not None:
            self._psf_fwhm = fwhm
            self._psf_fwhm_pix = fwhm / self._pxs

        if enclose_radius is not None:
            self._psf_enclose_radius = enclose_radius
            self._psf_enclose_radius_pix = enclose_radius / self._pxs

    def __repr__(self):
        '''
        Print property of the image.
        '''
        return f'Image of {self._target} in {self._telescope}-{self._band}'


class Atlas(object):
    '''
    An atlas of images. Again, for the moment, we assume that there is only one
    science target in each image.
    '''

    def __init__(self, filenames: list, coord_sky: tuple, telescope_list: list = None,
                 band_list: list = None, verbose=True):
        '''
        Parameters
        ----------
        filenames : list
            List of filenames.
        coord_sky : tuple
            Sky coordinate of the target, (deg, deg) or (hms, dms)
        '''
        self._image_list = []
        for loop, f in enumerate(filenames):
            if telescope_list is not None:
                tel = telescope_list[loop]
            else:
                tel = None

            if band_list is not None:
                band = band_list[loop]
            else:
                band = None

            self._image_list.append(Image(filename=f, coord_sky=coord_sky,
                                          telescope=tel, band=band,
                                          verbose=verbose))

        self._coord_sky = coord_sky
        c_sky = read_coordinate(coord_sky[0], coord_sky[1])
        self._ra_deg = c_sky.ra.deg
        self._dec_deg = c_sky.dec.deg
        self._n_image = len(self._image_list)

    def adapt_masks(self, filename, interpolate_scale=None, plot=False,
                    fig=None, axs=None, norm_kwargs=None, verbose=False):
        '''
        Adapt the masks and segmentations from a FITS file.

        Parameters
        ----------
        filename : string
            The FITS file name of the reference data with masks and
            segmentations.
        interpolate_scale (optional) : float
            The scale to interpolate the data across the mask, units: arcsec.
        plot : bool (default: False)
            Plot the results if True.
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
        '''
        img_ref = Image(filename, verbose=verbose)

        assert hasattr(img_ref, '_psf_enclose_radius'), \
            'The psf_enclose_radius is required for a reference image file.'

        # The scale to interpolate mask
        if interpolate_scale is None:
            assert hasattr(img_ref, '_psf_enclose_radius'), 'Need to provide the _psf_enclose_radius in the reference image!'
            # Take the PSF scale of the reference image if interpolate_scale is not specified
            self._mask_interpolate_scale = img_ref._psf_enclose_radius
        else:
            self._mask_interpolate_scale = interpolate_scale

        for loop, img in enumerate(self._image_list):
            if verbose:
                print(f'[adapt_masks] image {loop}: {img}')

            img.adapt_to(img_ref, plot=plot, verbose=verbose)

    def gen_photometry_aperture(self, filename):
        hdul = fits.open(filename)
        header = hdul[0].header
        w0 = WCS(header)
        w = self._wcs_match
        pxs = np.abs(WCS(header).wcs.cdelt[0]) * 3600
        pxs_matched = self._pxs_match
        ra = header['RA']
        dec = header["DEC"]
        pa0 = header["APER_PA"]
        ep0 = (
        header['XCOORD'] + header["APER_SMA"] * math.cos(pa0), header['yCOORD'] + header["APER_SMA"] * math.sin(pa0))
        ep_world = w0.pixel_to_world(ep0[0], ep0[1])
        ep = w.world_to_pixel(ep_world)
        sky_coord = SkyCoord(ra, dec, unit='deg')
        x, y = w.world_to_pixel(sky_coord)
        sma = header["APER_SMA"] * pxs / pxs_matched
        smb = header["APER_SMB"] * pxs / pxs_matched
        pa = math.atan((ep[1] - y) / (ep[0] - x))

        aper = EllipticalAperture((x, y), sma, smb, pa)

        return aper

    def aperture_photometry(self, aperture, mask=None, rannu_in=1.25,
                            rannu_out=1.6,
                            error=True, bkgsub=True, calibration_list=None,
                            nsample=300,
                            area_sample=[0.02, 0.04, 0.08, 0.16],
                            plot=False, ncols=1, axs=None,
                            norm_kwargs=None, text_kwargs=None):
        '''
        Aperture photometry of the matched images.

        Parameters
        ----------
        aperture : EllipticalAperture
            The elliptical aperture to measure the image.
        mask (optional) : 2D array
            The mask of the contaminants.
        rannu_in, rannu_out : float (default: 1.25, 1.60)
            The inner and outer annulus semimajor axes, units: pixel. The values
            follow Clark et al. (2017).
        plot : bool (default: False)
            Plot the results.
        ncols : int (default: 1)
            The number panels in each row.
        ax : Axis
            The axis to plot.
        norm_kwargs (optional) : dict
            The keywords to normalize the data image.
        text_kwargs (optional) : dict
            The keywords of the text.
        '''
        if plot:
            if axs is None:
                nrows = int(np.ceil(self._n_image / ncols))
                fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
                axs = axs.flatten()
                fig.subplots_adjust(wspace=0.05, hspace=0.05)

            if norm_kwargs is None:
                norm_kwargs = dict(percent=99.99, stretch='asinh', asinh_a=0.001)

            if text_kwargs is None:
                text_kwargs = dict(fontsize=16, color='cyan')
            else:
                assert 'transform' not in text_kwargs
                assert ('va' not in text_kwargs) | ('verticalalignment' not in text_kwargs)
                assert ('ha' not in text_kwargs) | ('horizontalalignment' not in text_kwargs)

        phot_list = []
        for loop, image in enumerate(self._data_match):
            if plot:
                ax = axs[loop]
                ax.set_xticklabels([])
                ax.set_yticklabels([])

                img = self._image_list[loop]
                ax.text(0.05, 0.95, f'[{loop}] {img._telescope}-{img._band}',
                        ha='left', va='top', transform=ax.transAxes, **text_kwargs)
            else:
                ax = None

            if error:
                flux, sigma = image_photometry(image, aperture,
                                               calibration_uncertainty=calibration_list[loop],
                                               mask=mask, bkgsub=bkgsub, rannu_in=rannu_in,
                                               rannu_out=rannu_out, error=error,
                                               nsample=nsample,
                                               area_sample=area_sample,
                                               plot=plot, ax=ax,
                                               norm_kwargs=norm_kwargs)
            else:
                flux = image_photometry(image, aperture,
                                        calibration_uncertainty=calibration_list[loop],
                                        mask=mask, bkgsub=bkgsub, rannu_in=rannu_in,
                                        rannu_out=rannu_out, error=error,
                                        nsample=nsample,
                                        area_sample=area_sample,
                                        plot=plot, ax=ax,
                                        norm_kwargs=norm_kwargs)

            if loop > 0:
                ax.get_legend().remove()
            else:
                ax.legend(loc='lower left', bbox_to_anchor=(0, 1), fontsize=16, ncols=2, handlelength=1)

    def clean_image(self, image_index: int = None, model_box_size=None,
                    model_filter_size=3, skip_reduction=False, plot=False,
                    fig=None, axs=None, norm_kwargs=None, interactive=False,
                    verbose=False):
        '''
        Clean up the contaminants in the image.
        '''
        if image_index == None:
            n_image = self._n_image
        else:
            n_image = 1

        if skip_reduction:
            for loop, img in enumerate(self._image_list):
                assert hasattr(img, '_data_subbkg'), \
                    f'Image {loop} ({img}) does not have _data_subbkg'
                assert hasattr(img, '_mask_contaminant'), \
                    f'Image {loop} ({img}) does not have _mask_contaminant'
                assert hasattr(img, '_data_clean'), \
                    f'Image {loop} ({img}) does not have _data_clean'
        else:
            if model_box_size is None:
                model_box_size = int(self._mask_interpolate_scale)

            if image_index == None:
                for loop, img in enumerate(self._image_list):
                    if verbose:
                        print(f'[clean_image] model image {loop}: {img}')

                    # Convert the box size into pixel units for each image
                    model_box_size_pix = int(model_box_size / img._pxs)

                    img.gen_model_galaxy(box_size=model_box_size_pix,
                                         filter_size=model_filter_size,
                                         plot=False)

                    if verbose:
                        print(f'[clean_image] clean image {loop}: {img}')

                    img.gen_image_clean(plot=False)
            else:
                img = self._image_list[image_index]

        if plot:
            if interactive:
                ipy = get_ipython()
                ipy.run_line_magic('matplotlib', 'tk')

                def on_close(event):
                    ipy.run_line_magic('matplotlib', 'inline')

            if axs is None:
                fig, axs = plt.subplots(n_image, 3, figsize=(15, n_image * 5))
                fig.subplots_adjust(wspace=0.03, hspace=0.1)
            else:
                assert fig is not None, 'Please provide fig together with axs!'

            if interactive:
                fig.canvas.mpl_connect('close_event', on_close)

            if norm_kwargs is None:
                norm_kwargs = dict(percent=99.99, stretch='asinh', asinh_a=0.001)

            if image_index == None:
                if n_image != 1:
                    for loop in range(n_image):
                        self.plot_single_image(loop, fig=fig, axs=axs[loop, :], verbose=verbose,
                                               norm_kwargs=norm_kwargs, interactive=False)
                else:
                    self.plot_single_image(loop, fig=fig, axs=axs[:], verbose=verbose,
                                           norm_kwargs=norm_kwargs, interactive=False)
            else:
                self.plot_single_image(image_index, fig=fig, axs=axs, verbose=verbose,
                                       norm_kwargs=norm_kwargs, interactive=False)

    def match_image(self, psf_fwhm, image_size, pixel_scale=None,
                    plot=False, progress_bar=False, verbose=False):
        '''
        Generate the matched images.

        Parameters
        ----------
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
        '''
        images, output_wcs = gen_images_matched(self, psf_fwhm, image_size=image_size,
                                                pixel_scale=pixel_scale,
                                                progress_bar=progress_bar,
                                                verbose=verbose)

        self._data_match = images
        self._wcs_match = output_wcs
        self._pxs_match = np.abs(output_wcs.wcs.cdelt[0]) * 3600
        self._shape_match = output_wcs.pixel_shape

        if plot:
            self.plot_atlas(ncols=3, data_type='data_match', show_info='size', show_units='arcmin', interactive=False)

    def gen_mask_contaminant(self, image_index: int = None, expand_inner=1,
                             expand_edge=1, expand_outer=1, expand_manual=1, 
                             plot=False, fig=None, axs=None, norm_kwargs=None,
                             interactive=False, verbose=False):
        '''
        Generate the contaminant mask.

        Parameters
        ----------
        image_index (optional) : int
            The index of the image to generate the mask_contaminant. Work on all
            images in the list if None.
        expand_inner : float (default: 1)
            The expand_factor of the inner mask.
        expand_edge : float (default: 1.2)
            The expand_factor of the edge mask.
        expand_outer : float (default: 1.2)
            The expand_factor of the outer mask.
        plot : bool (default: False)
            Plot the image and mask if True.
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
        '''
        if image_index == None:
            n_image = self._n_image
        else:
            n_image = 1

        if plot:
            if interactive:
                ipy = get_ipython()
                ipy.run_line_magic('matplotlib', 'tk')

                def on_close(event):
                    ipy.run_line_magic('matplotlib', 'inline')

            if axs is None:
                fig, axs = plt.subplots(n_image, 2, figsize=(10, n_image * 5))
                fig.subplots_adjust(wspace=0.03, hspace=0.1)

                if n_image == 1:
                    img = self._image_list[image_index]
                    axs[1].sharex(axs[0])
                    axs[1].sharey(axs[0])
                    axs[0].set_xticklabels([])
                    axs[0].set_yticklabels([])
                    axs[0].set_ylabel(f'{img._telescope}-{img._band}', fontsize=18)
                else:
                    for loop, img in enumerate(self._image_list):
                        axs[loop, 1].sharex(axs[loop, 0])
                        axs[loop, 1].sharey(axs[loop, 0])
                        axs[loop, 0].set_xticklabels([])
                        axs[loop, 0].set_yticklabels([])
                        axs[loop, 0].set_ylabel(f'{img._telescope}-{img._band}', fontsize=18)

            else:
                assert fig is not None, 'Please provide fig together with axs!'

            if interactive:
                fig.canvas.mpl_connect('close_event', on_close)

            if norm_kwargs is None:
                norm_kwargs = dict(percent=99.99, stretch='asinh', asinh_a=0.001)

        if image_index == None:
            for loop, img in enumerate(self._image_list):
                if plot:
                    axs_u = axs[loop, :]
                else:
                    axs_u = None

                img.gen_mask_contaminant(expand_inner=expand_inner,
                                         expand_edge=expand_edge,
                                         expand_outer=expand_outer, 
                                         expand_manual=expand_manual,
                                         plot=plot, fig=fig, axs=axs_u,
                                         norm_kwargs=norm_kwargs,
                                         interactive=False, verbose=verbose)
        else:
            img = self._image_list[image_index]
            img.gen_mask_contaminant(expand_inner=expand_inner,
                                     expand_edge=expand_edge,
                                     expand_outer=expand_outer, plot=plot,
                                     fig=fig, axs=axs,
                                     norm_kwargs=norm_kwargs,
                                     interactive=interactive, verbose=verbose)

    def plot_atlas(self, ncols=1, data_type='data', show_info: str = None,
                   show_units: str = None, show_mask_target=False, text_kwargs=None, fig=None, axs=None,
                   norm_kwargs=None, interactive=False, verbose=False):
        '''
        Plot the image atlas.
        '''
        if interactive:
            ipy = get_ipython()
            ipy.run_line_magic('matplotlib', 'tk')

            def on_close(event):
                ipy.run_line_magic('matplotlib', 'inline')

        nrows = int(np.ceil(self._n_image / ncols))

        if axs is None:
            fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
            axs = axs.flatten()
            fig.subplots_adjust(wspace=0.05, hspace=0.05)
        else:
            assert fig is not None, 'Please provide fig together with axs!'

        if interactive:
            fig.canvas.mpl_connect('close_event', on_close)

        if norm_kwargs is None:
            norm_kwargs = dict(percent=99.99, stretch='asinh', asinh_a=0.001)

        if text_kwargs is None:
            text_kwargs = dict(fontsize=16, color='cyan')
        else:
            assert 'transform' not in text_kwargs
            assert ('va' not in text_kwargs) | ('verticalalignment' not in text_kwargs)
            assert ('ha' not in text_kwargs) | ('horizontalalignment' not in text_kwargs)

        for loop, ax in enumerate(axs):
            if loop >= self._n_image:
                ax.axis('off')
                continue

            img = self._image_list[loop]

            # The matched data are saved separately in Atlas
            if data_type == 'data_match':
                x = self._data_match[loop]
                pixel_scale = self._pxs_match
            else:
                x = getattr(img, f'_{data_type}', None)
                assert x is not None, f'Cannot find the data type ({data_type})!'
                pixel_scale = img._pxs

            norm = simple_norm(x, **norm_kwargs)
            ax.imshow(x, origin='lower', cmap='Greys_r', norm=norm)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.text(0.05, 0.95, f'[{loop}] {img._telescope}-{img._band}',
                    ha='left', va='top', transform=ax.transAxes, **text_kwargs)

            if show_mask_target:
                assert hasattr(self, '_mask_target'), 'Please use set_mask_match() to set a mask first!'
                xlim = ax.get_xlim();
                ylim = ax.get_ylim()
                plot_mask_contours(self._mask_target, ax=ax, color='magenta', lw=0.5)
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

            if show_info is not None:
                if show_info == 'shape':
                    ax.text(0.05, 0.05, f'Shape: {x.shape}', ha='left',
                            va='bottom', transform=ax.transAxes, **text_kwargs)

                if show_info == 'size':
                    if show_units is not None:
                        size_y, size_x = np.array(x.shape) * pixel_scale * units.arcsec
                        size_x = size_x.to_value(show_units)
                        size_y = size_y.to_value(show_units)

                        ax.text(0.05, 0.05, f'Size: ({size_x:.0f}, {size_y:.0f}) {show_units}',
                                transform=ax.transAxes, va='bottom', ha='left', **text_kwargs)
                    else:
                        size_y, size_x = np.array(x.shape) * pixel_scale
                        ax.text(0.05, 0.05, f'Size: ({size_x:.0f}, {size_y:.0f}) arcsec',
                                transform=ax.transAxes, va='bottom', ha='left', **text_kwargs)

    def plot_single_image(self, image_index, fig=None, axs=None,
                          norm_kwargs=None, interactive=False, verbose=False):
        '''
        Plot single images.
        '''
        img = self._image_list[image_index]

        if interactive:
            ipy = get_ipython()
            ipy.run_line_magic('matplotlib', 'tk')

            def on_close(event):
                ipy.run_line_magic('matplotlib', 'inline')

        if axs is None:
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            fig.subplots_adjust(wspace=0.05, hspace=0.05)
        else:
            assert fig is not None, 'Please provide fig together with axs!'

        if interactive:
            fig.canvas.mpl_connect('close_event', on_close)

        if norm_kwargs is None:
            norm_kwargs = dict(percent=99.99, stretch='asinh', asinh_a=0.001)

        ax = axs[0]
        norm = simple_norm(img._data_subbkg, **norm_kwargs)
        ax.imshow(img._data_subbkg, origin='lower', cmap='Greys_r', norm=norm)
        xlim = ax.get_xlim();
        ylim = ax.get_ylim()
        plot_mask_contours(img._mask_outer, ax=ax, verbose=verbose, color='C0', lw='0.5')
        plot_mask_contours(img._mask_edge, ax=ax, verbose=verbose, color='C1', lw='0.5')
        plot_mask_contours(img._mask_inner, ax=ax, verbose=verbose, color='C2', lw='0.5')
        ax.set_xlim(xlim);
        ax.set_ylim(ylim)
        ax.set_title('Image', fontsize=18)
        ax.set_ylabel(f'{img._telescope}-{img._band}', fontsize=18)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax = axs[1]
        ax.sharex(axs[0])
        ax.sharey(axs[0])
        ax.imshow(img._model_galaxy, origin='lower', cmap='Greys_r', norm=norm)
        ax.set_title('Model galaxy', fontsize=18)

        ax = axs[2]
        ax.sharex(axs[0])
        ax.sharey(axs[0])
        norm = simple_norm(img._data_clean, **norm_kwargs)
        ax.imshow(img._data_clean, origin='lower', cmap='Greys_r', norm=norm)
        ax.set_title('Cleaned image', fontsize=18)

    def remove_background(self, box_fraction=0.02, filter_size=3, verbose=False):
        '''
        Remove the background.
        '''
        for loop, img in enumerate(self._image_list):
            if verbose:
                print(f'[remove_background] image {loop}: {img}')

            box_size = int(img._shape[0] * box_fraction)
            img.gen_model_background(box_size=box_size, filter_size=filter_size)
            img.background_subtract2()

    def set_mask_coverage(self, image_index, mask=None, shape='rect', mask_kwargs=None,
                          fill_value=None, plot=False, fig=None, axs=None, 
                          norm_kwargs=None, interactive=False, verbose=False):
        '''
        Set the coverage mask.

        Parameters
        ----------
        image_index : int
            The index of the image.
        mask (optional) : bool
            The coverage mask with True indicate pixels to be discarded. One can
            manually define the coverage mask with this input.
        shape : string (default: 'rect')
            The shape of the predefined mask with True indicate pixels to be
            discarded. The predefined shapes are circular ('circ') and
            rectangular ('rect').
        mask_kwargs (optional) : dict
            The parameters of the predefined shape function.
            'circ' : x, y, radius
                x, y : The center pixel coordinate of the circle.
                radius : The radius of the circle, units: pixel.
            'rect' : xmin, xmax, ymin, ymax
                The ranges of x and y axes.
        plot : bool (default: False)
            Plot the image and the mask if True.
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
        '''
        img = self._image_list[image_index]
        img.set_mask_coverage(mask, shape=shape, mask_kwargs=mask_kwargs, 
                              fill_value=fill_value, plot=plot, fig=fig, axs=axs, 
                              norm_kwargs=norm_kwargs, interactive=interactive, 
                              verbose=verbose)

    def set_mask_match(self, mask, mask_type='mask_target', input_wcs=None, verbose=False):
        '''
        Set the mask of the matched data.

        Parameters
        ----------
        mask : 2D array
            Input mask
        mask_type : string
            The type of the mask, 'mask_target' or 'mask_contaminant'.
        input_wcs (optional) : WCS
            The input WCS. If specified, the mask will be reprojected to match
            the Atlas's WCS.
        verbose : bool (default: True)
            Show details if True.
        '''
        assert mask_type in ['mask_target', 'mask_contaminant'], f'Cannot find the mask type ({mask_type})!'

        if input_wcs is not None:
            mask = adapt_mask(mask, input_wcs, self._wcs_match, self._shape_match, verbose=verbose)

        setattr(self, f'_{mask_type}', mask)

    def __getitem__(self, key):
        '''
        Get the item of the image list.

        Parameters
        ----------
        key : int
            The index of the image list.

        Notes
        -----
        FIXME: We can make it more useful.
        '''
        return self._image_list[key]

    def __len__(self):
        '''
        Get the length of the image list.
        '''
        return self._n_image

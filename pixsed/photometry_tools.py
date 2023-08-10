import random
from math import log, sqrt, ceil, log10
import os

import astropy.units as u
import extinction
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
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
from photutils.background import Background2D, MedianBackground
from photutils.detection import DAOStarFinder
from photutils.profiles import RadialProfile, CurveOfGrowth
from photutils.segmentation import detect_sources, make_2dgaussian_kernel, deblend_sources, SourceCatalog, SourceFinder
from reproject import reproject_adaptive
from scipy import interpolate

from .utils import read_coordinate, plot_image, circular_error_estimate
from .utils import xmatch_gaiadr3

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

    def __init__(self, data, header, psf_fwhm, target_coordinate, id, wavelength=None,
                 filter_name=None, telescope_name=None):
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
        self._data_org = data
        self._data = data.copy()
        self._header = header
        self._shape = data.shape
        self._wcs = WCS(header)
        self._pxs = (abs(self._header['CDELT1']) * 3600.)
        self._id = id
        self._data_subbkg = None
        self._image_cleaned_simple = None
        self._image_cleaned = None
        self._psf_oversample = None
        self._mask_extend = None
        self._mask_point = None
        self._mask_stars = None
        self._mask_galaxy = None
        self._background_model = None
        self._image_cleaned_background = None
        self._sources_overlap = None
        self._sources_segmentation = None
        self._galaxy_model = None
        self._segmentation = None
        self._bkg_mean = None
        self._bkg_median = None
        self._bkg_std = None
        self._isolist = None
        self._ellipse_model = None
        self._data_sources = None

        self._wavelength = wavelength

        self._filter = filter_name

        self._telescope = telescope_name

        self._psf_FWHM = psf_fwhm / self._pxs

        w = self._wcs
        self._ra, self._dec = target_coordinate
        self._coord = read_coordinate(self._ra, self._dec)
        ra_pix, dec_pix = w.world_to_pixel(self._coord)
        self._coord_pix = (float(ra_pix), float(dec_pix))

    def background_model(self, box_size=None, filter_size=None):
        """
        Generate the background model.Using median background method.
        Parameter
        ----------
        box_size: int.
            The size used to calculate the local median.
            If None, the default value is 25 times fwhm.
        filter_size: int.
            The kernel size used to smooth the background model.
            If None, the default value is 3 pixels.
        """
        if not box_size:
            box_size = int(self._psf_FWHM * 25)
        if not filter_size:
            filter_size = 3
        img = self._data.copy()
        mask_background = self._mask_background
        sigma_clip = SigmaClip(sigma=3.)
        bkg_estimator = MedianBackground()
        bkg = Background2D(img, (box_size, box_size), mask=mask_background, filter_size=(filter_size, filter_size),
                           sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
        self._background_model = bkg.background

    def plot_background_model(self, percentile=98.):
        """
        Plot the background model.
        Parameter
        -----------
        percentile: float.
            The percentile of normalization.
        """
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
        ax = ax.ravel()
        norm = simple_norm(self._data, percent=percentile, stretch='asinh')
        ax[0].imshow(self._data, cmap='gray', norm=norm, origin='lower')
        ax[1].imshow(self._background_model, cmap='gray', origin='lower')

    def background_properties(self, mask_type='quick', sigma=3, maxiters=5, **kwargs):
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
        **kwargs : Other parameters of sigma_clipped_stats()

        Returns
        -------
        mean, median, stddev : floats
            The returns of sigma_clipped_stats().
        '''
        if mask_type == 'quick':
            mask = None
        elif mask_type == 'segmentation':
            if not hasattr(self, '_mask_segmentation'):
                raise ValueError(
                    'The background mask (_mask_segmentation) is not generated! Please run mask_segmentation()!')
            else:
                mask = self._mask_segmentation
        else:
            if not hasattr(self, '_mask_background'):
                raise ValueError(
                    'The background mask (_mask_background) is not generated! Please run mask_background()!')
            mask = self._mask_background

        res = sigma_clipped_stats(self._data, mask=mask, sigma=sigma, maxiters=maxiters, **kwargs)
        self._bkg_mean, self._bkg_median, self._bkg_std = res
        return self._bkg_mean, self._bkg_median, self._bkg_std

    def background_subtract(self, method='full'):
        """
        Remove the background of the image data.
        Parameter
        ----------
        method: str.
                If the method is 'simple', the function just uses the median of the image to do the subtraction.
                If the method is 'full', the function  uses the background model to do the subtraction.
        """
        self._data_subbkg = self._data.copy()
        if method == 'simple':
            assert hasattr(self, '_bkg_median'), 'Please run background_properties() first!'
            self._data_subbkg -= self._bkg_median
        elif method == 'full':
            assert hasattr(self, '_background_model'), 'Please run background_model() first!'
            self._data_subbkg -= self._background_model

    def plot_background_subtract(self, percentile1=98., percentile2=98.):
        """
        Plot the background subtracted image.
        Parameter
        -----------
        percentile1: float
        percentile2: float
        """
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
        ax = ax.ravel()
        norm1 = simple_norm(self._data, percent=percentile1, stretch='asinh')
        ax[0].imshow(self._data, cmap='gray', norm=norm1, origin='lower')
        norm2 = simple_norm(self._data_subbkg, percent=percentile2, stretch='asinh')
        ax[1].imshow(self._data_subbkg, cmap='gray', norm=norm2, origin='lower')

    def detect_segmentation(self, threshold, fwhm=3, size=5, npixels=10,
                            deblend=False, nlevels=32, contrast=0.001,
                            progress_bar=False):
        '''
        Detect the source with image segmentation.

        Parameters
        ----------
        threshold : float
            The threshold of the source detection.
        fwhm : float (default: 3)
            The FWHM of the Gaussian kernel.
        size : int (default: 5)
            The kernel size.
        npixel : int (default: 10)
            The number of pixels.
        '''
        kernel = make_2dgaussian_kernel(fwhm=fwhm, size=size)  # FWHM = 3.0
        convolved_data = convolve(self._data, kernel)
        segment_map = detect_sources(convolved_data, threshold, npixels=npixels)

        if deblend:
            self._segmentation = deblend_sources(convolved_data, segment_map, npixels=npixels,
                                                 nlevels=nlevels, contrast=contrast, progress_bar=progress_bar)
        else:
            self._segmentation = segment_map

        self._segmentation_catalog = SourceCatalog(self._data, self._segmentation,
                                                   convolved_data=convolved_data)

    def get_sources_overlap(self, detect_thres=15., xmatch_radius=3., xmatch_table='vizier:I/355/gaiadr3',
                            xmatch_gmag=20., xmatch_distance=23., xmatch_plxerror=0.3, careful=False, safe_radius=5.):
        '''
        Get the foreground stars that overlap with the target galaxy.
        Parameter
        -----------
        detect_thres: float.
            The threshold for DAOFinder.
        xmatch_radius: float.
            The xmatch radius. arcsec.
        xmatch_table: str.
            The Gaia ctalog table used for xmatch.
        xmatch_gmag: float.
            The max G band magnitude of the stars which will be picked out.
        xmatch_distance: float.
            The max distance of the stars which will be picked out. kpc.
        xmatch_plxerror: float.
            The max rate of the error of parallax over the parallax.
        Return
        ---------
        A table contains Ra, Dec, and Gmag.
        '''
        assert hasattr(self, '_mask_background'), 'Please run mask_background() first!'
        assert hasattr(self, '_data_subbkg'), 'Please run background_subtract() first!'
        # initial detection
        mask_galaxy = self._mask_galaxy
        daofind = DAOStarFinder(threshold=detect_thres * self._bkg_std, fwhm=self._psf_FWHM)
        sources = daofind(self._data_subbkg)
        # Gaia Xmatch
        w = self._wcs
        sources_world = Table(names=['Ra', 'Dec'])  # build up table used for xmatch.
        if sources:
            for i in range(len(sources)):
                x = int(sources['xcentroid'][i])
                y = int(sources['ycentroid'][i])
                if not mask_galaxy[y, x]:
                    continue
                sky = w.pixel_to_world(sources['xcentroid'][i], sources['ycentroid'][i])
                sources_world.add_row([sky.ra, sky.dec])
            t_o = XMatch.query(cat1=sources_world,
                               cat2=xmatch_table,
                               max_distance=xmatch_radius * u.arcsec, colRA1='Ra', colDec1='Dec')  # Gaia xmatch.
            front_stars_set = set()
            for i in range(len(t_o)):  # Use some standards to judge, cleaning out the fake detections.
                if t_o['Plx'][i] != '--' and t_o['Plx'][i] > 0. and t_o['Gmag'][i] < xmatch_gmag:
                    if 1 / t_o['Plx'][i] < xmatch_distance and t_o['e_Plx'][i] < xmatch_plxerror * t_o['Plx'][i]:
                        front_stars_set.add((t_o['Ra'][i], t_o['Dec'][i], t_o['Gmag'][i]))

            if careful:
                for i in range(len(t_o)):
                    if t_o['PSS'][i] != '--':
                        if t_o['PSS'][i] > 0.99:
                            if t_o['Gmag'][i] < 15:
                                front_stars_set.add((t_o['Ra'][i], t_o['Dec'][i], t_o['Gmag'][i]))

        else:
            front_stars_set = set()
        front_stars_list = list(front_stars_set)
        front_stars_world = Table(
            names=['Ra', 'Dec', 'Gmag', 'mean', 'std', 'radius'])  # Build up the final foreground stars table.
        for j in range(len(front_stars_set)):
            x_t, y_t = w.world_to_pixel(SkyCoord(front_stars_list[j][0], front_stars_list[j][1], unit='deg'))
            center_distance = sqrt((x_t - self._coord_pix[0]) ** 2 +
                                   (y_t - self._coord_pix[1]) ** 2) * self._pxs
            if center_distance <= safe_radius:
                continue
            front_stars_world.add_row([front_stars_list[j][0], front_stars_list[j][1], front_stars_list[j][2], 0, 0, 0])

        self._sources_overlap = front_stars_world

    
    def get_sources_overlap2(self, detect_thres=15., xmatch_radius=3., threshold_gmag=25., 
                             threshold_pss=0.95, threshold_plx=2, center_radius=5.):
        '''
        Get the foreground stars that overlap with the target galaxy.
        Parameter
        -----------
        detect_thres: float
            The threshold for DAOFinder.
        xmatch_radius: float
            The xmatch radius. arcsec.
        threshold_gmag: float
            The max G band magnitude of the stars which will be picked out.
        threshold_pss (optional): float
            The threshold to select stars according to Gaia probability of single star.
        threshold_plx (optional): float
            The threshold to select stars according to Gaia parallax SNR.
        
        Return
        ---------
        A table contains Ra, Dec, and Gmag.
        '''
        assert hasattr(self, '_mask_background'), 'Please run mask_background() first!'
        assert hasattr(self, '_data_subbkg'), 'Please run background_subtract() first!'

        # initial detection
        mask_galaxy = self._mask_galaxy
        daofind = DAOStarFinder(threshold=detect_thres * self._bkg_std, fwhm=self._psf_FWHM)
        sources = daofind(self._data_subbkg, mask=~mask_galaxy)

        # Gaia Xmatch
        w = self._wcs
        if sources:
            c = w.pixel_to_world(sources['xcentroid'], sources['ycentroid'])
            sources_world = Table([c.ra.deg, c.dec.deg], names=['ra', 'dec'])
            t_o = xmatch_gaiadr3(sources_world, xmatch_radius, colRA1='ra', colDec1='dec')  # Gaia xmatch.

            fltr = (t_o['Gmag'] < threshold_gmag)

            if threshold_pss is not None:
                fltr &= (t_o['PSS'] > threshold_pss) & ~t_o['PSS'].mask
            
            if threshold_plx is not None:
                plx_snr = (t_o['Plx'] / t_o['e_Plx']) 
                fltr &= (plx_snr > threshold_plx) & ~t_o['Plx'].mask

            t_s = t_o[fltr]

            null_array = np.zeros(len(t_s))
            front_stars_set = Table([t_s['ra'], t_s['dec'], t_s['Gmag'], null_array, null_array, null_array], 
                                    names=['Ra', 'Dec', 'Gmag', 'mean', 'std', 'radius'])

            x_t, y_t = w.world_to_pixel(SkyCoord(t_s['ra'], t_s['dec'], unit='deg'))
            center_distance = np.sqrt((x_t - self._coord_pix[0]) ** 2 + 
                                      (y_t - self._coord_pix[1]) ** 2) * self._pxs
            fltr = center_distance > center_radius
            self._sources_overlap = front_stars_set[fltr]
        else:
            self._sources_overlap = Table(names=['Ra', 'Dec', 'Gmag', 'mean', 'std', 'radius'])  # Build up the final foreground stars table.


    def plot_sources_overlap(self, percentile=98., xlim=None, ylim=None, zoomin=None):
        w = self._wcs
        front_stars_world = self._sources_overlap
        plot_tb = Table(names=['x', 'y'])
        for i in range(len(front_stars_world)):
            plot_sky_coord = SkyCoord(front_stars_world[i]['Ra'], front_stars_world[i]['Dec'], unit='deg')
            plot_x, plot_y = w.world_to_pixel(plot_sky_coord)
            plot_tb.add_row([plot_x, plot_y])
        positions = np.transpose((plot_tb['x'], plot_tb['y']))
        apertures = CircularAperture(positions, r=4.0)

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
        ax = ax.ravel()
        norm = simple_norm(self._data_subbkg, percent=percentile, stretch='asinh')
        ax[0].imshow(self._data_subbkg, cmap='gray', norm=norm, origin='lower')
        ax[1].imshow(self._data_subbkg, cmap='gray', norm=norm, origin='lower')
        apertures.plot(color='red', lw=5, alpha=0.5)
        if xlim and ylim:
            ax[0].set_xlim(xlim[0], xlim[1])
            ax[0].set_ylim(ylim[0], ylim[1])
            ax[1].set_xlim(xlim[0], xlim[1])
            ax[1].set_ylim(ylim[0], ylim[1])
        elif zoomin:
            rate = 1 / (zoomin / 100)
            xl = np.shape(self._data)[0]
            yl = np.shape(self._data)[1]
            ax[0].set_xlim(int(xl / 2 - rate * (xl / 2)), int(xl / 2 + rate * (xl / 2)))
            ax[0].set_ylim(int(yl / 2 - rate * (yl / 2)), int(yl / 2 + rate * (yl / 2)))
            ax[1].set_xlim(int(xl / 2 - rate * (xl / 2)), int(xl / 2 + rate * (xl / 2)))
            ax[1].set_ylim(int(yl / 2 - rate * (yl / 2)), int(yl / 2 + rate * (yl / 2)))

    def get_isophote(self, plot=False):
        '''
        Fit the isophote of the target and generate the ellipse model.
        (May include more complicated functions.)

        Parameters
        ----------
        plot : bool (default: False)
            Plot the isophote fitting results.
        '''
        self._isolist = None
        self._ellipse_model = None

    def get_psf(self, thres=20., half_length=10, oversample=4, iter=3, nsamples=300):
        '''
        Get the PSF model. (May include other properties of the PSF.)
        
        Parameters
        ----------
        method: str
            if method is 'direct', it will get the psf fwhm directly from the Clark's paper.
            if method is 'full', it will build up the psf model from the image.
        plot : bool (default: False)
            Plot the information to show the psf building procedure.
        '''

        hl = int(half_length * self._psf_FWHM)  # pixels
        good_label = []
        mea, med, std = sigma_clipped_stats(self._data_subbkg, mask=self._mask_galaxy + self._mask_extend)
        daofind = DAOStarFinder(fwhm=self._psf_FWHM, threshold=thres * std)
        sources = daofind(self._data_subbkg, mask=self._mask_galaxy + self._mask_extend)
        # step 1
        flux_box = sources['flux']
        peak_box = sources['peak']
        fmea, fmed, fstd = sigma_clipped_stats(np.array(flux_box))
        pmea, pmed, pstd = sigma_clipped_stats(np.array(peak_box))
        for i in range(len(sources)):
            if fmea - 3 * fstd <= sources[i]['flux'] <= fmea + 3 * fstd and pmea - 3 * pstd <= sources[i][
                'peak'] <= pmea + 3 * pstd:
                good_label.append(i)
        # step 2
        good_label1 = []
        for label in good_label:
            xc = int(sources[label]['xcentroid'])
            yc = int(sources[label]['ycentroid'])
            if xc < 3 * hl or xc > np.shape(self._data_subbkg)[0] - 3 * hl or yc < 3 * hl or yc > np.shape(
                    self._data_subbkg)[1] - 3 * hl:
                continue
            good_label1.append(label)
        # step 3
        nddata = NDData(data=self._data_subbkg)
        selected_stars_tbl = Table(names=['x', 'y'])
        count = 0
        if not nsamples:
            nsamples = len(good_label1)
        for label in good_label1:
            if count > nsamples:
                break
            temp = [sources[label]['xcentroid'], sources[label]['ycentroid']]
            selected_stars_tbl.add_row(temp)
            count += 1
        selected_stars = extract_stars(nddata, selected_stars_tbl, size=2 * hl - 1)
        epsf_builder = EPSFBuilder(oversampling=oversample, maxiters=iter, smoothing_kernel='quadratic',
                                   progress_bar=False)
        epsf, fitted_stars = epsf_builder(selected_stars)
        self._psf = epsf.data
        self._psf_oversample = oversample

    def plot_psf_model(self, percentile=99.):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
        ax = ax.ravel()
        psf_model = self._psf
        norm = simple_norm(psf_model, stretch='sqrt', percent=percentile)
        ax[0].imshow(psf_model, norm=norm, origin='lower')
        max_radius = np.shape(psf_model)[0] // 2 - 1
        edge_radii = [radii for radii in range(int(max_radius) + 1)]
        xc = (np.shape(psf_model)[0] - 1) // 2
        yc = (np.shape(psf_model)[1] - 1) // 2
        rp = RadialProfile(psf_model, (xc, yc), edge_radii,
                           error=None, mask=None)
        x = rp.radius
        x = [self._pxs * (i / self._psf_oversample) for i in x]
        y = rp.profile
        ax[1].plot(x, y)
        ax[1].set_xlabel('radius / asec')
        ax[1].set_ylabel('intense')

    def get_radial_profile(self, data, ra_max, step=1., mask=None, kind='linear', plot=False):
        '''
        Get the radial surface bright profile of the target.

        Parameters
        ----------
        data: 2d array
            The data needs to be measured.
        ra_max: float.
            The max radius for measurement.
        step: float. optional.
            The step length for measurement.
        mask: 2d array. Containing bool. optional.
            The mask for measurement.
        kind: str. optional.
            The method for smoothing the curve. ('linear', 'cubic' ...)
        plot : bool (default: False)
            Plot the radial profile.
        '''
        edge_radii = [radii for radii in range(0, int(ra_max) + 1, step)]
        rp = RadialProfile(data, self._coord_pix, edge_radii,
                           error=None, mask=mask)
        x = rp.radius
        y = rp.profile
        self._rp = interpolate.interp1d(x, y, kind=kind)

        if plot:
            self._rp.plot()

    def mask_background(self):
        '''
        Get the mask of the target galaxy and all the sources on the background.
        Parameter
        ----------
        plot:bool.
            if True,it will plot the background mask.
        '''
        mask_stars = self._mask_stars
        mask_galaxy = self._mask_galaxy
        self._mask_background = mask_stars + mask_galaxy

    def plot_mask_background(self, percentile=98.):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
        ax = ax.ravel()
        norm = simple_norm(self._data, stretch='asinh', percent=percentile)
        ax[0].imshow(self._data, norm=norm, cmap='gray', origin='lower')
        ax[1].imshow(self._mask_background, cmap='gray', origin='lower')

    def mask_galaxy(self, thres=1., iter=10, kernel_size=None):
        '''
        Get the mask of the target galaxy.
        Assuming that there is only one target object on the background.
        If the coord of the object is None, then the target which has the largest
        area will be taken into consideration.
        Parameter
        ----------
        thres: float.
            The threshold of the segmentation.
        iter: int.
            The number of times to convolve the extension mask.
        kernel_fwhm: float.
            The fwhm of the convolution kernel.
        kernel_size: int.
            The size of the convolution kernel.
        expand: float.
            The smaller, the more expansive. Should be less than 1.0 .
        plot: bool.
        '''
        if not kernel_size:
            psf = self._psf_FWHM  # pixels
            psf = ceil(psf)
            if psf % 2 != 1:
                psf += 1
            kernel_size = psf
        mea, med, std = self.background_properties()
        img_sub = self._data - med
        kernel = make_2dgaussian_kernel(fwhm=3, size=5)
        convolved_data = convolve(img_sub, kernel)
        segment_map = detect_sources(convolved_data, thres * std, npixels=12)
        cat = SourceCatalog(img_sub, segment_map, convolved_data=convolved_data)
        sources_tb = cat.to_table()
        if self._coord:
            x, y = [int(i) for i in self._coord_pix]
            label = segment_map.data[y, x]
        else:
            max_area = 0.
            label = 0
            for i in range(len(sources_tb)):
                if sources_tb[i]['area'] > max_area:
                    max_area = sources_tb[i]['area']
                    label = sources_tb[i]['label']
        mask_galaxy = segment_map == label
        kernel_galaxy = make_2dgaussian_kernel(7, size=kernel_size)
        for i in range(iter):
            mask_galaxy = convolve(mask_galaxy, kernel_galaxy)
            mask_galaxy = mask_galaxy != 0
        self._mask_galaxy = mask_galaxy

    def plot_mask_galaxy(self, percentile=98.):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
        ax = ax.ravel()
        norm = simple_norm(self._data, stretch='asinh', percent=percentile)
        ax[0].imshow(self._data, norm=norm, cmap='gray', origin='lower')
        ax[1].imshow(self._mask_galaxy, cmap='gray', origin='lower')

    def mask_stars(self, thres=4., thres_area=100, iter_point=3, iter_extend=5,
                   kernel_size_point=None, kernel_size_extend=None):
        '''
        Get the mask of the background stars sources, including the expansive stars as well as the point stars.
        Parameter
        ----------
        thres: float.
            The threshold of the segmentation.
        thres_area: int.
            The threshold area between point sources and extensive sources.
        iter_point: int.
            The number of times to convolve the point stars mask.
        iter_extend: int.
            The number of times to convolve the expansive stars mask.
        kernel_fwhm_point: float.
            The fwhm of the convolution kernel for point sources.
        kernel_size_point: int.
            The size of the convolution kernel for point sources.
        kernel_fwhm_point: float.
            The fwhm of the convolution kernel for extensive sources.
        kernel_size_point: int.
            The size of the convolution kernel for extensive sources.
        plot: bool.
        '''
        assert hasattr(self, '_mask_galaxy'), 'Please run mask_galaxy() first!'
        if not kernel_size_point:
            psf = self._psf_FWHM  # pixels
            psf = ceil(psf)
            if psf % 2 != 1:
                psf += 1
            kernel_size_point = psf
        if not kernel_size_extend:
            psf = self._psf_FWHM  # pixels
            psf = ceil(psf)
            if psf % 2 != 1:
                psf += 1
            kernel_size_extend = psf
        galaxy_mask = self._mask_galaxy
        mea, med, std = self._bkg_mean, self._bkg_median, self._bkg_std
        img_sub = self._data - med

        kernel = make_2dgaussian_kernel(fwhm=self._psf_FWHM, size=5)
        convolved_data = convolve(img_sub, kernel)
        segment_map = detect_sources(convolved_data, thres * std, npixels=12, mask=galaxy_mask)
        cat = SourceCatalog(img_sub, segment_map, convolved_data=convolved_data)
        sources_tb = cat.to_table()
        extend_label = []
        point_label = []
        for i in range(len(sources_tb)):
            if sources_tb[i]['area'].value >= thres_area * np.pi * (self._psf_FWHM ** 2):
                extend_label.append(sources_tb[i]['label'])
            else:
                point_label.append(sources_tb[i]['label'])
        mask_extend = np.zeros(self._shape, dtype=bool)
        mask_point = np.zeros(self._shape, dtype=bool)
        for i in extend_label:
            mask_extend = mask_extend + (segment_map.data == i)
        mask_point = segment_map.data != 0
        kernel_point = make_2dgaussian_kernel(7., size=kernel_size_point)
        kernel_extend = make_2dgaussian_kernel(7., size=kernel_size_extend)

        for i in range(iter_extend):
            mask_extend = convolve(mask_extend, kernel_extend)
            mask_extend = mask_extend != 0

        for i in range(iter_point):
            mask_point = convolve(mask_point, kernel_point)
            mask_point = mask_point != 0

        mask_stars = mask_point + mask_extend
        self._mask_stars = mask_stars
        self._mask_point = mask_point
        self._mask_extend = mask_extend

    def plot_mask_stars(self, percentile=98.):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
        ax = ax.ravel()
        norm = simple_norm(self._data, stretch='asinh', percent=percentile)
        ax[0].imshow(self._data, norm=norm, cmap='gray', origin='lower')
        ax[1].imshow(self._mask_stars, cmap='gray', origin='lower')

    def mask_segmentation(self, kernel_fwhm=None, kernel_size=5):
        '''
        Get the source mask directly from the image segmentation.
        '''
        mask = self._segmentation.data > 0

        if kernel_fwhm is None:
            self._mask_segmentation = mask
        else:
            kernel_size = max([kernel_size, 2 * kernel_fwhm + 1])
            kernel = make_2dgaussian_kernel(fwhm=kernel_fwhm, size=kernel_size)
            self._mask_segmentation = convolve(mask, kernel) > 0.1

    def mask_target_isophote(self):
        '''
        Get the mask of the target based on the source isophote.
        '''
        self._mask_target = None

    def plot_data(self, ax=None, percentile=99.5, vmin=None, vmax=None, stretch=None,
                  origin='lower', cmap='gray_r', show_target=True, **kwargs):
        """
        Plot the image data.

        Parameters
        ----------
        ax : Figure axis
            The axis handle of the figure.
        vmin : float
            The minimum scale of the image.
        vmax : float
            The maximum scale of the image.
        stretch : stretch object
            The stretch used to normalize the image color scale.
        origin : string (default: 'lower')
            The origin of the image.
        cmap : string (default: 'gray_r')
            The colormap.
        show_target : bool (default: True)
            Mark the target position if True.
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
        ax = plot_image(self._data, ax=ax, percentile=percentile, vmin=vmin, vmax=vmax, stretch=stretch,
                        origin=origin, cmap=cmap, **kwargs)

        if show_target:
            ax.plot(self._coord_pix[0], self._coord_pix[1], marker='+', color='r', ms=10)
        return ax

    def plot_segmentation(self, ax=None):
        '''
        Plot the segmentation.
        '''
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))

        ax.imshow(self._segmentation, origin='lower', cmap=self._segmentation.cmap)
        return ax

    def remove_sources_simple(self, l_half=18, catalog=None, min_ra=3, max_ra=50):
        '''
        Remove the contaminating sources in the image. Should run mask_stars() first.
        Parameter
        -----------
        l: int.The length of the local box used for estimating the local background.
        catalog: table. The world coorde of the stars that overlap with the target galaxy.
        plot: bool.
            If True, will plot the sources removed image.
        percent: float.
        xlim, ylim: tuple.
        Return
        -----------
        2d array. The cleaned image.
        '''
        assert hasattr(self, '_bkg_median'), 'Please run background_properties() first!'
        assert hasattr(self, '_data_subbkg'), 'Please run background_subtract() first!'
        assert hasattr(self, '_mask_stars'), 'Please run mask_stars() first!'
        assert hasattr(self, '_sources_overlap'), 'Please run get_sources_overlap() first!'
        image_cleaned = self._data_subbkg.copy()
        fwhm = self._psf_FWHM
        length = int(l_half * self._psf_FWHM)
        # clean stars on the background.
        x_len = self._shape[0]
        y_len = self._shape[1]
        new_bkg_mea, new_bkg_med, new_bkg_std = sigma_clipped_stats(self._data_subbkg, mask=self._mask_background)
        for xc in range(x_len):
            for yc in range(y_len):
                if self._mask_stars[xc, yc]:
                    image_cleaned[xc, yc] = random.gauss(new_bkg_mea, new_bkg_std)
        self._image_cleaned_background = image_cleaned.copy()
        # clean stars that overlap with the target galaxy.
        if catalog is None:
            cat_world = self._sources_overlap
        else:
            cat_world = catalog
        w = self._wcs
        for i in range(len(cat_world)):
            sky = SkyCoord(cat_world['Ra'][i], cat_world['Dec'][i], frame='icrs', unit='deg')
            x, y = w.world_to_pixel(sky)
            xc, yc = int(x), int(y)
            if ((xc - length <= 0) or (yc - length <= 0)) or (
                    (xc + 2 * length >= np.shape(image_cleaned)[0]) or (yc + 2 * length >= np.shape(image_cleaned)[1])):
                continue
            sample = image_cleaned[yc - length:yc + length + 1, xc - length:xc + length + 1]
            mask_sample = np.zeros(np.shape(sample), dtype='bool')
            for a in range(-int(length / 3), int(length / 3) + 1):
                for b in range(-int(length / 3), int(length / 3) + 1):
                    if sqrt(a ** 2 + b ** 2) <= length / 3:
                        mask_sample[length + a, length + b] = True
            mea_sample, med_sample, std_sample = sigma_clipped_stats(sample, maxiters=20, mask=mask_sample)
            edge_radii = [radii for radii in range(0, int(length) + 1)]
            rp = RadialProfile(self._data_subbkg, (xc, yc), edge_radii)
            h = rp.profile
            r = rp.radius
            flag = True
            clean_ra = 0
            for d in range(int(fwhm), len(h)):
                if (h[d] < mea_sample + 2 * std_sample) and flag:
                    clean_ra = int(r[d])
                    flag = False
            if clean_ra < min_ra:
                clean_ra = int(2 * self._psf_FWHM)
            if clean_ra > max_ra:
                continue
            big_ra = 3 * clean_ra
            big_ap = []
            for a in range(-big_ra, big_ra + 1):
                for b in range(-big_ra, big_ra + 1):
                    if big_ra > sqrt(a ** 2 + b ** 2) > clean_ra:
                        big_ap.append(image_cleaned[yc + b, xc + a])
            mea_big_ap, med_big_ap, std_big_ap = sigma_clipped_stats(np.array(big_ap))
            cat_world[i]['mean'] = mea_big_ap
            cat_world[i]['std'] = std_big_ap
            cat_world[i]['radius'] = clean_ra
            for a in range(-clean_ra, clean_ra + 1):
                for b in range(-clean_ra, clean_ra + 1):
                    if sqrt(a ** 2 + b ** 2) <= clean_ra:
                        image_cleaned[yc + b, xc + a] = random.gauss(mea_big_ap, std_big_ap)

        self._image_cleaned_simple = image_cleaned
        self._sources_overlap = cat_world

    def plot_remove_sources_simple(self, percentile=99., xlim=None, ylim=None, zoomin=None):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
        ax = ax.ravel()
        norm = simple_norm(self._data_subbkg, percent=percentile, stretch='asinh')
        ax[0].imshow(self._data_subbkg, cmap='gray', norm=norm, origin='lower')
        ax[1].imshow(self._image_cleaned_simple, cmap='gray', norm=norm, origin='lower')
        if xlim and ylim:
            ax[0].set_xlim(xlim[0], xlim[1])
            ax[0].set_ylim(ylim[0], ylim[1])
            ax[1].set_xlim(xlim[0], xlim[1])
            ax[1].set_ylim(ylim[0], ylim[1])
        elif zoomin:
            rate = 1 / (zoomin / 100)
            xl = np.shape(self._data_subbkg)[0]
            yl = np.shape(self._data_subbkg)[1]
            ax[0].set_xlim(int(xl / 2 - rate * (xl / 2)), int(xl / 2 + rate * (xl / 2)))
            ax[0].set_ylim(int(yl / 2 - rate * (yl / 2)), int(yl / 2 + rate * (yl / 2)))
            ax[1].set_xlim(int(xl / 2 - rate * (xl / 2)), int(xl / 2 + rate * (xl / 2)))
            ax[1].set_ylim(int(yl / 2 - rate * (yl / 2)), int(yl / 2 + rate * (yl / 2)))

    def get_galaxy_model(self, boxsize=None, filter_size=None):
        if not boxsize:
            psf = self._psf_FWHM
            psf = ceil(psf)
            boxsize = 2 * psf - 1
        if not filter_size:
            filter_size = boxsize // 4 * 2 + 1
        sample = self._image_cleaned_simple.copy()
        sigma_clip = SigmaClip(sigma=3.)
        bkg_estimator = MedianBackground()
        bkg = Background2D(sample, (boxsize, boxsize), filter_size=(filter_size, filter_size),
                           sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

        self._galaxy_model = bkg.background

    def plot_galaxy_model(self, percentile=99.9, xlim=None, ylim=None, zoomin=None):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
        ax = ax.ravel()
        norm = simple_norm(self._image_cleaned_simple, percent=percentile, stretch='asinh')
        ax[0].imshow(self._image_cleaned_simple, cmap='gray', norm=norm, origin='lower')
        ax[1].imshow(self._galaxy_model, cmap='gray', norm=norm, origin='lower')
        if xlim and ylim:
            ax[0].set_xlim(xlim[0], xlim[1])
            ax[0].set_ylim(ylim[0], ylim[1])
            ax[1].set_xlim(xlim[0], xlim[1])
            ax[1].set_ylim(ylim[0], ylim[1])
        elif zoomin:
            rate = 1 / (zoomin / 100)
            xl = np.shape(self._image_cleaned_simple)[0]
            yl = np.shape(self._image_cleaned_simple)[1]
            ax[0].set_xlim(int(xl / 2 - rate * (xl / 2)), int(xl / 2 + rate * (xl / 2)))
            ax[0].set_ylim(int(yl / 2 - rate * (yl / 2)), int(yl / 2 + rate * (yl / 2)))
            ax[1].set_xlim(int(xl / 2 - rate * (xl / 2)), int(xl / 2 + rate * (xl / 2)))
            ax[1].set_ylim(int(yl / 2 - rate * (yl / 2)), int(yl / 2 + rate * (yl / 2)))

    def get_sources_segmentation(self, thres=3., deblend_contrast=0.001, deblend_nlevels=8):
        sources = self._data_subbkg - self._galaxy_model
        sources -= sigma_clipped_stats(sources)[0]
        mea, med, std = sigma_clipped_stats(sources)
        kernel = make_2dgaussian_kernel(self._psf_FWHM, size=5)
        convolved_data = convolve(sources, kernel)
        finder = SourceFinder(npixels=10, contrast=deblend_contrast, nlevels=deblend_nlevels, progress_bar=False)
        segment_map = finder(convolved_data, thres * std)

        self._sources_segmentation = segment_map
        self._data_sources = sources

    def plot_sources(self, percentile=99., xlim=None, ylim=None, zoomin=None):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
        ax = ax.ravel()
        norm1 = simple_norm(self._data_subbkg, percent=percentile, stretch='asinh')
        ax[0].imshow(self._data_subbkg, cmap='gray', norm=norm1, origin='lower')
        norm2 = simple_norm(self._data_sources, percent=percentile, stretch='asinh')
        ax[1].imshow(self._data_sources, cmap='gray', norm=norm2, origin='lower')
        if xlim and ylim:
            ax[0].set_xlim(xlim[0], xlim[1])
            ax[0].set_ylim(ylim[0], ylim[1])
            ax[1].set_xlim(xlim[0], xlim[1])
            ax[1].set_ylim(ylim[0], ylim[1])
        elif zoomin:
            rate = 1 / (zoomin / 100)
            xl = np.shape(self._data_subbkg)[0]
            yl = np.shape(self._data_subbkg)[1]
            ax[0].set_xlim(int(xl / 2 - rate * (xl / 2)), int(xl / 2 + rate * (xl / 2)))
            ax[0].set_ylim(int(yl / 2 - rate * (yl / 2)), int(yl / 2 + rate * (yl / 2)))
            ax[1].set_xlim(int(xl / 2 - rate * (xl / 2)), int(xl / 2 + rate * (xl / 2)))
            ax[1].set_ylim(int(yl / 2 - rate * (yl / 2)), int(yl / 2 + rate * (yl / 2)))

    def plot_sources_segmentation(self, percentile=99., xlim=None, ylim=None, zoomin=None):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
        ax = ax.ravel()
        norm = simple_norm(self._data_subbkg, percent=percentile, stretch='asinh')
        ax[0].imshow(self._data_subbkg, cmap='gray', norm=norm, origin='lower')
        ax[1].imshow(self._sources_segmentation, cmap=self._sources_segmentation.cmap, origin='lower')
        if xlim and ylim:
            ax[0].set_xlim(xlim[0], xlim[1])
            ax[0].set_ylim(ylim[0], ylim[1])
            ax[1].set_xlim(xlim[0], xlim[1])
            ax[1].set_ylim(ylim[0], ylim[1])
        elif zoomin:
            rate = 1 / (zoomin / 100)
            xl = np.shape(self._data_subbkg)[0]
            yl = np.shape(self._data_subbkg)[1]
            ax[0].set_xlim(int(xl / 2 - rate * (xl / 2)), int(xl / 2 + rate * (xl / 2)))
            ax[0].set_ylim(int(yl / 2 - rate * (yl / 2)), int(yl / 2 + rate * (yl / 2)))
            ax[1].set_xlim(int(xl / 2 - rate * (xl / 2)), int(xl / 2 + rate * (xl / 2)))
            ax[1].set_ylim(int(yl / 2 - rate * (yl / 2)), int(yl / 2 + rate * (yl / 2)))

    def remove_sources(self, interaction=False):
        sample = self._image_cleaned_background.copy()
        segment_map = self._sources_segmentation
        cat_world = self._sources_overlap
        w = WCS(self._header)
        for i in range(len(cat_world)):
            sky = SkyCoord(cat_world['Ra'][i], cat_world['Dec'][i], frame='icrs', unit='deg')
            x, y = w.world_to_pixel(sky)
            xc, yc = int(x), int(y)
            label = segment_map.data[yc, xc]
            radius = int(cat_world[i]['radius'])
            mea = cat_world[i]['mean']
            std = cat_world[i]['std']

            for a in range(-2 * radius, 2 * radius + 1):
                for b in range(-2 * radius, 2 * radius + 1):
                    if segment_map.data[yc + b, xc + a] == label:
                        sample[yc + b, xc + a] = random.gauss(mea, std)
        self._image_cleaned = sample

    def plot_remove_sources(self, percentile1=99., percentile2=99.95, xlim=None, ylim=None, zoomin=None):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
        ax = ax.ravel()
        norm1 = simple_norm(self._data_subbkg, percent=percentile1, stretch='asinh')
        ax[0].imshow(self._data_subbkg, cmap='gray', norm=norm1, origin='lower')
        norm2 = simple_norm(self._image_cleaned, percent=percentile2, stretch='asinh')
        ax[1].imshow(self._image_cleaned, cmap='gray', norm=norm2, origin='lower')
        if xlim and ylim:
            ax[0].set_xlim(xlim[0], xlim[1])
            ax[0].set_ylim(ylim[0], ylim[1])
            ax[1].set_xlim(xlim[0], xlim[1])
            ax[1].set_ylim(ylim[0], ylim[1])
        elif zoomin:
            rate = 1 / (zoomin / 100)
            xl = np.shape(self._data_subbkg)[0]
            yl = np.shape(self._data_subbkg)[1]
            ax[0].set_xlim(int(xl / 2 - rate * (xl / 2)), int(xl / 2 + rate * (xl / 2)))
            ax[0].set_ylim(int(yl / 2 - rate * (yl / 2)), int(yl / 2 + rate * (yl / 2)))
            ax[1].set_xlim(int(xl / 2 - rate * (xl / 2)), int(xl / 2 + rate * (xl / 2)))
            ax[1].set_ylim(int(yl / 2 - rate * (yl / 2)), int(yl / 2 + rate * (yl / 2)))

    def save_image(self, filename=None, overwrite=False):
        '''
        Save the image.

        Parameters
        ----------
        filename (optional) : string
            The filename of the PSF model.
        overwrite : bool (default: False)
            Overwrite the existing file.
        '''
        header = self._header
        img = self._image_cleaned
        hdu_pri = fits.PrimaryHDU(img, header=header)
        hdul_new = fits.HDUList([hdu_pri])

        os.makedirs('cleaned_images', exist_ok=True)
        if filename is None:
            filename = 'cleaned_images/NGC{}_{}_{}_cleaned.fits'.format(str(self._id), self._telescope, self._filter)

        hdul_new.writeto(filename, overwrite=overwrite)

    def save_psf(self, filename=None, overwrite=False):
        '''
        Save the PSF model of this image.

        Parameters
        ----------
        filename (optional) : string
            The filename of the PSF model.
        overwrite : bool (default: False)
            Overwrite the existing file.
        '''
        oversampling = self._psf_oversample
        img = self._psf
        hdu_pri = fits.PrimaryHDU(img, header=None)
        hdul_new = fits.HDUList([hdu_pri])

        os.makedirs('psf_images', exist_ok=True)
        if filename is None:
            filename = 'psf_images/NGC{}_{}_{}_psf_{}.fits'.format(str(self._id), self._telescope, self._filter, oversampling)

        hdul_new.writeto(filename, overwrite=overwrite)


class Atlas(object):
    '''
    An atlas of images. Again, for the moment, we assume that there is only one 
    science target in each image.
    '''

    def __init__(self, target_coordinate, image_list, header_list, name_list):
        '''
        Parameters
        ----------
        image_list : list
            A list of images (2d array).
        name_list: list
            a list of names (strings).
        '''
        self._image_list = image_list
        self._header_list = header_list
        self._name_list = name_list
        self._ra, self._dec = target_coordinate
        self._coord = None
        self._dict = {  # wavelength(um), pixel size(asec), psf fwhm(asec)
            'GALEX': {'FUV': (0.1528, 3.2, 4.3), 'NUV': (0.2271, 3.2, 5, 3)},
            'SDSS': {'u': (0.3551, 0.45, 1.3), 'g': (0.4686, 0.45, 1.3), 'r': (0.6166, 0.45, 1.3),
                     'i': (0.7480, 0.45, 1.3), 'z': (0.8932, 0.45, 1.3)},
            '2MASS': {'J': (1.25, 1., 2.), 'H': (1.65, 1., 2.), 'Ks': (2.16, 1., 2.)},
            'WISE': {'3.4': (3.4, 1.375, 6.1), '4.6': (4.6, 1.375, 6.4), '12': (12., 1.375, 6.5),
                     '22': (22., 1.375, 12)}
        }
        self._id = None
        self._telescope_list = []
        self._filter_list = []
        self._wavelength_list = []
        self._pxs_list = []
        self._fwhm_list = []
        for i in range(len(name_list)):
            name = name_list[i].strip()
            name_temp = name.split('_')
            if not self._id:
                self._id = name_temp[0][3:]
            telescope_temp = name_temp[1]
            filter_temp = name_temp[2]
            self._telescope_list.append(telescope_temp)
            self._filter_list.append(filter_temp)
            self._wavelength_list.append(self._dict[telescope_temp][filter_temp][0])
            self._pxs_list.append(self._dict[telescope_temp][filter_temp][1])
            self._fwhm_list.append(self._dict[telescope_temp][filter_temp][2])

            self._header = None
            self._fwhm = None
            self._image_list_mathced = None
            self._circular_measurement = None

    def match_images(self, match_header=None, make_fits=True):
        '''
        Get a new list of images with matched resolution, pixel scale, and size.
        Parameter
        -----------
        match_header: optional
            If 'None', it will use the 2MASS header to match.
        Return
        ---------
        a list
        Containing a series of 2d array.
        '''
        # match resolution
        new_img_list = self._image_list
        header_list = self._header_list
        fwhm_list = self._fwhm_list
        pixel_size = self._pxs_list
        wavelength = self._wavelength_list
        sigma_list = [i / (2 * sqrt(2 * log(2))) for i in fwhm_list]
        sigma_kernel = [sqrt(max(sigma_list) ** 2 - sigma_list[i] ** 2) for i in range(len(sigma_list))]
        sigma_kernel_pixel = [sigma_kernel[i] / pixel_size[i] for i in range(len(sigma_kernel))]
        conv_img_list = []
        #  fwhm_pixel = [fwhm_list[i] / pixel_size[i] for i in range(len(fwhm_list))]
        size = []
        for i in range(len(sigma_kernel_pixel)):
            if ceil(sigma_kernel_pixel[i]) % 2 == 0:
                size.append(ceil(sigma_kernel_pixel[i]) + 1)
            else:
                size.append(ceil(sigma_kernel_pixel[i]))
        for i in range(len(new_img_list)):
            if sigma_kernel[i] == 0.:
                conv_img_list.append(new_img_list[i])
                continue
            kernel = make_2dgaussian_kernel((2 * sqrt(2 * log(2))) * sigma_kernel_pixel[i], size=3 * size[i])
            convolved_data = convolve(new_img_list[i], kernel)
            conv_img_list.append(convolved_data)
        # reproject
        if not match_header:
            for i in range(len(header_list)):
                if header_list[i]['TELESCOP'].split('/').strip() == '2MASS':
                    match_header = header_list[i]
                    break
        else:
            match_header = match_header
        rpj_img_list = []
        for i in range(len(conv_img_list)):
            array, _ = reproject_adaptive((conv_img_list[i], WCS(header_list[i])),
                                          WCS(match_header), shape_out=(match_header['NAXIS1'], match_header['NAXIS2']),
                                          kernel='gaussian', conserve_flux=True, boundary_mode='ignore')
            rpj_img_list.append(array)

        self._image_list_mathced = rpj_img_list
        self._header = match_header  # um
        self._fwhm = max(fwhm_list)

        if make_fits:
            os.makedirs('matched_images')
            for i in range(len(new_img_list)):
                tele_name = self._telescope_list[i]
                filt_name = self._filter_list[i]
                hdu_pri = fits.PrimaryHDU(rpj_img_list[i], header=match_header)
                hdul_new = fits.HDUList([hdu_pri])
                hdul_new.writeto('matched_images/NGC{}_{}_{}_matched.fits'.format(str(self._id), tele_name, filt_name),
                                 overwrite=True)

    def get_error(self, r=(10, 20, 30, 40, 50, 60), nexample=50, mask_radius=None):
        '''
        Get the relationship between log10(std) and log10(area).
        Parameter
        -----------
        r: tuple. optional.
            Contain the radius of circular samples. pixels.
        nexample: int.
            The number of sampling.

        self._error_log_line
        -------------
        The function of log10(std) and log10(area). ('asec')
        '''
        img_list_matched = self._image_list_mathced
        header = self._header
        self._coord = read_coordinate(self._ra, self._dec)
        w = WCS(header)
        xc, yc = w.world_to_pixel(self._coord)

        mask = np.zeros(np.shape(img_list_matched[0]))
        ra = int(mask_radius)
        for a in range(-ra, ra + 1):
            for b in range(-ra, ra + 1):
                if sqrt(a ** 2 + b ** 2) <= ra:
                    mask[int(yc) + b, int(xc) + a] = True
        pxs = (abs(self._header['CDELT1']) * 3600.)
        fitted_line_list = []
        for k in range(len(img_list_matched)):
            error_list = []
            for rr in r:
                error_list.append(circular_error_estimate(img_list_matched[k], mask, rr, nexample, percent=99.))
            line_init = models.Linear1D()
            r_list = [(rr * pxs) for rr in r]  # world r.
            area_list = [(np.pi * rr ** 2) for rr in r_list]
            fit = fitting.LinearLSQFitter()
            x_fit = [log10(i) for i in area_list]
            y_fit = [log10(j) for j in error_list]
            fitted_line = fit(line_init, x_fit, y_fit)
            fitted_line_list.append(fitted_line)
        self._error_log_line_list = fitted_line_list

    def circular_measurement(self, radius, a_v):
        '''
        Do the circular measurement.
        radius: float
            The max radius for measurement, which must containing all the flux. 'asec'
        a_v: float.
            The a_v for galactic extinction.
            https://irsa.ipac.caltech.edu/applications/DUST/
        gala_extinction: bool
            If True, it will do the galactic extinction correction.

        self._circular_measurement
        ----------
        A dictionary. Containing functions (input is radius in 'asec').
        Containing the multi-band flux, and error. ('Jy')
        '''
        wavelength = self._wavelength_list
        header = self._header
        self._coord = read_coordinate(self._ra, self._dec)
        w = WCS(header)
        xc, yc = w.world_to_pixel(self._coord)
        pxs = (abs(self._header['CDELT1']) * 3600.)

        ra = radius / pxs

        photometry_dict = {}

        a_list_mag = extinction.fitzpatrick99(np.array([wv * 10000 for wv in wavelength]), a_v)
        a_list_flux = [10 ** (mg / 2.5) for mg in a_list_mag]
        for i in range(len(wavelength)):
            max_radius = int(ra)
            radii = [j for j in range(1, max_radius + 1)]
            cog = CurveOfGrowth(self._image_list_mathced[i], (xc, yc), radii=radii)
            h = cog.profile
            r = cog.radius
            r_c = [rr * pxs for rr in r]

            h_c = [hh * a_list_flux[i] for hh in h]
            flux = interpolate.interp1d(r_c, h_c)

            photometry_dict[wavelength[i]] = {}
            photometry_dict[wavelength[i]]['flux'] = flux

            def mkfun(i):
                def error_temp(xr):
                    return a_list_flux[i] * (10 ** self._error_log_line_list[i](log10(np.pi * (xr ** 2))))
                return error_temp
            photometry_dict[wavelength[i]]['error'] = mkfun(i)
        self._circular_measurement = photometry_dict

    def make_catalog(self, measurement_ra, redshift=0, id=0, plot=False):
        ref = {
            0.1528: 'FUV', 0.2271: 'NUV', 0.3551: 'u_sdss', 0.4686: 'g_sdss', 0.6166: 'r_sdss', 0.7480: 'i_sdss',
            0.8932: 'z_sdss', 1.25: 'J_2mass', 1.65: 'H_2mass', 2.16: 'Ks_2mass', 3.4: 'WISE1', 4.6: 'WISE2',
            12: 'WISE3', 22: 'WISE4'
        }
        data = self._circular_measurement
        k = data.keys()
        k_list = sorted(k)
        os.makedirs('catalog', exist_ok=True)

        for r in measurement_ra:
            f = open('catalog/{}_{}.txt'.format(id, r), 'x')
            f.write('#id redshift ')
            for key in k_list:
                f.write('{} {}_err '.format(ref[key], ref[key]))
            f.write('alpha delta mask ')
            f.write('{} {} '.format(id, redshift))
            for key in k_list:
                f.write('{} {} '.format(data[key]['flux'](r) / 1000, data[key]['error'](r) / 1000))
            f.write('26.68679 -0.67865 0 ')
            f.close()
        if plot:
            for r in measurement_ra:
                plt.plot([log10(i) for i in k_list], [log10(data[i]['flux'](r) / 1000) for i in k_list],
                         marker='x', markersize=3, label='{} arcsec radius'.format(r), linewidth=1)
                plt.xlabel('lg(wavelength) / μm')
                plt.ylabel('lg(flux) / mJy')
                plt.legend()

    def __getitem__(self, items):
        '''
        Get the image object.
        '''
        if self._image_list_mathced is not None:
            return self._image_list_mathced[items]
        else:
            return self._image_list[items]

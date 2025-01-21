import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import corner
import dynesty
from datetime import datetime
from astropy.io import fits
from sedpy.observate import load_filters, getSED

from .sed_model import SEDModel_single
from .sed_model import BadModelError
from .utils_sed import ptform_uniform, ptform_gaussian
from .utils_sed import convert_mJy_to_flam, template_to_SED


class SEDfitter_single(object):
    '''
    This is the object of a SED fitter using Bayesian inference.
    '''
    def __init__(self, data, models, zred=0, DL=None, wave_plot=None, verbose=True):
        '''
        Parameters
        ----------
        data : dict
            A dictionary containing the data to be fitted.
            The keys are the names of the bands, and the values are
            dictionaries containing the following keys:
            'flux' : list of float
                The flux of the band in mJy.
            'flux_err' : list of float
                The error of the flux in mJy
            'bands' : list of string
                The name of the bands that should follow the format of sedpy. 
                `https://github.com/bd-j/sedpy/blob/main/sedpy/data/filters/README.md`
        models : dict
            The dictionary of the SED models. The key is the model name, 
            and the value is the model parameters. The common parameters of all 
            models are: frame, fix_params, priors, and verbose.
        '''
        self._data = data
        self._filters = load_filters(data['bands'])
        self._models = models
        self._verbose = verbose
        
        if wave_plot is None:
            self._wave_plot = np.logspace(-1, 4, 1000)
        else:
            self._wave_plot = wave_plot

        self._sedmodel = SEDModel_single(models=models, zred=zred, DL=DL, verbose=verbose)
        self._zred = zred
        self._DL = self._sedmodel._DL

        self._free_params = self._sedmodel._free_params  # Free parameters
        self._params_label = self._sedmodel._params_label  # Label of the free parameters
        self._params_physical = self._sedmodel._params_physical  # The physical parameters and their units
        self._params_log = self._sedmodel._params_log  # The parameters that are in log space
        self._dim = len(self._free_params)
        self._priors = self._sedmodel._priors

    def check(self):
        '''
        Check the input parameters.
        '''
        # Check the data
        if self._verbose:
            print('[check]: Checking the input data...')
        if not isinstance(self._data, dict):
            raise TypeError('The input data should be a dictionary!')
        for key in ['flux', 'flux_err', 'bands']:
            assert key in self._data.keys(), f'The input data should contain the key "{key}"!'
        assert len(self._data['flux']) == len(self._data['flux_err']), \
            'The length of the flux and flux_err should be the same!'
        assert len(self._data['flux']) == len(self._data['bands']), \
            'The length of the flux and bands should be the same!'
        if self._verbose:
            print('  The input data is valid!')

    def get_results(self, nsample=100, q=[16, 50, 84]):
        '''
        Collect the fitting results.
        '''
        results = {}
        results['samples_phy'] = self.get_samples_phy()
        results['pos_max'] = self.pos_max()
        results['pos_posterior'] = self.pos_posterior(nsample=nsample)

        # Generate the model SEDs
        tdict = {k: [] for k in self._sedmodel._model_names}
        tdict['total_model'] = []
        for pos in results['pos_posterior']:
            mdict = self._sedmodel.gen_templates(params=pos)
            for k, v in mdict.items():
                tdict[k].append(v)
            
            tdict['total_model'].append(self._sedmodel.gen_template_sum(params=pos, wave_interp=self._wave_plot))
        results['templates'] = tdict

        # Calculate the median and uncertainty of the parameters
        samples = self.get_samples()
        p0, p1, p2 = np.percentile(samples, q=q, axis=0)
        results['params'] = dict(
            parameters = self._free_params,
            median = p1,
            lower = p1 - p0,
            upper = p2 - p1,
        )
        p0, p1, p2 = np.percentile(list(results['samples_phy'].values()), q=q, axis=1)
        results['params_phy'] = dict(
            parameters = list(self._params_physical.keys()),
            units = list(self._params_physical.values()),
            median = p1,
            lower = p1 - p0,
            upper = p2 - p1,
        )

        self._results = results

    def get_samples(self):
        '''
        Get the equal weighted samples.
        '''
        return self._dyresults.samples_equal()

    def get_samples_phy(self):
        '''
        Get the samples of the physical parameters.
        '''
        samples = self.get_samples()
        samples_phy = []

        if self._verbose:
            print('[get_samples_phy]: Converting the samples to physical parameters...')
            rangeLoop = tqdm(range(samples.shape[0]))
        else:
            rangeLoop = range(samples.shape[0])
        for loop in rangeLoop:
            pdict = self._sedmodel.get_params_phy(samples[loop, :])
            samples_phy.append(list(pdict.values()))

        samples_phy = dict(zip(pdict.keys(), np.array(samples_phy).T))
        return samples_phy

    def loglike(self, params):
        '''
        Calculate the log likelihood of the model given the data.
        '''
        # Calculate the model SED
        try:
            phot = self._sedmodel.gen_sed(params=params, filters=self._filters)
        except BadModelError:
            return -1e16

        # Calculate the log likelihood
        chisq = -0.5 * np.sum(((phot - self._data['flux']) / self._data['flux_err'])**2)
        return chisq

    def make_cache(self, n):
        '''
        Make the cache for the SED model.
        '''
        if self._verbose:
            print(f'  Generating the cache for process {n}...')
        self._sedmodel.gen_cache()
                
    def make_outputs(self, fileprefix, overwrite=True):
        '''
        Make the output files.
        '''
        # Check the existence of the path
        filepath = '/'.join(fileprefix.split('/')[:-1])
        if filepath == '':
            filepath = '.'

        if not os.path.exists(filepath):
            os.makedirs(filepath)

        self.plot_corner()
        plt.savefig(fileprefix+'_corner.png', dpi=300, bbox_inches='tight')
        plt.close()

        self.plot_corner_phy()
        plt.savefig(fileprefix+'_corner_phy.png', dpi=300, bbox_inches='tight')
        plt.close()

        self.plot_results()
        plt.savefig(fileprefix+'_results.png', dpi=300, bbox_inches='tight')
        plt.close()

        self.save(fileprefix+'.fits', overwrite=overwrite)

    def plot_corner(self, parnames=None, **kwargs):
        '''
        Plot the corner plot of the fitting results.
        '''
        if parnames is None:
            parnames = self._free_params
        
        s = self.get_samples()
        s_show = []
        l_show = []
        for pn in parnames:
            idx = self._sedmodel._free_params.index(pn)
            s_show.append(s[:, idx])
            l_show.append(self._params_label[pn])
        s_show = np.array(s_show).T

        if 'labels' not in kwargs:
            kwargs['labels'] = l_show
        
        if 'show_titles' not in kwargs:
            kwargs['show_titles'] = True

        if 'quantiles' not in kwargs:
            kwargs['quantiles'] = [0.16, 0.5, 0.84]

        return corner.corner(s_show, **kwargs)

    def plot_corner_phy(self, **kwargs):
        '''
        Plot the corner plot of the physical parameters.
        '''
        # Check the existence of the results
        assert hasattr(self, '_results'), 'Please run get_results() first!'
        samples = self._results['samples_phy']

        s_show = []
        l_show = []
        for loop, (k, v) in enumerate(samples.items()):
            s_show.append(v)
            l_show.append(self._params_label.get(k, f'P{loop}'))
        s_show = np.array(s_show).T

        fltr = np.isnan(s_show)
        if fltr.any():
            fltr = ~fltr.any(axis=1)
            s_show = s_show[fltr, :]

        if 'labels' not in kwargs:
            kwargs['labels'] = l_show
        
        if 'show_titles' not in kwargs:
            kwargs['show_titles'] = True

        if 'quantiles' not in kwargs:
            kwargs['quantiles'] = [0.16, 0.5, 0.84]

        return corner.corner(s_show, **kwargs)

    def plot_results(self, fig=None, axs=None):
        '''
        Plot the fitting results.
        '''
        if self._wave_plot is None:
            wave_plot = np.logspace(-1, 4, 1000)
        else:
            wave_plot = self._wave_plot

        # Check the existence of the results
        assert hasattr(self, '_results'), 'Please run get_results() first!'
        tdict = self._results['templates']
        
        if axs is None:
            fig = plt.figure(figsize=(8, 6))
            ax0 = fig.add_axes([0.05, 0.25, 0.9, 0.8])
            ax1 = fig.add_axes([0.05, 0.05, 0.9, 0.2])
            ax1.sharex(ax0)
        # Plot the data
        pwave = np.array([f.wave_effective for f in self._filters]) / 1e4  # micron
        ax0.errorbar(pwave, self._data['flux'], yerr=self._data['flux_err'], 
                     fmt='o', color='k', zorder=4, label='Data')
        xmin = pwave.min() * 0.8
        xmax = pwave.max() * 1.2
        ax0.set_xlim(xmin, xmax)

        # Plot the model SEDs
        count = 0
        for k, v in tdict.items():
            if k == 'total_model':
                flux_total_l, flux_total, flux_total_h = np.percentile(v, [16, 50, 84], axis=0)
                ax0.plot(wave_plot, flux_total, color='red', ls='-', lw=2, label='Model spectrum')
                ax0.fill_between(wave_plot, flux_total_l, flux_total_h, color='red', alpha=0.2)

                # plot the model photometric points
                wave_aa = wave_plot * 1e4
                flam = convert_mJy_to_flam(wave_aa, flux_total)
                phots = getSED(wave_aa, flam, self._filters, linear_flux=True) * 3.631e6  # mJy
                ax0.scatter(pwave, phots, color='k', edgecolor='red', s=40, marker='s', lw=1.5, 
                            zorder=4, label='Model photometry')
            else:
                v = np.array(v)
                wave = v[0, 0, :]
                flux_comp_l, flux_comp, flux_comp_h = np.percentile(v[:, 1, :], [16, 50, 84], axis=0)
                ax0.plot(wave, flux_comp, color=f'C{count}', ls='--', lw=1, label=k)
                ax0.fill_between(wave, flux_comp_l, flux_comp_h, color=f'C{count}', alpha=0.2)
                count += 1

        fltr = (wave_plot > xmin) & (wave_plot < xmax)
        ymin = np.concatenate([self._data['flux']-self._data['flux_err'], flux_total_l[fltr]]).min()
        ymax = np.concatenate([self._data['flux']+self._data['flux_err'], flux_total_h[fltr]]).max()
        ax0.set_ylim(ymin*0.8, ymax*1.2)
        ax0.legend(loc='upper left', fontsize=16, handlelength=1, frameon=False)
        ax0.set_ylabel(r'$F_\nu$ (mJy)', fontsize=24)
        ax0.set_xscale('log')
        ax0.set_yscale('log')

        # Plot the residuals
        wave_aa = wave_plot * 1e4
        flam = convert_mJy_to_flam(wave_aa, flux_total)
        sedm = getSED(wave_aa, flam, self._filters, linear_flux=True) * 3.631e6  # mJy
        fres = (self._data['flux'] - sedm) / self._data['flux_err']
        ax1.errorbar(pwave, fres, fmt='o', color='k')
        ax1.axhline(0, color='k', ls='--')
        ax1.axhspan(-1, 1, ls='--', facecolor='none', edgecolor='k')
        ax1.set_ylim([-3.5, 3.5])
        ax1.set_xlabel(r'Observed wavelength ($\mu$m)', fontsize=20)
        ax1.set_ylabel(r'Res. ($\sigma$)', fontsize=20)
        ax1.minorticks_on()
        return fig, [ax0, ax1]

    def pos_max(self):
        '''
        Find the position of the maximum likelihood.
        '''
        return self._dyresults.samples[-1, :]

    def pos_posterior(self, nsample=1):
        '''
        Generate random positions from the posterior.
        '''
        s = self.get_samples()
        idx = np.random.choice(np.arange(s.shape[0]), size=nsample)
        return s[idx, :]

    def pos_prior(self, nsample=1):
        '''
        Generate random positions from the prior.
        '''
        return self._sedmodel.params_random(nsample=nsample)

    def ptform(self, u):
        '''
        Transforms samples "u" drawn from unit cube to samples to prior.
        '''
        params = []
        for loop, pn in enumerate(self._free_params):
            prior = self._priors[pn]
            if prior['type'] == 'uniform':  # Uniform prior
                params.append(ptform_uniform(u[loop], prior))
            else:  # Gaussian prior
                params.append(ptform_gaussian(u[loop], prior))
        params = np.array(params)
        return params

    def run_dynesty(self, ncpu=1, sampler_name='DynamicNestedSampler', sampler_kws={}, run_nested_kws={}):
        '''
        Run the fit with dynesty. Only use the DynamicNestedSampler.

        Parameters
        ----------
        ncpu : int (default: 1)
            The number of the cpu.
        sampler_name : string (default: 'DynamicNestedSampler')
            The name of the sampler, 'DynamicNestedSampler' or 'NestedSampler'.
        sampler_kws : dict
            The keywords passed to the sampler object.
        run_nested_kws : dict
            The keywords passed to the run_nested function.
        '''
        if 'bound' not in sampler_kws:
            sampler_kws['bound'] = 'multi'
        
        if 'sample' not in sampler_kws:
            sampler_kws['sample'] = 'auto'
            
        if self._verbose is True:
            print('######################## run dynesty ########################')
            print(' Fit with {0}'.format(sampler_name))
            print(' Dynesty version: {}'.format(dynesty.__version__))
            print(' Number of CPUs: {0}'.format(ncpu))
            
            for p, v in sampler_kws.items():
                print(' [sampler] {0}: {1}'.format(p, v))
            
            for p, v in run_nested_kws.items():
                print(' [run_nested] {0}: {1}'.format(p, v))
            print('############################################################')
            
        if 'pool' not in sampler_kws:
            if ncpu > 1:
                from multiprocessing import Pool
                sampler_kws['pool'] = Pool(processes=ncpu)
                sampler_kws['pool'].map(self.make_cache, range(ncpu))

            else:
                sampler_kws['pool'] = None
            sampler_kws['queue_size'] = ncpu

        if sampler_name == 'DynamicNestedSampler':
            sampler = dynesty.DynamicNestedSampler(self.loglike, self.ptform,
                                                   ndim=self._dim, **sampler_kws)
        elif sampler_name == 'NestedSampler':
            sampler = dynesty.NestedSampler(self.loglike, self.ptform,
                                            ndim=self._dim, **sampler_kws)
        else:
            raise ValueError(
                'The sampler name ({0}) is not recognized!'.format(sampler_name))

        sampler.run_nested(**run_nested_kws)
        if self._verbose:
            print('The sampling is finished!')
            
        if sampler_kws.get('pool', None) is not None:
            sampler.pool.close()

        self._sampler = sampler
        self._dyresults = sampler.results
        #self.get_results()

    def save(self, filename, overwrite=True):
        '''
        Save the fitting results.
        '''
        # Check the existence of the path
        filepath = '/'.join(filename.split('/')[:-1])
        if filepath == '':
            filepath = '.'

        if not os.path.exists(filepath):
            os.makedirs(filepath)
        
        # Check the existence of the results
        assert hasattr(self, '_results'), 'Please run get_results() first!'

        # Save the file
        now = datetime.now()

        hdr = fits.Header()
        hdr['zred'] = self._zred
        hdr['DL'] = self._DL
        hdr['verbose'] = self._verbose

        hdr['NMODEL'] = len(self._sedmodel._model_names)
        for loop, mn in enumerate(self._sedmodel._model_names):
            hdr[f'MODEL{loop}'] = mn

        hdr['COMMENT'] = f'Fitted by PIXSED on {now.strftime("%d/%m/%YT%H:%M:%S")}'
        hduList = [fits.PrimaryHDU([1], header=hdr)]

        # Create the data extension
        cols_data = []
        for k, v in self._data.items():
            if k in ['flux', 'flux_err']:
                cols_data.append(fits.Column(name=k, format='E', array=v))
            elif k in ['bands']:
                cols_data.append(fits.Column(name=k, format='20A', array=v))
        hduList.append(fits.BinTableHDU.from_columns(cols_data, name=f'data'))

        # Create the model extension
        cols_wave = [fits.Column(name=f'wave_plot', format='E', 
                                array=self._wave_plot, unit='micron')]
        hduList.append(fits.BinTableHDU.from_columns(cols_wave, name=f'wave_plot'))

        for mn, mp in self._sedmodel._model_dict.items():
            name = f'Model:{mn}'
            
            hdr_m = fits.Header()
            hdr_m['frame'] = mp._frame
            if mn == 'fsps_parametric':
                k = list(mp._fsps_kwargs.keys())
                v = list(mp._fsps_kwargs.values())
                cols_fsps = [fits.Column(name=f'parameters', format='50A', array=k),
                             fits.Column(name=f'values', format='E', array=v)]
                hduList.append(fits.BinTableHDU.from_columns(cols_fsps, name=f'{name}-fsps'))

            elif mn == 'Cat3d_H_wind':
                tempList = mp._template_path.split('/')
                hdr_m['NPATH'] = len(tempList)
                for loop, tp in enumerate(tempList):
                    hdr_m[f'PATH{loop}'] = tp
                hdr_m['wavelim'] = ', '.join([f'{ii}' for ii in mp._wavelim])

            k = list(mp._fix_params.keys())
            v = list(mp._fix_params.values())
            cols_fix = [fits.Column(name=f'parameters', format='50A', array=k),
                        fits.Column(name=f'values', format='E', array=v)]
            hduList.append(fits.BinTableHDU.from_columns(cols_fix, name=f'{name}-fix_params', header=hdr_m))
        
            if len(mp._priors) > 0:
                p0, p1, p2 = [], [], []
                for prior in mp._priors.values():
                    v = list(prior.values())
                    p0.append(v[0])
                    p1.append(v[1])
                    p2.append(v[2])
                cols_prior = [fits.Column(name=f'parameters', format='20A', array=list(mp._priors.keys())),
                              fits.Column(name=f'p0', format='20A', array=p0),
                              fits.Column(name=f'p1', format='E', array=p1),
                              fits.Column(name=f'p2', format='E', array=p2)]
                hduList.append(fits.BinTableHDU.from_columns(cols_prior, name=f'{name}-priors'))
        
        # Create the extension of dynesty results
        hdr_dyresult = fits.Header()
        if 'nlive' in self._dyresults:
            hdr_dyresult['nlive'] = self._dyresults['nlive']
            hdr_dyresult['dynamicnestedsampler'] = False
        else:
            hdr_dyresult['dynamicnestedsampler'] = True
        hdr_dyresult['niter'] = self._dyresults['niter']
        hdr_dyresult['eff'] = self._dyresults['eff']

        cols_dyresults = []
        cols_dyresults.append(fits.Column(name='ncall', format='I', array=self._dyresults['ncall']))
        cols_dyresults.append(fits.Column(name='samples', format=f'{self._dim}E', array=self._dyresults['samples']))
        cols_dyresults.append(fits.Column(name='samples_id', format='E', array=self._dyresults['samples_id']))
        cols_dyresults.append(fits.Column(name='samples_it', format='E', array=self._dyresults['samples_it']))
        cols_dyresults.append(fits.Column(name='samples_u', format=f'{self._dim}E', array=self._dyresults['samples_u']))
        cols_dyresults.append(fits.Column(name='logwt', format='E', array=self._dyresults['logwt']))
        cols_dyresults.append(fits.Column(name='logl', format='E', array=self._dyresults['logl']))
        cols_dyresults.append(fits.Column(name='logvol', format='E', array=self._dyresults['logvol']))
        cols_dyresults.append(fits.Column(name='logz', format='E', array=self._dyresults['logz']))
        cols_dyresults.append(fits.Column(name='logzerr', format='E', array=self._dyresults['logzerr']))
        cols_dyresults.append(fits.Column(name='information', format='E', array=self._dyresults['information']))
        cols_dyresults.append(fits.Column(name='bound_iter', format='E', array=self._dyresults['bound_iter']))
        cols_dyresults.append(fits.Column(name='samples_bound', format='E', array=self._dyresults['samples_bound']))
        cols_dyresults.append(fits.Column(name='scale', format='E', array=self._dyresults['scale']))

        if 'batch_nlive' in self._dyresults:
            cols_dyresults.append(fits.Column(name='samples_batch', format='E', array=self._dyresults['samples_batch']))
            cols_dyresults.append(fits.Column(name='samples_n', format='E', array=self._dyresults['samples_n']))

        hduList.append(fits.BinTableHDU.from_columns(cols_dyresults, name='dyresults', header=hdr_dyresult))
        
        if 'batch_nlive' in self._dyresults:
            cols_batches = []
            cols_batches.append(fits.Column(name='batch_nlive', format='E', array=self._dyresults['batch_nlive']))
            cols_batches.append(fits.Column(name='batch_bounds', format='2E', array=self._dyresults['batch_bounds']))
            hduList.append(fits.BinTableHDU.from_columns(cols_batches, name='dyresults_batches'))

        # Create the results extension
        cols_phy = []
        for k, v in self._results['samples_phy'].items():
            cols_phy.append(fits.Column(name=k, format=f'E', array=v))
        hduList.append(fits.BinTableHDU.from_columns(cols_phy, name='results_phy'))

        pos_p = self._results['pos_posterior'].T
        cols_pos = [fits.Column(name='pos_max', format='E', array=self._results['pos_max']),
                    fits.Column(name='pos_posterior', format=f'{pos_p.shape[1]}E', array=pos_p)]
        hduList.append(fits.BinTableHDU.from_columns(cols_pos, name='results_pos'))

        array = np.array(self._results['templates']['total_model']).T
        cols_total = [fits.Column(name='flux', format=f'{array.shape[1]}E', array=array)]
        hduList.append(fits.BinTableHDU.from_columns(cols_total, name='results_templates:total_model'))

        for k, v in self._results['templates'].items():
            if k == 'total_model':
                continue
            
            v = np.array(v)
            wave = v[:, 0, :].T
            flux = v[:, 1, :].T
            cols_temp = [fits.Column(name='wave', format=f'{wave.shape[1]}E', array=wave),
                         fits.Column(name='flux', format=f'{flux.shape[1]}E', array=flux)]
            hduList.append(fits.BinTableHDU.from_columns(cols_temp, name=f'results_templates:{k}'))

        cols_params = []
        for k, v in self._results['params'].items():
            if k == 'parameters':
                cols_params.append(fits.Column(name=k, format='50A', array=v))
            else:
                cols_params.append(fits.Column(name=k, format=f'E', array=v))
        hduList.append(fits.BinTableHDU.from_columns(cols_params, name='results_params'))

        cols_params_phy = []
        for k, v in self._results['params_phy'].items():
            if k in ['parameters', 'units']:
                cols_params_phy.append(fits.Column(name=k, format='50A', array=v))
            else:
                cols_params_phy.append(fits.Column(name=k, format=f'E', array=v))
        hduList.append(fits.BinTableHDU.from_columns(cols_params_phy, name='results_params_phy'))

        # Save the file
        hduList = fits.HDUList(hduList)
        hduList.writeto(filename, overwrite=overwrite)


def read_SEDfitter_single(filename):
    '''
    Read the SEDfitter_single results and reconstruct the object.
    '''
    hdul = fits.open(filename)
    hdr = hdul[0].header
    zred = hdr['zred']
    DL = hdr['DL']
    verbose = hdr['verbose']

    nmodel = hdr['NMODEL']
    model_names = []
    for loop in range(nmodel):
        model_names.append(hdr[f'MODEL{loop}'])

    # Load the data
    data = {}
    data['flux'] = hdul['data'].data['flux']
    data['flux_err'] = hdul['data'].data['flux_err']
    data['bands'] = hdul['data'].data['bands'].astype(str)

    # Load the model
    wave_plot = hdul['wave_plot'].data['wave_plot']

    models = {}
    for mn in model_names:
        name = f'Model:{mn}'
        hdr_mn = hdul[name+'-fix_params'].header
        frame = hdr_mn['frame']
        fix_params = dict(zip(hdul[name+'-fix_params'].data['parameters'], 
                              hdul[name+'-fix_params'].data['values']))
        
        priors = {}
        for loop, pn in enumerate(hdul[name+'-priors'].data['parameters']):
            ptype = hdul[name+'-priors'].data['p0'][loop]
            if ptype == 'uniform':
                priors[pn] = dict(type=ptype, low=hdul[name+'-priors'].data['p1'][loop], 
                                  high=hdul[name+'-priors'].data['p2'][loop])
            elif ptype == 'gaussian':
                priors[pn] = dict(type=ptype, loc=hdul[name+'-priors'].data['p1'][loop], 
                                  scale=hdul[name+'-priors'].data['p2'][loop])
        
        models[mn] = dict(frame=frame, fix_params=fix_params, priors=priors, verbose=verbose)

        if mn == 'fsps_parametric':
            fsps_kwargs = {}
            for loop, pn in enumerate(hdul[name+'-fsps'].data['parameters']):
                fsps_kwargs[pn] = hdul[name+'-fsps'].data['values'][loop]
            fsps_kwargs['add_dust_emission'] = bool(fsps_kwargs['add_dust_emission'])
            models[mn].update(fsps_kwargs)
        elif mn == 'Cat3d_H_wind':
            tempList = [hdr_mn[f'PATH{loop}'] for loop in range(hdr_mn['NPATH'])]
            models[mn]['template_path'] = '/'.join(tempList)
            wave_lim = hdr_mn['wavelim'].split(', ')
            models[mn]['wavelim'] = [float(wave_lim[0]), float(wave_lim[1])]
    
    sfit = SEDfitter_single(data=data, models=models, zred=zred, DL=DL, wave_plot=wave_plot, verbose=verbose)

    # Load the dynesty results
    res = hdul['dyresults']
    dynamicnestedsampler = res.header['dynamicnestedsampler']
    dyresults = {
        'niter': res.header['niter'],
        'eff': res.header['eff'],
    }
    for cn in res.data.columns.names:
        dyresults[cn] = np.array(res.data[cn])
        
    if dynamicnestedsampler:
        dyresults['samples_batch'] = res.data['samples_batch']
        dyresults['samples_n'] = np.array(res.data['samples_n'])

        res_bat = hdul['dyresults_batches']
        dyresults['batch_nlive'] = np.array(res_bat.data['batch_nlive'])
        dyresults['batch_bounds'] = np.array(res_bat.data['batch_bounds'])
    else:
        dyresults['nlive'] = res.header['nlive']
    
    sfit._dyresults = dynesty.results.Results(dyresults)

    # Load the results
    results = {}
    results['samples_phy'] = {}
    for cn in hdul['results_phy'].data.columns.names:
        results['samples_phy'][cn] = np.array(hdul['results_phy'].data[cn])

    results['pos_max'] = np.array(hdul['results_pos'].data['pos_max'])
    results['pos_posterior'] = np.array(hdul['results_pos'].data['pos_posterior']).T

    results['templates'] = {}
    results['templates']['total_model'] = np.array(hdul['results_templates:total_model'].data['flux']).T
    for mn in model_names:
        if mn == 'total_model':
            continue
        wave = np.array(hdul[f'results_templates:{mn}'].data['wave']).T
        array = np.zeros([wave.shape[0], 2, wave.shape[1]])
        array[:, 0, :] = wave
        array[:, 1, :] = np.array(hdul[f'results_templates:{mn}'].data['flux']).T
        results['templates'][mn] = array

    results['params'] = {}
    for cn in hdul['results_params'].data.columns.names:
        results['params'][cn] = np.array(hdul['results_params'].data[cn])
    
    results['params_phy'] = {}
    for cn in hdul['results_params_phy'].data.columns.names:
        results['params_phy'][cn] = np.array(hdul['results_params_phy'].data[cn])
    
    sfit._results = results

    return sfit


class SEDfitter_combined(object):
    '''
    This object conducts the SED fitting for multiple components of the galaxy.
    '''
    def __init__(self, data_dict, model_list, zred=0, DL=None, wave_plot=None, verbose=True):
        '''
        Parameters
        ----------
        data : dict
            A dictionary containing the data to be fitted.
            The keys are the number of components (0, 1, 2, ...) and the total 
            galaxy ("total"). 
            The values are dictionaries containing the following keys:
                'flux' : list of float
                    The flux of the band in mJy.
                'flux_err' : list of float
                    The error of the flux in mJy
                'bands' : list of string
                    The name of the bands that should follow the format of sedpy. 
                    `https://github.com/bd-j/sedpy/blob/main/sedpy/data/filters/README.md`
        models : list
            A list of dictionaries of SED models. The elements are the model 
            parameters. The common parameters of all models are: frame, 
            fix_params, priors, and verbose.
        '''
        self._data_dict = data_dict

        self._filter_dict = {}
        for k, v in data_dict.items():
            self._filter_dict[k] = load_filters(v['bands'])

        self._model_list = model_list
        self._verbose = verbose
        
        if wave_plot is None:
            self._wave_plot = np.logspace(-1, 4, 1000)
        else:
            self._wave_plot = wave_plot

        self._sedmodel_list = [SEDModel_single(models=m, zred=zred, DL=DL, verbose=verbose) 
                               for m in model_list]
        self._zred = zred
        self._DL = self._sedmodel_list[0]._DL

        self._free_params = []  # Free parameters
        self._free_params_index = []  # The index cut of different model components
        self._params_label = {}  # Label of the free parameters
        self._priors = {}  # Priors
        self._params_physical = {}  # The physical parameters and their units
        self._params_log = []  # The parameters that are in log space
        n0 = 0
        for n, m in enumerate(self._sedmodel_list):
            self._free_params += [f'{n}:{p}' for p in m._free_params]
            self._free_params_index.append([n0, n0+len(m._free_params)])
            n0 += len(m._free_params)

            for pn, v in m._params_label.items():
                self._params_label[f'{n}:{pn}'] = f'C{n}: {v}'

            for pn, prior in m._priors.items():
                self._priors[f'{n}:{pn}'] = prior
            
            for p, v in m._params_physical.items():
                self._params_physical[f'{n}:{p}'] = v 

            self._params_log += [f'{n}:{p}' for p in m._params_log] 

        self._dim = len(self._free_params)

    def check(self):
        '''
        Check the input parameters.
        '''
        # Check the data
        if self._verbose:
            print('[check]: Checking the input data...')

        if not isinstance(self._data_dict, dict):
            raise TypeError('The input data should be a dictionary!')
        
        assert 'total' in self._data_dict.keys(), \
            'The input data should contain the key "total"!'

        for k, v in self._data_dict.items():
            for key in ['flux', 'flux_err', 'bands']:
                assert key in v.keys(), f'The input data should contain the key "{key}"!'

            assert len(v['flux']) == len(v['flux_err']), \
                'The length of the flux and flux_err should be the same!'

            assert len(v['flux']) == len(v['bands']), \
                'The length of the flux and bands should be the same!'
            
            if k != 'total':
                assert int(k) < len(self._model_list), \
                    f'The "{k}"th element do not have a model!'

        if self._verbose:
            print('  The input data is valid!')

    def get_results(self, 
                    nsample:int = 100, 
                    q:list = [16, 50, 84], 
                    total_parnames:list = None):
        '''
        Collect the fitting results.
        '''
        results = {}
        results['samples_phy'] = self.get_samples_phy(total_parnames=total_parnames)
        results['pos_max'] = self.pos_max()
        results['pos_posterior'] = self.pos_posterior(nsample=nsample)

        # Generate the model SEDs
        tdict = {f'{k}': [] for k in range(len(self._sedmodel_list))}
        tdict['total'] = []
        for pos in results['pos_posterior']:
            total_model = np.zeros(len(self._wave_plot))
            for k, m in enumerate(self._sedmodel_list):
                n0, n1 = self._free_params_index[k]
                ts = m.gen_template_sum(params=pos[n0:n1], wave_interp=self._wave_plot)
                tdict[f'{k}'].append(ts)
                total_model += ts
            tdict['total'].append(total_model)
        results['templates'] = tdict

        # Calculate the median and uncertainty of the parameters
        samples = self.get_samples()
        p0, p1, p2 = np.percentile(samples, q=q, axis=0)
        results['params'] = dict(
            parameters = self._free_params,
            median = p1,
            lower = p1 - p0,
            upper = p2 - p1,
        )
        p0, p1, p2 = np.percentile(list(results['samples_phy'].values()), q=q, axis=1)
        results['params_phy'] = dict(
            parameters = list(self._params_physical.keys()),
            units = list(self._params_physical.values()),
            median = p1,
            lower = p1 - p0,
            upper = p2 - p1,
        )

        self._results = results

    def get_samples(self):
        '''
        Get the equal weighted samples.
        '''
        return self._dyresults.samples_equal()

    def get_samples_phy(self, 
                        total_parnames:list = None, 
                        **kwargs):
        '''
        Get the samples of the physical parameters.
        '''
        samples = self.get_samples()
        samples_phy = []

        if self._verbose:
            print('[get_samples_phy]: Converting the samples to physical parameters...')
            rangeLoop = tqdm(range(samples.shape[0]))
        else:
            rangeLoop = range(samples.shape[0])

        for loop in rangeLoop:
            s_phy_list = []
            for n, m in enumerate(self._sedmodel_list):
                n0, n1 = self._free_params_index[n]
                pdict = m.get_params_phy(samples[loop, n0:n1], **kwargs)
                s_phy_list += list(pdict.values())
            samples_phy.append(s_phy_list)

        samples_phy = dict(zip(self._params_physical.keys(), np.array(samples_phy).T))

        # Calculate some parameters summing up the different components
        if total_parnames is not None:
            for pn in total_parnames:
                parList = np.zeros(samples.shape[0])

                for loop, m in enumerate(self._sedmodel_list):
                    k = f'{loop}:{pn}'
                    p = samples_phy.get(k, None)

                    if p is not None:
                        if k in self._params_log:
                            parList = np.log10(10**p + 10**parList)
                        else:
                            parList += p
                        label = m._params_label[pn]
                        unit = m._params_physical[pn]

                k = f'total:{pn}'
                samples_phy[k] = parList
                self._params_label[k] = f'Total: {label}'
                self._params_physical[k] = unit
        
        return samples_phy

    def get_sed_component(self, params, N, filters):
        '''
        Get the SED of a single component.
        '''
        m = self._sedmodel_list[N]
        n0, n1 = self._free_params_index[N]
        phot = self._sedmodel_list[N].gen_sed(params=params[n0:n1], filters=filters)
        return phot
    
    def get_sed_total(self, params, filters):
        '''
        Get the total SED.
        '''
        phot = np.zeros(len(filters))
        for N, m in enumerate(self._sedmodel_list):
            phot += self.get_sed_component(params, N, filters)
        return phot

    def loglike(self, params):
        '''
        Calculate the log likelihood of the model given the data.

        Notes
        -----
        Calculate the models separately to save time.
        '''
        tempDict = {}
        for n, m in enumerate(self._sedmodel_list):
            n0, n1 = self._free_params_index[n]

            try:
                tempDict[n] = m.gen_templates(params[n0:n1])
            except BadModelError:
                return -1e16

        chisq = 0
        for k, v in self._data_dict.items():
            filters = self._filter_dict[k]
            phot = np.zeros(len(filters))

            if k == 'total':
                # Go through all galaxy components
                for td in tempDict.values():
                    # Go through all model templates
                    for wave, flux in td.values():
                        wave_aa = wave * 1e4
                        flam = convert_mJy_to_flam(wave_aa, flux)
                        phot += getSED(wave_aa, flam, filters, linear_flux=True) * 3.631e6 # mJy

                chisq += -0.5 * np.sum(((phot - v['flux']) / v['flux_err'])**2)
            else:
                for wave, flux in tempDict[int(k)].values():
                    wave_aa = wave * 1e4
                    flam = convert_mJy_to_flam(wave_aa, flux)
                    phot += getSED(wave_aa, flam, filters, linear_flux=True) * 3.631e6 # mJy

                chisq += -0.5 * np.sum(((phot - v['flux']) / v['flux_err'])**2)
        
        return chisq

    def make_outputs(self, fileprefix, comp_phot=True, res_ylim=None, overwrite=True):
        '''
        Make the output files.

        Parameters
        ----------
        fileprefix : string
            The prefix of the output files.
        overwrite : bool (default: True)
            Whether to overwrite the existing files.

        Notes
        -----
        The output files include:
            - fileprefix.fits: The fitting results.
            - fileprefix_corner.png: The corner plot of the fitting results.
            - fileprefix_corner_phy.png: The corner plot of the physical parameters.
            - fileprefix_results.png: The fitting results.
        '''
        # Check the existence of the path
        filepath = '/'.join(fileprefix.split('/')[:-1])
        if filepath == '':
            filepath = '.'

        if not os.path.exists(filepath):
            os.makedirs(filepath)

        self.plot_corner()
        plt.savefig(fileprefix+'_corner.png', dpi=300, bbox_inches='tight')
        plt.close()

        self.plot_corner_phy()
        plt.savefig(fileprefix+'_corner_phy.png', dpi=300, bbox_inches='tight')
        plt.close()

        self.plot_results(comp_phot=comp_phot, res_ylim=res_ylim)
        plt.savefig(fileprefix+'_results.png', dpi=300, bbox_inches='tight')
        plt.close()

        self.save(fileprefix+'.fits', overwrite=overwrite)

    def plot_corner(self, parnames=None, **kwargs):
        '''
        Plot the corner plot of the fitting results.
        '''
        if parnames is None:
            parnames = self._free_params
        
        s = self.get_samples()
        s_show = []
        l_show = []
        for pn in parnames:
            idx = self._free_params.index(pn)
            s_show.append(s[:, idx])
            l_show.append(self._params_label[pn])
        s_show = np.array(s_show).T

        if 'labels' not in kwargs:
            kwargs['labels'] = l_show
        
        if 'show_titles' not in kwargs:
            kwargs['show_titles'] = True

        if 'quantiles' not in kwargs:
            kwargs['quantiles'] = [0.16, 0.5, 0.84]

        return corner.corner(s_show, **kwargs)

    def plot_corner_phy(self, **kwargs):
        '''
        Plot the corner plot of the physical parameters.
        '''
        # Check the existence of the results
        assert hasattr(self, '_results'), 'Please run get_results() first!'
        samples = self._results['samples_phy']

        s_show = []
        l_show = []
        for loop, (k, v) in enumerate(samples.items()):
            s_show.append(v)
            l_show.append(self._params_label.get(k, f'P{loop}'))
        s_show = np.array(s_show).T

        fltr = np.isnan(s_show)
        if fltr.any():
            fltr = ~fltr.any(axis=1)
            s_show = s_show[fltr, :]

        if 'labels' not in kwargs:
            kwargs['labels'] = l_show
        
        if 'show_titles' not in kwargs:
            kwargs['show_titles'] = True

        if 'quantiles' not in kwargs:
            kwargs['quantiles'] = [0.16, 0.5, 0.84]

        return corner.corner(s_show, **kwargs)
            
    def plot_results(self, comp_phot=False, fig=None, axs=None, res_ylim=None):
        '''
        Plot the fitting results.
        '''
        if self._wave_plot is None:
            wave_plot = np.logspace(-1, 4, 1000)
        else:
            wave_plot = self._wave_plot

        # Check the existence of the results
        assert hasattr(self, '_results'), 'Please run get_results() first!'
        tdict = self._results['templates']
        
        if axs is None:
            fig = plt.figure(figsize=(8, 6))
            ax0 = fig.add_axes([0.05, 0.25, 0.9, 0.8])
            ax1 = fig.add_axes([0.05, 0.05, 0.9, 0.2])
            ax1.sharex(ax0)
        # Plot the data
        yminList = []
        ymaxList = []
        for k, v in self._data_dict.items():
            pwave = np.array([f.wave_effective for f in self._filter_dict[k]]) / 1e4  # micron

            if k == 'total':
                ax0.errorbar(pwave, v['flux'], yerr=v['flux_err'], fmt='o', 
                             color='k', zorder=4, label='Total')
            else:
                ax0.errorbar(pwave, v['flux'], yerr=v['flux_err'], fmt='o', 
                             color=f'C{k}', zorder=4, label=f'Comp. {k}')

            yminList.append(np.min(v['flux'] - v['flux_err']))
            ymaxList.append(np.max(v['flux'] + v['flux_err']))

        xmin = pwave.min() * 0.8
        xmax = pwave.max() * 1.2
        ax0.set_xlim(xmin, xmax)

        # Plot the model SEDs
        count = 0
        phot_dict = {}
        for k, v in tdict.items():
            if k == 'total':
                flux_total_l, flux_total, flux_total_h = np.percentile(v, [16, 50, 84], axis=0)
                ax0.plot(wave_plot, flux_total, color='red', ls='-', lw=2, label='Model total', alpha=0.5)
                ax0.fill_between(wave_plot, flux_total_l, flux_total_h, color='red', alpha=0.2)

                # plot the model photometric points
                pwave_total, phots_total = template_to_SED(wave_plot, flux_total, self._filter_dict['total'], 
                                                           wave_unit='micron', flux_unit='mJy')
                phot_dict['total'] = [pwave_total, phots_total, 'k']
                ax0.scatter(pwave_total, phots_total, color='k', edgecolor='red', s=40, marker='s', lw=1.5, 
                            zorder=4, label='Model photometry')
            else:
                v = np.array(v)
                flux_comp_l, flux_comp, flux_comp_h = np.percentile(v, [16, 50, 84], axis=0)
                ax0.plot(wave_plot, flux_comp, color=f'C{count}', ls='--', lw=1, label=f'Comp. {k}')
                ax0.fill_between(wave_plot, flux_comp_l, flux_comp_h, color=f'C{count}', alpha=0.2)
                
                # plot the model photometric points
                if (comp_phot) and (f'{k}' in self._data_dict):
                    pwave_comp, phots_comp = template_to_SED(wave_plot, flux_comp, self._filter_dict[k], 
                                                             wave_unit='micron', flux_unit='mJy')
                    phot_dict[k] = [pwave_comp, phots_comp, f'C{count}']
                    ax0.scatter(pwave_comp, phots_comp, color=f'C{count}', edgecolor=f'gray', s=40, marker='s', lw=1.5, 
                                alpha=0.8, zorder=4)

                count += 1
        fltr = (wave_plot > xmin) & (wave_plot < xmax)
        ymin = np.min([np.min(yminList), np.min(flux_total_l[fltr])])
        ymax = np.max([np.max(ymaxList), np.max(flux_total_h[fltr])])
        ax0.set_ylim(ymin*0.8, ymax*1.2)
        ax0.legend(loc='upper left', fontsize=16, handlelength=1, frameon=False)
        ax0.set_ylabel(r'$F_\nu$ (mJy)', fontsize=24)
        ax0.set_xscale('log')
        ax0.set_yscale('log')

        # Plot the residuals
        for k, v in self._data_dict.items():
            pwave, phots, c = phot_dict.get(k, [None, None, None])
            if (pwave is not None) and (phots is not None):
                fres = (v['flux'] - phots) / v['flux_err']
                ax1.errorbar(pwave, fres, fmt='o', color=c)

        ax1.axhline(0, color='k', ls='--')
        ax1.axhspan(-1, 1, ls='--', facecolor='none', edgecolor='k')

        if res_ylim is not None:
            ax1.set_ylim(res_ylim)
        else:
            ax1.set_ylim([-3.5, 3.5])

        ax1.set_xlabel(r'Observed wavelength ($\mu$m)', fontsize=20)
        ax1.set_ylabel(r'Res. ($\sigma$)', fontsize=20)
        ax1.minorticks_on()
        return fig, [ax0, ax1]

    def pos_max(self):
        '''
        Find the position of the maximum likelihood.
        '''
        return self._dyresults.samples[-1, :]

    def pos_posterior(self, nsample=1):
        '''
        Generate random positions from the posterior.
        '''
        s = self.get_samples()
        idx = np.random.choice(np.arange(s.shape[0]), size=nsample)
        return s[idx, :]

    def pos_prior(self, nsample=1):
        '''
        Generate random positions from the prior.
        '''
        pos = [m.params_random(nsample=nsample) for m in self._sedmodel_list]
        return np.concatenate(pos, axis=0)

    def ptform(self, u):
        '''
        Transforms samples "u" drawn from unit cube to samples to prior.
        '''
        params = []
        for loop, pn in enumerate(self._free_params):
            prior = self._priors[pn]
            if prior['type'] == 'uniform':  # Uniform prior
                params.append(ptform_uniform(u[loop], prior))
            else:  # Gaussian prior
                params.append(ptform_gaussian(u[loop], prior))
        params = np.array(params)
        return params

    def run_dynesty(self, ncpu=1, sampler_name='DynamicNestedSampler', sampler_kws={}, run_nested_kws={}):
        '''
        Run the fit with dynesty. Only use the DynamicNestedSampler.

        Parameters
        ----------
        ncpu : int (default: 1)
            The number of the cpu.
        sampler_name : string (default: 'DynamicNestedSampler')
            The name of the sampler, 'DynamicNestedSampler' or 'NestedSampler'.
        sampler_kws : dict
            The keywords passed to the sampler object.
        run_nested_kws : dict
            The keywords passed to the run_nested function.
        '''
        if 'bound' not in sampler_kws:
            sampler_kws['bound'] = 'multi'
        
        if 'sample' not in sampler_kws:
            sampler_kws['sample'] = 'auto'
            
        if self._verbose is True:
            print('######################## run dynesty ########################')
            print(' Fit with {0}'.format(sampler_name))
            print(' Dynesty version: {}'.format(dynesty.__version__))
            print(' Number of CPUs: {0}'.format(ncpu))
            
            for p, v in sampler_kws.items():
                print(' [sampler] {0}: {1}'.format(p, v))
            
            for p, v in run_nested_kws.items():
                print(' [run_nested] {0}: {1}'.format(p, v))
            print('############################################################')
            
        if sampler_name == 'DynamicNestedSampler':
            sampler = dynesty.DynamicNestedSampler(self.loglike, self.ptform,
                                                   ndim=self._dim, **sampler_kws)
        elif sampler_name == 'NestedSampler':
            sampler = dynesty.NestedSampler(self.loglike, self.ptform,
                                            ndim=self._dim, **sampler_kws)
        else:
            raise ValueError(
                'The sampler name ({0}) is not recognized!'.format(sampler_name))

        sampler.run_nested(**run_nested_kws)
        if self._verbose:
            print('The sampling is finished!')
            
        if sampler_kws.get('pool', None) is not None:
            sampler.pool.close()

        self._sampler = sampler
        self._dyresults = sampler.results

    def save(self, filename, overwrite=True):
        '''
        Save the fitting results.
        '''
        # Check the existence of the path
        filepath = '/'.join(filename.split('/')[:-1])
        if filepath == '':
            filepath = '.'

        if not os.path.exists(filepath):
            os.makedirs(filepath)
        
        # Check the existence of the results
        assert hasattr(self, '_results'), 'Please run get_results() first!'

        # Save the file
        now = datetime.now()

        hdr = fits.Header()
        hdr['zred'] = (self._zred, 'Redshift')
        hdr['DL'] = (np.round(self._DL, 3), 'Luminosity distance in Mpc')
        hdr['verbose'] = self._verbose
        hdr['DATCMP'] = (', '.join(list(self._data_dict.keys())), 'Data components')
        hdr['NMODEL'] = (len(self._sedmodel_list), 'Number of models')
        for Nmod, m in enumerate(self._sedmodel_list):
            hdr[f'MODEL{Nmod}'] = (', '.join(m._model_names), 'Model component names')

        hdr['COMMENT'] = f'Fitted by PIXSED on {now.strftime("%d/%m/%YT%H:%M:%S")}'
        hduList = [fits.PrimaryHDU([1], header=hdr)]

        # Create the data extension
        for dn, data in self._data_dict.items():
            cols_data = []
            for k, v in data.items():
                if k in ['flux', 'flux_err']:
                    cols_data.append(fits.Column(name=k, format='E', array=v))
                elif k in ['bands']:
                    cols_data.append(fits.Column(name=k, format='20A', array=v))
            hduList.append(fits.BinTableHDU.from_columns(cols_data, name=f'data:{dn}'))

        # Create the model extension
        cols_wave = [fits.Column(name=f'wave_plot', format='E', 
                                array=self._wave_plot, unit='micron')]
        hduList.append(fits.BinTableHDU.from_columns(cols_wave, name=f'wave_plot'))

        for Nmod, m in enumerate(self._sedmodel_list):
            for mn, mp in m._model_dict.items():
                name = f'Model {Nmod}:{mn}'
                
                hdr_m = fits.Header()
                hdr_m['frame'] = mp._frame
                if mn == 'fsps_parametric':
                    k = list(mp._fsps_kwargs.keys())
                    v = list(mp._fsps_kwargs.values())
                    cols_fsps = [fits.Column(name=f'parameters', format='50A', array=k),
                                 fits.Column(name=f'values', format='E', array=v)]
                    hduList.append(fits.BinTableHDU.from_columns(cols_fsps, name=f'{name}-fsps'))

                elif mn == 'Cat3d_H_wind':
                    tempList = mp._template_path.split('/')
                    hdr_m['NPATH'] = len(tempList)
                    for loop, tp in enumerate(tempList):
                        hdr_m[f'PATH{loop}'] = tp
                    hdr_m['wavelim'] = ', '.join([f'{ii}' for ii in mp._wavelim])

                k = list(mp._fix_params.keys())
                v = list(mp._fix_params.values())
                cols_fix = [fits.Column(name=f'parameters', format='50A', array=k),
                            fits.Column(name=f'values', format='E', array=v)]
                hduList.append(fits.BinTableHDU.from_columns(cols_fix, name=f'{name}-fix_params', header=hdr_m))
        
                if len(mp._priors) > 0:
                    p0, p1, p2 = [], [], []
                    for prior in mp._priors.values():
                        v = list(prior.values())
                        p0.append(v[0])
                        p1.append(v[1])
                        p2.append(v[2])
                    cols_prior = [fits.Column(name=f'parameters', format='20A', array=list(mp._priors.keys())),
                                  fits.Column(name=f'p0', format='20A', array=p0),
                                  fits.Column(name=f'p1', format='E', array=p1),
                                  fits.Column(name=f'p2', format='E', array=p2)]
                    hduList.append(fits.BinTableHDU.from_columns(cols_prior, name=f'{name}-priors'))
        
        # Create the extension of dynesty results
        hdr_dyresult = fits.Header()
        if 'nlive' in self._dyresults:
            hdr_dyresult['nlive'] = self._dyresults['nlive']
            hdr_dyresult['dynamicnestedsampler'] = False
        else:
            hdr_dyresult['dynamicnestedsampler'] = True
        hdr_dyresult['niter'] = self._dyresults['niter']
        hdr_dyresult['eff'] = self._dyresults['eff']

        cols_dyresults = []
        cols_dyresults.append(fits.Column(name='ncall', format='I', array=self._dyresults['ncall']))
        cols_dyresults.append(fits.Column(name='samples', format=f'{self._dim}E', array=self._dyresults['samples']))
        cols_dyresults.append(fits.Column(name='samples_id', format='E', array=self._dyresults['samples_id']))
        cols_dyresults.append(fits.Column(name='samples_it', format='E', array=self._dyresults['samples_it']))
        cols_dyresults.append(fits.Column(name='samples_u', format=f'{self._dim}E', array=self._dyresults['samples_u']))
        cols_dyresults.append(fits.Column(name='logwt', format='E', array=self._dyresults['logwt']))
        cols_dyresults.append(fits.Column(name='logl', format='E', array=self._dyresults['logl']))
        cols_dyresults.append(fits.Column(name='logvol', format='E', array=self._dyresults['logvol']))
        cols_dyresults.append(fits.Column(name='logz', format='E', array=self._dyresults['logz']))
        cols_dyresults.append(fits.Column(name='logzerr', format='E', array=self._dyresults['logzerr']))
        cols_dyresults.append(fits.Column(name='information', format='E', array=self._dyresults['information']))
        cols_dyresults.append(fits.Column(name='bound_iter', format='E', array=self._dyresults['bound_iter']))
        cols_dyresults.append(fits.Column(name='samples_bound', format='E', array=self._dyresults['samples_bound']))
        cols_dyresults.append(fits.Column(name='scale', format='E', array=self._dyresults['scale']))

        if 'batch_nlive' in self._dyresults:
            cols_dyresults.append(fits.Column(name='samples_batch', format='E', array=self._dyresults['samples_batch']))
            cols_dyresults.append(fits.Column(name='samples_n', format='E', array=self._dyresults['samples_n']))

        hduList.append(fits.BinTableHDU.from_columns(cols_dyresults, name='dyresults', header=hdr_dyresult))
        
        if 'batch_nlive' in self._dyresults:
            cols_batches = []
            cols_batches.append(fits.Column(name='batch_nlive', format='E', array=self._dyresults['batch_nlive']))
            cols_batches.append(fits.Column(name='batch_bounds', format='2E', array=self._dyresults['batch_bounds']))
            hduList.append(fits.BinTableHDU.from_columns(cols_batches, name='dyresults_batches'))

        # Create the results extension
        cols_phy = []
        for k, v in self._results['samples_phy'].items():
            cols_phy.append(fits.Column(name=k, format=f'E', array=v))
        hduList.append(fits.BinTableHDU.from_columns(cols_phy, name='results_phy'))

        pos_p = self._results['pos_posterior'].T
        cols_pos = [fits.Column(name='pos_max', format='E', array=self._results['pos_max']),
                    fits.Column(name='pos_posterior', format=f'{pos_p.shape[1]}E', array=pos_p)]
        hduList.append(fits.BinTableHDU.from_columns(cols_pos, name='results_pos'))

        #array = np.array(self._results['templates']['total_model']).T
        #cols_total = [fits.Column(name='flux', format=f'{array.shape[1]}E', array=array)]
        #hduList.append(fits.BinTableHDU.from_columns(cols_total, name='results_templates:total_model'))

        for k, flux in self._results['templates'].items():
            flux = np.array(flux).T
            cols_temp = [fits.Column(name='flux', format=f'{flux.shape[1]}E', array=flux)]
            hduList.append(fits.BinTableHDU.from_columns(cols_temp, name=f'results_templates:{k}'))

        cols_params = []
        for k, v in self._results['params'].items():
            if k == 'parameters':
                cols_params.append(fits.Column(name=k, format='50A', array=v))
            else:
                cols_params.append(fits.Column(name=k, format=f'E', array=v))
        hduList.append(fits.BinTableHDU.from_columns(cols_params, name='results_params'))

        cols_params_phy = []
        for k, v in self._results['params_phy'].items():
            if k in ['parameters', 'units']:
                cols_params_phy.append(fits.Column(name=k, format='50A', array=v))
            else:
                cols_params_phy.append(fits.Column(name=k, format=f'E', array=v))
        hduList.append(fits.BinTableHDU.from_columns(cols_params_phy, name='results_params_phy'))

        # Save the file
        hduList = fits.HDUList(hduList)
        hduList.writeto(filename, overwrite=overwrite)


def read_SEDfitter_combined(filename):
    '''
    Read the SEDfitter_single results and reconstruct the object.
    '''
    hdul = fits.open(filename)
    hdr = hdul[0].header
    zred = hdr['zred']
    DL = hdr['DL']
    verbose = hdr['verbose']
    data_components = hdr['DATCMP'].split(', ')
    nmodel = hdr['NMODEL']
    model_names = {}
    for loop in range(nmodel):
        model_names[f'{loop}'] = hdr[f'MODEL{loop}'].split(', ')

    # Load the data
    data_dict = {}
    for dn in data_components:
        data_dict[dn] = {}
        data_dict[dn]['flux'] = hdul[f'data:{dn}'].data['flux']
        data_dict[dn]['flux_err'] = hdul[f'data:{dn}'].data['flux_err']
        data_dict[dn]['bands'] = hdul[f'data:{dn}'].data['bands'].astype(str)

    # Load the model
    wave_plot = hdul['wave_plot'].data['wave_plot']

    model_list = []
    for Nmod, mns in model_names.items():
        models = {}
        for mn in mns:
            name = f'Model {Nmod}:{mn}'
            hdr_mn = hdul[name+'-fix_params'].header
            frame = hdr_mn['frame']
            fix_params = dict(zip(hdul[name+'-fix_params'].data['parameters'], 
                                  hdul[name+'-fix_params'].data['values']))
            
            priors = {}
            for loop, pn in enumerate(hdul[name+'-priors'].data['parameters']):
                ptype = hdul[name+'-priors'].data['p0'][loop]
                if ptype == 'uniform':
                    priors[pn] = dict(type=ptype, low=hdul[name+'-priors'].data['p1'][loop], 
                                      high=hdul[name+'-priors'].data['p2'][loop])
                elif ptype == 'gaussian':
                    priors[pn] = dict(type=ptype, loc=hdul[name+'-priors'].data['p1'][loop], 
                                      scale=hdul[name+'-priors'].data['p2'][loop])
            
            models[mn] = dict(frame=frame, fix_params=fix_params, priors=priors, verbose=verbose)

            if mn == 'fsps_parametric':
                fsps_kwargs = {}
                for loop, pn in enumerate(hdul[name+'-fsps'].data['parameters']):
                    fsps_kwargs[pn] = hdul[name+'-fsps'].data['values'][loop]
                fsps_kwargs['add_dust_emission'] = bool(fsps_kwargs['add_dust_emission'])
                models[mn].update(fsps_kwargs)
            elif mn == 'Cat3d_H_wind':
                tempList = [hdr_mn[f'PATH{loop}'] for loop in range(hdr_mn['NPATH'])]
                models[mn]['template_path'] = '/'.join(tempList)
                wave_lim = hdr_mn['wavelim'].split(', ')
                models[mn]['wavelim'] = [float(wave_lim[0]), float(wave_lim[1])]

        model_list.append(models)
    
    sfit = SEDfitter_combined(data_dict=data_dict, model_list=model_list, 
                              zred=zred, DL=DL, wave_plot=wave_plot, 
                              verbose=verbose)

    # Load the dynesty results
    res = hdul['dyresults']
    dynamicnestedsampler = res.header['dynamicnestedsampler']
    dyresults = {
        'niter': res.header['niter'],
        'eff': res.header['eff'],
    }
    for cn in res.data.columns.names:
        dyresults[cn] = np.array(res.data[cn])
        
    if dynamicnestedsampler:
        dyresults['samples_batch'] = res.data['samples_batch']
        dyresults['samples_n'] = np.array(res.data['samples_n'])

        res_bat = hdul['dyresults_batches']
        dyresults['batch_nlive'] = np.array(res_bat.data['batch_nlive'])
        dyresults['batch_bounds'] = np.array(res_bat.data['batch_bounds'])
    else:
        dyresults['nlive'] = res.header['nlive']
    
    sfit._dyresults = dynesty.results.Results(dyresults)

    # Load the results
    results = {}
    results['samples_phy'] = {}
    for cn in hdul['results_phy'].data.columns.names:
        results['samples_phy'][cn] = np.array(hdul['results_phy'].data[cn])

    results['pos_max'] = np.array(hdul['results_pos'].data['pos_max'])
    results['pos_posterior'] = np.array(hdul['results_pos'].data['pos_posterior']).T

    results['templates'] = {}
    results['templates']['total'] = np.array(hdul['results_templates:total'].data['flux']).T
    for mn in model_names:
        if mn == 'total_model':
            continue
        results['templates'][mn] = np.array(hdul[f'results_templates:{mn}'].data['flux']).T

    results['params'] = {}
    for cn in hdul['results_params'].data.columns.names:
        results['params'][cn] = np.array(hdul['results_params'].data[cn])
    
    results['params_phy'] = {}
    for cn in hdul['results_params_phy'].data.columns.names:
        results['params_phy'][cn] = np.array(hdul['results_params_phy'].data[cn])
    
    sfit._results = results

    return sfit
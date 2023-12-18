from typing import Any
from tqdm import tqdm
import numpy as np
from astropy import units

from fsps import StellarPopulation
from sedpy.observate import getSED
from scipy.interpolate import interp1d
from .sed_template import load_template
from .utils import package_path
from .utils_sed import convert_mJy_to_flam
from .utils_constants import Lsun, Mpc2cm, cosmo

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc("xtick", direction='out', labelsize=16)
mpl.rc("ytick", direction='out', labelsize=16)
mpl.rc("xtick.major", width=1., size=8)
mpl.rc("ytick.major", width=1., size=8)
mpl.rc("xtick.minor", width=1., size=5)
mpl.rc("ytick.minor", width=1., size=5)


class SEDModel_single(object):
    '''
    The combined model for a single SED.
    '''
    def __init__(self, models, zred=0, DL=None, verbose=True):
        '''
        Initialize the SEDModel object.

        Parameters
        ----------
        models : dict
            The dictionary of the SED models. The key is the model name, 
            and the value is the model parameters.
        '''
        self._models = models
        self._zred = zred
        if DL is None:
            if zred == 0:
                self._DL = 10
            else:
                self._DL = cosmo.luminosity_distance(zred).value
        else:
            self._DL = DL

        self._model_dict = {}
        for mn, mp in models.items():
            # This model do not fit the redshift
            if mp.get('fix_params', None) is None:
                mp['fix_params'] = {}
            mp['fix_params']['zred'] = self._zred
            mp['fix_params']['DL'] = self._DL

            # Turn off the verbose
            mp['verbose'] = False

            if mn == 'fsps_parametric':
                self._model_dict[mn] = fsps_parametric(**mp)
            elif mn == 'Cat3d_H_wind':
                self._model_dict[mn] = Cat3d_H_wind(**mp)
            else:
                raise ValueError(f'The model {mn} is not recognised!')
        self._model_names = list(self._model_dict.keys())
        
        self._parnames = []
        self._fix_params = {}
        self._priors = {}
        self._params_physical = {}
        self._parslicer = {}
        pslicer1, pslicer2 = 0, 0
        for mn, mp in self._model_dict.items():
            self._parnames += [f'{mn}:{pn}' for pn in mp._parnames]
            self._fix_params.update({f'{mn}:{pn}' : v for pn, v in mp._fix_params.items()})
            self._priors.update({f'{mn}:{pn}' : v for pn, v in mp._priors.items()})
            self._params_physical.update({f'{mn}:{pn}' : v for pn, v in mp._params_physical.items()})

            # The parameter slicer
            pslicer1 = pslicer2
            pslicer2 += len(mp._free_params)
            self._parslicer[mn] = slice(pslicer1, pslicer2)
        
        self._free_params = [pn for pn in self._parnames if pn not in self._fix_params]
        self._params_label = self.get_params_label()

        self._verbose = verbose
        if verbose:
            print(self)

    def gen_cache(self):
        '''
        Generate the cache for the models.
        '''
        for mn, mp in self._model_dict.items():
            if mn == 'fsps_parametric':
                mp.gen_cache()
            elif mn == 'Cat3d_H_wind':
                pass
            else:
                raise ValueError(f'The model {mn} is not recognised!')

    def gen_sed(self, params, filters):
        '''
        Generate the model SED with the input values of the free parameters.
        '''
        mdict = self.gen_templates(params)
        phots = []
        for wave, flux in mdict.values():
            wave_aa = wave * 1e4
            flam = convert_mJy_to_flam(wave_aa, flux)
            phots.append(getSED(wave_aa, flam, filters, linear_flux=True))
        phots = np.sum(phots, axis=0) * 3.631e6  # mJy
        return phots

    def gen_templates(self, params) -> dict:
        '''
        Generate the model templates with the input values of the free parameters.

        Parameters
        ----------
        params : dict
            The list of the model parameters.

        Returns
        -------
        mdict : dict
        '''
        mdict = {}
        for mn, mp in self._model_dict.items():
            kwargs = dict(zip(mp._free_params, params[self._parslicer[mn]]))

            try:
                mdict[mn] = mp(**kwargs)
            except Exception as e:
                raise BadModelError('[SEDModel_single]: Bad model -- {}!'.format(e))

        return mdict
    
    def gen_template_sum(self, params, wave_interp) -> np.array:
        '''
        Generate the sum of the model templates with the input values of the free parameters.

        Parameters
        ----------
        params : dict
            The list of the model parameters.
        wave_interp : array_like
            The wavelength array for the interpolation.

        Returns
        -------
        flux_interp : array_like
            The interpolated flux of the model templates.
        '''
        mdict = self.gen_templates(params)
        flux_interp = []
        for mp in mdict.values():
            flux_interp.append(10**np.interp(np.log10(wave_interp), np.log10(mp[0]), np.log10(mp[1])))
        flux_interp = np.sum(flux_interp, axis=0)
        return flux_interp

    def get_params_label(self):
        '''
        Return the labels of the parameters.
        '''
        params_label = {}
        for mn, mp in self._model_dict.items():
            params_label.update({f'{mn}:{pn}' : v for pn, v in mp._params_label.items() 
                                 if ((pn in mp._free_params) or (pn in mp._params_physical))})
        return params_label

    def get_params_phy(self, params):
        '''
        Get the physical parameters from the model.
        '''
        params_phy = {}
        for mn, mp in self._model_dict.items():
            kwargs = dict(zip(mp._free_params, params[self._parslicer[mn]]))

            if mn == 'fsps_parametric':
                pdict = mp.get_params_phy(**kwargs)
            elif mn == 'Cat3d_H_wind':
                pdict = mp.get_params_phy(**kwargs)
            else:
                raise ValueError(f'The model {mn} is not recognised!')
            
            params_phy.update({f'{mn}:{pn}' : v for pn, v in pdict.items()})

        return params_phy

    def params_random(self, nsample=1) -> list:
        '''
        Generate the random values of the free parameters according to 
        the priors.
        '''
        params = []
        for pn in self._free_params:
            if self._priors[pn]['type'] == 'uniform':
                params.append(np.random.uniform(low=self._priors[pn]['low'], high=self._priors[pn]['high'], size=nsample))
            elif self._priors[pn]['type'] == 'normal':
                params.append(np.random.normal(loc=self._priors[pn]['loc'], scale=self._priors[pn]['scale'], size=nsample))
            else:
                raise ValueError(f'The prior type {self._priors[pn]["type"]} is not recognised!')
        return np.array(params)
    
    def plot_sed(self, params, filters=None, fig=None, ax=None, wave_plot=None, kwargs_sed=None):
        '''
        Plot the SED for a given set of parameters.
        '''
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        if wave_plot is None:
            wave_plot = np.logspace(-1, 4, 1000)
        
        mdict = self.gen_templates(params)
        for mn, mp in mdict.items():
            ax.plot(mp[0], mp[1], label=mn)

        if filters is not None:
            phots = self.gen_sed(params, filters)
            pwave = np.array([f.wave_effective for f in filters]) / 1e4  # micron
            
            if kwargs_sed is None:
                kwargs_sed = dict(marker='s', ms=8, mfc='none', mec='k', mew=2, 
                                  ls='none', zorder=8, label='Model SED')
            ax.plot(pwave, phots, **kwargs_sed)

        # Calculate the total model
        flux_total = self.gen_template_sum(params, wave_plot)
        ax.plot(wave_plot, flux_total, color='red', lw=2, label='Total model')

        ax.legend(loc='best', fontsize=16)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'Observed wavelength ($\mu$m)', fontsize=24)
        ax.set_ylabel(r'$F_\nu$ (mJy)', fontsize=24)
        return fig, ax

    def __repr__(self) -> str:
        '''
        Return the string representation of the model.
        '''
        plist = ['The single SED model combining: \n  ' + ', '.join(self._model_names)]
        plist.append('Fixed parameters: ')
        for k, v in self._fix_params.items():
            plist.append(f'  {k} = {v}')       
        plist.append('Free parameters: ')
        plist.append('  {}'.format('\n  '.join(self._free_params)))
        return '\n'.join(plist)


class ComponentModel(object):
    '''
    The example of a component model.  All of the component models should be 
    consistent with this example.
    '''
    def __init__(self, frame='obs', fix_params=None, priors=None, verbose=True):
        self._frame = frame

        # The name of all the model parameters
        self._parnames = ['zred', 'DL']
        # The label of all the parameters for plotting
        self._params_label = []
        # The name of the physical parameters and their units
        self._params_physical = {}

        # Fixed parameters
        if fix_params is None:
            self._fix_params = {}
        else:
            self._fix_params = fix_params

        # The free parameters
        self._free_params = [pn for pn in self._parnames if pn not in self._fix_params]

        # The priors
        if priors is None:
            self._priors_input = {}
        else:
            self._priors_input = priors
        self._priors = self.get_priors(**self._priors_input)
        
        self._verbose = verbose

    def check(self) -> None:
        '''
        Check the consistency of the model.
        '''
        pass

    def get_params_phy(self, **kwargs):
        '''
        Get the physical parameters from the model.
            
        Parameters
        ----------
        kwargs : dict
            All of the free parameters.
        '''
        params_phy = {}
        return params_phy

    def get_priors(self, **priors_input) -> dict:
        '''
        Return the priors of the free parameters.
        '''
        pass

    def __call__(self, **kwargs) -> (np.array, np.array):
        '''
        Calculate the flux for a given set of parameters.
        '''
        pass

    def __repr__(self) -> str:
        pass


class fsps_parametric(ComponentModel):
    '''
    The star and dust emission model based on FSPS. 
    The SFH is parametric (0, 1, 4, 5), and we fix sf_start=0 and sf_trunc=0.
    '''

    def __init__(self, frame='obs', fix_params=None, zcontinuous=1, imf_type=1, 
                 sfh=4, dust_type=2, add_dust_emission=True, priors=None, 
                 verbose=True, **fsps_kwargs):
        '''
        Parameters
        ----------
        frame : str (default: 'obs')
            Frame of reference for the SED model. Valid values are 'rest' and 'obs'.
        fix_params (optional) : dict
            The fixed parameters.
        zcontinuous : int (default: 1)
            Whether to interpolate the metallicity table. 
            0: no interpolation; 1: interpolate.
        imf_type : int (default: 1)
            The IMF type. 
            0: Salpeter (1955); 1: Chabrier (2003); 2: Kroupa (2001); 3: van Dokkum (2008); 4: Dave (2008).
        sfh : int (default: 4)
            The SFH type.
            0: SSP; 1: tau model; 4: delayed tau model; 5: delayed tau model with burst.
        dust_type : int (default: 2)
            The dust attenuation curve type. Default: Calzetti et al. (2000).
        add_dust_emission : bool (default: True)
            Whether to add the dust emission.
        priors (optional) : dict
            The priors of the free parameters. If not provided, the default.
        verbose : bool (default: True)
            Whether to print the model information.
        fsps_kwargs : dict
            Additional parameters of FSPS.

        FSPS parameters
        ---------------
        zred : float
            The redshift.
        DL : float
            The luminosity distance, units: Mpc.
        logMtot : float
            The total mass of stars on formation, units: log(Msun).
        '''
        self._frame = frame
        if frame == 'obs':
            self._z_index = 1.0
        elif frame == 'rest':
            self._z_index = 2.0
        else:
            raise ValueError("The frame '{0}' is not recognised!".format(frame))

        if fix_params is None:
            self._fix_params = dict(sf_start=0.0, sf_trunc=0.0)
        else:
            self._fix_params = fix_params
            if 'sf_start' not in fix_params:
                self._fix_params['sf_start'] = 0.0
            if 'sf_trunc' not in fix_params:
                self._fix_params['sf_trunc'] = 0.0
        
        # Note that the parameters of FSPS are not the same as the parameters of the model
        self._fsps_params = ['zcontinuous', 'imf_type', 'sfh', 'dust_type', 
                             'add_dust_emission', 'tage', 'logzsol', 'dust2', 
                             'duste_gamma', 'duste_umin', 'duste_qpah', 'tau', 
                             'const', 'tburst', 'fburst', 'sf_slope', 
                             'sf_start', 'sf_trunc']

        self._fsps_kwargs = fsps_kwargs.copy()
        self._fsps_kwargs['zcontinuous'] = zcontinuous
        self._fsps_kwargs['imf_type'] = imf_type
        self._fsps_kwargs['sfh'] = sfh
        self._fsps_kwargs['dust_type'] = dust_type
        self._fsps_kwargs['add_dust_emission'] = add_dust_emission
        for k, v in fsps_kwargs.items():
            assert self._fsps_kwargs[k] == v, f'The input FSPS parameter ({k}) is not consistent!'

        self._parnames = ['zred', 'DL', 'logMtot', 'tage', 'logzsol', 'dust2']
        self._params_label = {
            'zred': r'$z_\mathrm{redshift}$',
            'DL': r'$D_\mathrm{L}$ (Mpc)',
            'logMtot': r'$\log\,(M_\mathrm{total}/M_\odot)$',
            'tage': r'$t_\mathrm{age}$ (Gyr)',
            'logzsol': r'$\log\,(Z/Z_\odot)$',
            'dust2': r'$\tau_V$',
            'duste_gamma': r'$\gamma_\mathrm{DL07}$',
            'duste_umin': r'$U_\mathrm{min}$',
            'duste_qpah': r'$q_\mathrm{PAH}$',
            'tau': r'$\tau_\mathrm{SFH}$ (Gyr)',
            'const': r'$f_\mathrm{const}$',
            'fage_burst': r'$f_\mathrm{age, burst}$',
            'tburst': r'$t_\mathrm{burst}$',
            'fburst': r'$f_\mathrm{burst}$',
            'sf_slope': r'$\alpha_\mathrm{SFH}$',
            'sf_start': r'$t_\mathrm{start}$',
            'sf_trunc': r'$t_\mathrm{trunc}$',
            'logmstar': r'$\log\,(M_\star/M_\odot)$',
            'sfr': r'$\mathrm{SFR}$ ($M_\odot\,\mathrm{yr}^{-1}$)',
            'A_V': r'$A_V$',
            'logmdust': r'$\log\,(M_\mathrm{dust}/M_\odot)$',
        }
        self._params_physical = {
            'logmstar': 'solMass',
            'sfr': 'solMass / yr',
            'A_V': 'None',
            'logmdust': 'solMass',
        }

        if self._fsps_kwargs['add_dust_emission']:
            self._parnames += ['duste_gamma', 'duste_umin', 'duste_qpah']

        if self._fsps_kwargs['sfh'] in [1, 4]:
            self._parnames += ['tau', 'const', 'fage_burst', 'fburst', 'sf_start', 'sf_trunc']
        elif self._fsps_kwargs['sfh'] == 5:
            self._parnames += ['tau', 'sf_slope', 'fage_burst', 'fburst', 'sf_start', 'sf_trunc']
        elif self._fsps_kwargs['sfh'] == 0:
            self._parnames += ['tage']
        else:
            raise ValueError(f'The SFH type {self._fsps_kwargs["sfh"]} is not recognised!')

        self._free_params = [pn for pn in self._parnames if pn not in self._fix_params]

        if priors is None:
            self._priors_input = {}
        else:
            self._priors_input = priors
        self._priors = self.get_priors(**self._priors_input)

        self.check()

        self._sps = StellarPopulation(**self._fsps_kwargs)

        self._verbose = verbose
        if verbose:
            print(self)

    def check(self):
        '''
        Check the consistency of the model.
        '''
        assert self._fsps_kwargs['zcontinuous'] in [1], 'The zcontinuous must be 1!'
        assert self._fsps_kwargs['imf_type'] in [0, 1, 2, 3, 4], 'The IMF type must be 0, 1, 2, 3, or 4!'
        assert self._fsps_kwargs['sfh'] in [0, 1, 4, 5], 'The SFH type must be 0, 1, 4, or 5!'
        assert self._fsps_kwargs['dust_type'] in [2], 'Only allow Calzetti et al. (2000) attenuation curve for now!'
        assert (self._fsps_kwargs.get('zred', 0) == 0), 'The redshift must be 0!'

        assert isinstance(self._fix_params, dict), 'The fixed parameters must be a dictionary!'
        for pn in self._fix_params.keys():
            assert pn in self._parnames, f'Parameter {pn} is not in the model parameter list!'

        assert self._priors['logzsol']['type'] == 'uniform', 'The prior of logzsol must be uniform!'

    def complete_params(self, params):
        '''
        Complete the parameters with the fixed parameters.
        
        Parameters
        ----------
        params : dict
            The parameters.

        Returns
        -------
        params_complete : dict
            The complete parameters.
        '''
        params_complete = {}
        for pn in self._parnames:
            if pn in self._fix_params:
                params_complete[pn] = self._fix_params[pn]
            elif pn in params:
                params_complete[pn] = params[pn]
            else:
                raise ValueError(f'Missing parameter {pn}!')
        
        return params_complete

    def gen_cache(self, step=0.01):
        '''
        Run SPS over logzsol in order to get necessary data in cache/memory.
        '''
        self._sps = StellarPopulation(**self._fsps_kwargs)

        low = self._priors['logzsol']['low']
        high = self._priors['logzsol']['high']
        logzsol = np.arange(low, high, step)

        if self._verbose:
            print(f'Generating the cache for different logzsol (low={low}, high={high}, step={step})...')

        for lz in logzsol:
            self._sps.params['logzsol'] = lz
            self._sps._compute_csp()

    def get_labels(self, parname=None):
        '''
        Return the label(s) of the parameters.

        Parameters
        ----------
        parname (optional) : str
            The parameter name. If not provided, return the labels of all 
            the free parameters.
        
        Returns
        -------
        label(s) : dict or str
        '''
        if parname is None:
            return {pn:self._params_label[pn] for pn in self._free_params}
        else:
            return self._params_label.get(parname, None)

    def get_params_phy(self, sfr_dt=0.1, **kwargs):
        '''
        Get the physical parameters from the FSPS model.
        
        Parameters
        ----------
        sfr_dt : float (default: 0.1)
            The time interval for calculating the SFR, units: Gyr.
        kwargs : dict
            All of the free parameters.
        '''
        params = self.complete_params(kwargs)
        self.update_fsps(params)
        mass_total = 10**params['logMtot']

        params_phy = {}
        params_phy['logmstar'] = np.log10(self._sps.stellar_mass * mass_total)
        params_phy['sfr'] = self._sps.sfr_avg(times=params['tage'], dt=sfr_dt) * mass_total
        params_phy['A_V'] = self._sps.params['dust2'] * 2.5 * np.log10(np.e)

        if self._sps.params['add_dust_emission']:
            params_phy['logmdust'] = np.log10(self._sps.dust_mass * mass_total)
        return params_phy

    def get_priors(self, **priors_input):
        '''
        Return the priors of the free parameters.
        '''
        prior_default = dict(type='uniform', low=0.0, high=1e4)
        priors_dict = {
            'logMtot' : dict(type='uniform', low=5.0, high=13.0),
            'tage' : dict(type='uniform', low=0.001, high=13.8),
            'logzsol' : dict(type='uniform', low=-2.0, high=0.19),
            'dust2' : dict(type='uniform', low=0.0, high=2.0),
            'duste_gamma' : dict(type='uniform', low=0.0, high=0.2),
            'duste_umin': dict(type='uniform', low=0.1, high=25.0),
            'duste_qpah': dict(type='uniform', low=0.5, high=7.0),
            'tau' : dict(type='uniform', low=0.0, high=10.0),
            'const' : dict(type='uniform', low=0.0, high=1.0),
            'sf_slope' : dict(type='uniform', low=-5, high=5),
            'fburst' : dict(type='uniform', low=0.0, high=1.0),
            'fage_burst' : dict(type='uniform', low=0.0, high=1.0),
            }
        
        priors = {}
        for pn in self._free_params:
            if pn in priors_input:
                priors[pn] = priors_input[pn]
            elif pn in priors_dict:
                priors[pn] = priors_dict[pn]
            else:
                priors[pn] = prior_default
        return priors

    def params_random(self, nsample=1) -> np.array:
        '''
        Generate the random values of the free parameters according to 
        the priors.
        '''
        params = []
        for pn in self._free_params:
            if self._priors[pn]['type'] == 'uniform':
                params.append(np.random.uniform(low=self._priors[pn]['low'], high=self._priors[pn]['high'], size=nsample))
            elif self._priors[pn]['type'] == 'normal':
                params.append(np.random.normal(loc=self._priors[pn]['loc'], scale=self._priors[pn]['scale'], size=nsample))
            else:
                raise ValueError(f'The prior type {self._priors[pn]["type"]} is not recognised!')
        return np.array(params)

    def update_fsps(self, params):
        '''
        Update the FSPS parameters.

        Parameters
        ----------
        params : dict
            All the free parameters.
        '''
        for pn in params:
            if pn in self._fsps_params:
                self._sps.params[pn] = params[pn]
            elif pn == 'fage_burst':
                self._sps.params['tburst'] = params[pn] * params['tage']

    def __call__(self, **kwargs):
        '''
        Calculate the FSPS model flux for a given set of parameters.

        Parameters
        ----------
        All of the free parameters.

        Returns
        -------
        wave, flux: 1D array
            The wavelength and flux of the model. Units: micron and mJy.
        '''
        params = self.complete_params(kwargs)
        self.update_fsps(params)

        wave, spec = self._sps.get_spectrum(tage=params['tage'], peraa=False)
        wave /= 1e4  # micron
        flux = (1 + params['zred'])**self._z_index * spec * Lsun * 10**params['logMtot'] / (4 * np.pi * (params['DL'] * Mpc2cm)**2) * 1e26  # mJy
            
        if self._frame == 'obs':
            wave = wave * (1 + params['zred'])
        return wave, flux

    def __repr__(self) -> str:
        '''
        Return the string representation of the model.
        '''
        plist = ['The `fsps_parametric` model:']
        for k, v in self._fsps_kwargs.items():
            plist.append(f'  FSPS: {k} = {v}')

        for k, v in self._fix_params.items():
            plist.append(f'Fixed parameters: {k} = {v}')       
        plist.append('Free parameters: {}'.format(','.join(self._free_params)))
        return '\n'.join(plist)


class Cat3d_H_wind(ComponentModel):
    '''
    The CAT3D_H_WIND model.
    '''

    def __init__(self, template_path=None, wavelim=None, frame='obs', fix_params=None, priors=None, verbose=True):
        '''
        Initialize the SEDModel object.

        Parameters
        ----------
        template_path (optional) : str
            Path to the template file. If not provided, the default template 
            file will be used.
        wavelim (optional) : list
            Wavelength limits for the SED model. If not provided, the default 
            wavelength limits [0.1, 1e4] will be used, units: micron.
        frame : str (default: 'obs')
            Frame of reference for the SED model. Valid values are 'rest' and 'obs'.
        '''
        if template_path is None:
            self._template_path = f'{package_path}/templates/Cat3d_H_wind.fits'
        else:
            self._template_path = template_path

        if wavelim is None:
            self._wavelim = [0.1, 1e4]
        else:
            self._wavelim = wavelim

        self._frame = frame
        if frame == 'obs':
            self._z_index = 1.0
        elif frame == 'rest':
            self._z_index = 2.0
        else:
            raise ValueError("The frame '{0}' is not recognised!".format(frame))
        
        if fix_params is None:
            self._fix_params = {}
        else:
            self._fix_params = fix_params

        self._r0 = 1.1  # pc
        self._parnames = ['a', 'h', 'N0', 'inc', 'f_wd', 'a_w', 'Theta_w', 'Theta_sig', 'logL', 'DL', 'zred']
        self._params_label = {'zred': r'$z_\mathrm{redshift}$',
                              'DL': r'$D_\mathrm{L}$ (Mpc)',
                              'a': r'$a_\mathrm{CAT3D}$',
                              'h': r'$h_\mathrm{CAT3D}$',
                              'N0': r'$N_\mathrm{0,CAT3D}$',
                              'inc': r'$i_\mathrm{CAT3D}$',
                              'f_wd': r'$f_\mathrm{wd,CAT3D}$',
                              'a_w': r'$a_\mathrm{wd,CAT3D}$',
                              'Theta_w': r'$\theta_\mathrm{wd,CAT3D}$',
                              'Theta_sig': r'$\theta_\mathrm{\sigma}$',
                              'logL': r'$\log\,(L/\mathrm{erg\,s^{-1}})$',
                            }

        self._template = load_template(self._template_path)
        self._template_params = self._template._parnames

        self._free_params = [pn for pn in self._parnames if pn not in self._fix_params]
        self._params_physical = {}

        if priors is None:
            self._priors_input = {}
        else:
            self._priors_input = priors
        self._priors = self.get_priors(**self._priors_input)

        self.check()

        if verbose:
            print(self)
        
    def check(self):
        '''
        Check the consistency of the model.
        '''
        for pn in self._template_params:
            assert pn in self._parnames, f'Parameter {pn} is not in the model parameter list!'

        assert isinstance(self._fix_params, dict), 'The fixed parameters must be a dictionary!'
        for pn in self._fix_params.keys():
            assert pn in self._parnames, f'Parameter {pn} is not in the model parameter list!'
    
    def get_labels(self, parname=None):
        '''
        Return the label(s) of the parameters.

        Parameters
        ----------
        parname (optional) : str
            The parameter name. If not provided, return the labels of all 
            the free parameters.
        
        Returns
        -------
        label(s) : dict or str
        '''
        if parname is None:
            return {pn:self._params_label[pn] for pn in self._free_params}
        else:
            return self._params_label.get(parname, None)

    def get_priors(self, **priors_input):
        '''
        Return the priors of the free parameters.
        '''
        prior_default = dict(type='uniform', low=0.0, high=1e4)
        priors_dict = {
            'a' : dict(type='uniform', low=-3.5, high=0.0),
            'h' : dict(type='uniform', low=0.0, high=0.6),
            'N0' : dict(type='uniform', low=2.5, high=12.5),
            'inc' : dict(type='uniform', low=0.0, high=90.0),
            'f_wd' : dict(type='uniform', low=0.0, high=1.0),
            'a_w' : dict(type='uniform', low=-3.0, high=0.0),
            'Theta_w' : dict(type='uniform', low=30, high=45),
            'Theta_sig' : dict(type='uniform', low=5, high=17.5),
            'logL' : dict(type='uniform', low=30.0, high=48.0),
            }
        
        priors = {}
        for pn in self._free_params:
            if pn in priors_input:
                priors[pn] = priors_input[pn]
            elif pn in priors_dict:
                priors[pn] = priors_dict[pn]
            else:
                priors[pn] = prior_default
        return priors

    def __call__(self, **kwargs):
        '''
        Calculate the flux for a given set of parameters.

        Parameters
        ----------
        wave : array_like
            The wavelengths of the output flux, units: micron.
        a : float
            The index of the radial dust cloud distribution power law.
        h : float
            Vertical Gaussian distribution dimensionless scale height.
        N0 : float
            The number of clouds along an equatorial line-of-sight.
        inc : float
            Inclination angle in degrees.
        f_wd : float
            The fraction of the total dust mass in the warm dust component.
        a_w : float
            The index of the warm dust grain size distribution power law.
        Theta_w : float
            The temperature of the warm dust component, units: K.
        Theta_sig : float
            The temperature dispersion of the warm dust component, units: K.
        logL : float
            UV luminosity in erg/s in logarithmic scale.
        DL : float
            The luminosity distance, units: Mpc.
        z : float
            The redshift.

        Returns
        -------
        wave, flux : 1D array
            The wavelength and flux of the model. Units: micron and mJy.
        '''
        if 'wave' not in kwargs:
            wave = self._template._wavelength
            fltr = np.ones_like(wave, dtype=bool)
        else:
            wave = kwargs['wave']
            fltr = (wave > self._wavelim[0]) & (wave < self._wavelim[1])
            if np.sum(fltr) == 0:
                return np.zeros_like(wave)

        params = {}
        for pn in self._parnames:
            if pn in self._fix_params:
                params[pn] = self._fix_params[pn]
            elif pn in kwargs:
                params[pn] = float(kwargs[pn])
            else:
                raise ValueError(f'Missing parameter {pn}!')
        
        template_params = []
        for pn in self._template_params:
            template_params.append(params[pn])

        f0 = (1 + params['zred'])**self._z_index * 10**(params['logL'] - 46) * (self._r0 / params['DL'] * 1e-6)**2
        flux = np.zeros_like(wave)
        flux[fltr] = f0 * self._template(wave[fltr], template_params) * 1e29  # unit: mJy
            
        if self._frame == 'obs':
            wave = wave * (1 + params['zred'])
        return wave, flux

    def __repr__(self) -> str:
        '''
        Return the string representation of the model.
        '''
        plist = ['The `Cat3d_H_wind` model:']
        plist.append(f'  Template file: {self._template_path}')
        plist.append(f'  Wavelength limits: {self._wavelim}')
        plist.append(f'  Frame: {self._frame}')
        for k, v in self._fix_params.items():
            plist.append(f'Fixed parameters: {k} = {v}')       
        plist.append('Free parameters: {}'.format(','.join(self._free_params)))
        return '\n'.join(plist)
    

class BadModelError(Exception):
    '''
    The error for a bad model.
    '''
    pass
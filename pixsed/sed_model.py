from typing import Any
import numpy as np
from .sed_template import load_template
from .utils import package_path


class fsps_star_dust(object):
    '''
    The star and dust emission model based on FSPS.
    '''

    def __init__(self) -> None:
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass


class Cat3d_H_wind(object):
    '''
    The CAT3D_H_WIND model.
    '''

    def __init__(self, template_path=None, wavelim=None, frame='rest', fix_params=None):
        '''
        Initialize the SEDModel object.

        Parameters
        ----------
        template_path (optional) : str
            Path to the template file. If not provided, the default template file will be used.
        wavelim (optional) : list
            Wavelength limits for the SED model. If not provided, the default wavelength limits [0.1, 1e4] will be used.
        frame : str (default: 'rest')
            Frame of reference for the SED model. Valid values are 'rest' and 'obs'.
        '''
        if template_path is None:
            template_path = f'{package_path}/templates/Cat3d_H_wind.fits'

        if wavelim is None:
            self._wavelim = [0.1, 1e4]
        else:
            self._wavelim = wavelim

        if frame == 'rest':
            self._z_index = 2.0
        elif frame == 'obs':
            self._z_index = 1.0
        else:
            raise ValueError("The frame '{0}' is not recognised!".format(frame))
        
        if fix_params is None:
            self._fix_params = {}
        else:
            self._fix_params = fix_params

        self._r0 = 1.1  # pc
        self._parnames = ['a', 'h', 'N0', 'inc', 'f_wd', 'a_w', 'Theta_w', 'Theta_sig', 'logL', 'DL', 'z']
        self._template = load_template(template_path)
        self._template_params = self._template._parnames

        # Check the model consistency
        self.check()
        
    def check(self):
        '''
        Check the consistency of the model.
        '''
        for pn in self._template_params:
            assert pn in self._parnames, f'Parameter {pn} is not in the model parameter list!'

        assert isinstance(self._fix_params, dict), 'The fixed parameters must be a dictionary!'
        for pn in self._fix_params.keys():
            assert pn in self._parnames, f'Parameter {pn} is not in the model parameter list!'
    
    def __call__(self, wave, **kwargs):
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
        flux : array_like
            The calculated flux for the given set of parameters, units: mJy.
        '''
        fltr = (wave > self._wavelim[0]) & (wave < self._wavelim[1])
        if np.sum(fltr) == 0:
            return np.zeros_like(wave)

        params = {}
        for pn in self._parnames:
            if pn in self._fix_params:
                params[pn] = self._fix_params[pn]
            elif pn in kwargs:
                params[pn] = kwargs[pn]
            else:
                raise ValueError(f'Missing parameter {pn}!')
        
        template_params = []
        for pn in self._template_params:
            template_params.append(params[pn])

        f0 = (1 + params['z'])**self._z_index * 10**(params['logL'] - 46) * (self._r0 / params['DL'] * 1e-6)**2
        flux = np.zeros_like(wave)
        flux[fltr] = f0 * self._template(wave[fltr], template_params) * 1e29  # unit: mJy
        return flux


import glob
import tqdm
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy import units
from scipy.interpolate import splrep, splev
from sklearn.neighbors import KDTree


class Template(object):
    '''
    This is the object of a model template.
    '''

    def __init__(self, TList, CList, XList, splev_k, wavelength, parnames, modelInfo={}):
        self._TList     = TList
        self._CList     = CList
        self._XList   = XList
        self._splev_k   = splev_k
        self._wavelength = wavelength
        self._kdTree    = KDTree(XList)
        self._modelInfo = modelInfo
        self._parnames = parnames

    def __call__(self, x, pars):
        '''
        Return the interpolation result of the template nearest the input
        parameters.
        '''
        x = np.log10(x)
        ind = np.squeeze(
            self._kdTree.query(np.atleast_2d(pars), return_distance=False))
        tck = (self._TList[ind], self._CList[ind], self._splev_k)
        return 10**splev(x, tck)

    def get_nearestParameters(self, pars):
        """
        Return the nearest template parameters to the input parameters.
        """
        ind = np.squeeze(
            self._kdTree.query(np.atleast_2d(pars), return_distance=False))
        return self._XList[ind]

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, dict):
        self.__dict__ = dict


def load_template(filename):
    '''
    Load the SED template.
    '''
    hdul = fits.open(filename)
    splev_k = hdul[0].header['splrepk']
    parnames = hdul[0].header['PARAMS'].split(', ')
    XList = hdul['XList'].data
    TList = hdul['TList'].data
    CList = hdul['CList'].data
    wavelength = hdul['Wavelength'].data['Wavelength']
    hdul.close()

    template = Template(TList, CList, XList, splev_k, wavelength, parnames)
    return template


def gen_template_cat3d_H_wind(template_path, savename=None, overwrite=False, 
                              splrep_k=3):
    '''
    Generate the template of the CAT3D-WIND model.
    '''
    files = sorted(glob.glob(f'{template_path}/*'))
    
    iList = [0, 15, 30, 45, 60, 75, 90]
    N0List = [5, 7.5, 10]
    awList = [-0.50, -1.00, -1.50, -2.00, -2.50]
    fwdList = [0.15, 0.30, 0.45, 0.60, 0.75]
    thetawList = [30, 45]
    thetasigList = [7.50, 10.00, 15.00]
    aList = [-0.50, -1.00, -1.50, -2.00, -2.50, -3.00]
    hList = [0.10, 0.20, 0.30, 0.40, 0.50]

    XList = []
    TList = []
    CList = []
    for fn in tqdm.tqdm(files):
        f = open(fn, 'r')

        # Load the parameters
        for line in f:
            if line.startswith('#'):
                line = line.strip('#').strip()
                if line.startswith('a '):
                    a = float(line.split('=')[1])
                elif line.startswith('N0 '):
                    N0 = float(line.split('=')[1])
                elif line.startswith('h '):
                    h = float(line.split('=')[1])
                elif line.startswith('a_w '):
                    aw = float(line.split('=')[1])
                elif line.startswith('Theta_w '):
                    thetaw = float(line.split('=')[1])
                elif line.startswith('Theta_sig '):
                    thetasig = float(line.split('=')[1])
                elif line.startswith('f_wd '):
                    fwd = float(line.split('=')[1])
            else:
                break
        f.close()
        
        # Load the template
        tb = Table.read(fn, format='ascii')
        freq = tb['col1']
        wave = tb['col2']
        for loop, inc in enumerate(iList):
            idx = loop + 3
            flux = tb[f'col{idx}'] / freq
            tck = splrep(np.log10(wave), np.log10(flux), k=splrep_k)
            TList.append(tck[0])
            CList.append(tck[1])
            XList.append([a, h, N0, inc, fwd, aw, thetaw, thetasig])
    TList = np.array(TList)
    CList = np.array(CList)
    XList = np.array(XList)

    # Create a FITS file to save the template
    header = fits.header.Header()
    header['splrepk'] = splrep_k
    header.comments['splrepk'] = 'The order of the spline fit'
    header['PARAMS'] = 'a, h, N0, inc, f_wd, a_w, Theta_w, Theta_sig'
    header.comments['PARAMS'] = 'The order of parameters of XList'

    header['XLIST'] = 'The parameters of the template'
    header['TLIST'] = 'The the vector of knots from splrep, in log scale'
    header['CLIST'] = 'The the B-spline coefficients from splrep'
    header['TEMPLATE'] = 'CAT3D-WIND_SED_GRID'
    header['SOURCE'] = 'http://www.sungrazer.org/cat3d.html'
 
    header['PARAM:a'] = ', '.join([str(i) for i in aList])
    header['PARAM:h'] = ', '.join([str(i) for i in hList])
    header['PARAM:N0'] = ', '.join([str(i) for i in N0List])
    header['PARAM:inc'] = ', '.join([str(i) for i in iList])
    header['PARAM:f_wd'] = ', '.join([str(i) for i in fwdList])
    header['PARAM:a_w'] = ', '.join([str(i) for i in awList])
    header['PARAM:Theta_w'] = ', '.join([str(i) for i in thetawList])
    header['PARAM:Theta_sig'] = ', '.join([str(i) for i in thetasigList])
    header['EXPLAIN:a'] = 'Radial power law of the disk'
    header['EXPLAIN:h'] = 'Scale height of the disk'
    header['EXPLAIN:N0'] = 'Average number of dust clouds'
    header['EXPLAIN:inc'] = 'Inclination angle (degree)'
    header['EXPLAIN:f_wd'] = 'Wind-to-disk ratio'
    header['EXPLAIN:a_w'] = 'Radial distribution of dust clouds in the wind'
    header['EXPLAIN:Theta_w'] = 'Half-opening angle of the wind'
    header['EXPLAIN:Theta_sig'] = 'Angular width of the wind'

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header['date'] = now
    hdul = fits.HDUList(fits.PrimaryHDU(header=header))
    hdul.append(fits.ImageHDU(data=np.array(XList), name='XList'))
    hdul.append(fits.ImageHDU(data=np.array(TList), name='TList'))
    hdul.append(fits.ImageHDU(data=np.array(CList), name='CList'))

    wave_data = np.array(wave, dtype=[('Wavelength', 'f')])
    hdul.append(fits.BinTableHDU(data=wave_data, name='Wavelength'))
    hdul[-1].header['UNITS'] = 'micron'

    if savename is not None:
        hdul.writeto(savename, overwrite=overwrite)
    return hdul

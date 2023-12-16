import numpy as np

try:
    from astropy.cosmology import Planck18 as cosmo
except(ImportError):
    cosmo = None

Lsun = 3.846e33  # Solar luminosity: erg/s
Mpc2cm = 3.086e24  # Mpc to cm
ls_micron = 2.99792458e14  # Speed of light in micron/s
ls_AA = 2.99792458e18  # Speed of light in Angstrom/s
ls_km = 2.99792458e5  # Speed of light in km/s
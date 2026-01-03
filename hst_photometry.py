import os, glob, tarfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy as astpy
from astropy.io import fits
from drizzlepac import photeq

from photutils.aperture import aperture_photometry
from photutils.aperture import CircularAperture, CircularAnnulus

import stsynphot as stsyn

from synphot.models import BlackBody1D, PowerLawFlux1D
import synphot as syn
from synphot import SourceSpectrum, Observation
from synphot import Observation, units
import pysynphot as S
import warnings
warnings.filterwarnings("ignore")

mjds,pfl,instflux,vegamag,flamda = [[],[],[],[],[]]

#Vega spectrum
sp = S.Vega

#Image
src = ["NGC 5204 X-1","NGC 5204 X-1",\
       "NGC 1313 X-2","NGC 1313 X-2"]
input_images = ["hst_11227_wfpc2_f450w_drz.fits",\
                "hst_11227_wfpc2_f555w_drz.fits",\
                "f125W_obs1_drz.fits","f125W_obs2_drz.fits"]

wave_photplam = [4557.316,5442.935,12486.06,12486.06] #Pivot wavelength
dwave_photbw = [404.1931,522.2469,866.28125,866.28125] #Band-width
sens_photflam = [9.022466E-18,3.506696E-18,\
                 2.26206135e-20,2.26206135e-20] #Inverse sensitivity

aprad = 5.88 #Aperture radius in pixels: 0.15 arcseconds (~ 3 pix)

#Source location/centroid of interest
xpos = [525.25025,524.52373,2049.7449,2052.3796]
ypos = [490.44085,490.73874,1814.4468,1815.6733]
xposbackground = [506.84207,507.01246,2010.2328,2068.5784]
yposbackground = [496.32388,496.15443,1818.4774,1851.5605]

EE_r5 = 0.91 #Encircled energy for an aperture of radius r=10 pixels

for ctr in range(2,4):
    
    #Vega flux (erg/cm^2/s)
    vega_flux_wfc3 = dwave_photbw[ctr]*sp.sample(wave_photplam[ctr])
    
    #Pivot wavelength
    wavelambda = wave_photplam[ctr]
    
    #Correct for sensitivity variations of WFC3/IR detector
    photeq.photeq(input_images[ctr],readonly=False)
        
    with fits.open(input_images[ctr]) as fd:

        uncalibrated_ct_elec = fd[1].data
        mjd = fd[0].header['EXPSTART']
        date = fd[0].header['DATE-OBS']
        exptime = fd[0].header['EXPTIME']
        photflam = fd[0].header['PHOTFLAM']
    
    uncalibrated_ct_rate = uncalibrated_ct_elec/exptime/EE_r5
    
    if(ctr==0 or ctr==1):
        exptime = 1.0
    
    #Source position
    srcpositions = (xpos[ctr],ypos[ctr])
    aperturesrc = CircularAperture(srcpositions, aprad)
    
    #Background position
    positions_background = (xposbackground[ctr],yposbackground[ctr])
    aperture_background = CircularAperture(positions_background, aprad)
    
    phottablesrc =\
    aperture_photometry(uncalibrated_ct_rate, [aperturesrc])    
    apersbkg = [aperture_background]
    phottablebkg =\
    aperture_photometry(uncalibrated_ct_rate, apersbkg)
    
    ctratesrc = phottablesrc['aperture_sum_0'][0]
    ctratebkg = phottablebkg['aperture_sum_0'][0]
    ctratebacksub = ctratesrc - ctratebkg    
    
    flux = ctratebacksub*sens_photflam[ctr]*dwave_photbw[ctr]
    vega_mag =\
    - 2.5*np.log10(flux/vega_flux_wfc3)
    
    print(ctratebacksub,flux,vega_mag,exptime)
        
    mjds.append(mjd)
    pfl.append(sens_photflam)
    
    instflux.append(ctratebacksub)

    flamda.append(flux)
    vegamag.append(vega_mag)
    

wave_photplam = np.array(wave_photplam)
dwave_photbw = np.array(dwave_photbw)
mjds = np.array(mjds)
pfl = np.array(pfl)
vegamag = np.array(vegamag)
flamda = np.array(flamda)


    
    

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 13:35:22 2024

@author: erc_magnesia_raj
"""
import scipy.signal as sc
from astropy.io import fits
import numpy as np
from astropy.modeling import models,fitting
import glob
from scipy import integrate
import matplotlib.pyplot as plt

#Data
hdu = fits.open("stacked_calibx2.fits")
fobs = hdu[0].data
header = hdu[0].header
deltaL = header['CD1_1']
lamda_i = header['CRVAL1'] 
Nlamda = len(fobs)
lamda_f = lamda_i + Nlamda*deltaL
wave = np.arange(lamda_i, lamda_f, deltaL)

def gauss1d(x,amp,mean,std):
    
    ygauss = amp*np.exp(-0.5*(x-mean)**2/std**2)
    if(amp<0):
        ygauss = np.zeros(len(x))
    return ygauss

wmin = 3500
wmax = 6000
fluxem = fobs[wave>wmin]
waveem = wave[wave>wmin]
fluxem = fluxem[waveem<wmax]
waveem = waveem[waveem<wmax]
dwave = waveem[1]-waveem[0]

# Fit the spectrum and calculate the fitted flux values
cont = models.Polynomial1D(10)
g1 = models.Gaussian1D(amplitude=2e-16, mean=3731.98, stddev=2.7)
g2 = models.Gaussian1D(amplitude=1.8e-17, mean=3873.27, stddev=2.4)
g3 = models.Gaussian1D(amplitude=9.1e-18, mean=3893.43, stddev=2.0)
g4 = models.Gaussian1D(amplitude=8e-18, mean=3973.3, stddev=2.6)
g5 = models.Gaussian1D(amplitude=7e-18, mean=4104.95, stddev=2.9)
g6 = models.Gaussian1D(amplitude=1.5e-17, mean=4344.19, stddev=2.6)
g7 = models.Gaussian1D(amplitude=3e-18, mean=4367.61, stddev=2.15)
g8 = models.Gaussian1D(amplitude=1e-17, mean=4690.01, stddev=3.0)
g9 = models.Gaussian1D(amplitude=3.5e-17, mean=4865.95, stddev=2.5)
g10 = models.Gaussian1D(amplitude=3e-17, mean=4963.64, stddev=2.4)
g11 = models.Gaussian1D(amplitude=8.5e-17, mean=5011.64, stddev=2.38)
g12 = models.Gaussian1D(amplitude=3.4e-18, mean=5882.07, stddev=2.34)
g13 = models.Gaussian1D(amplitude=4.4e-18, mean=5576.51, stddev=4.14)
g14 = models.Gaussian1D(amplitude=1.4e-18, mean=4989.8, stddev=3.0)

g_emission = g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8 + g9 + g10 + g11 + g12 +\
             g13 + g14
g_cont = cont
g_total = g_emission + g_cont
fit_g = fitting.LevMarLSQFitter()

# Fit the model to the data
gmod = fit_g(g_total, waveem, fluxem, maxiter = 1000)
x_g = np.linspace(np.min(waveem), np.max(waveem), len(waveem))
modtot = gmod(x_g)

amp1,m1,sig1 = gmod.parameters[0:3]
amp2,m2,sig2 = gmod.parameters[3:6]
amp3,m3,sig3 = gmod.parameters[6:9]
amp4,m4,sig4 = gmod.parameters[9:12]
amp5,m5,sig5 = gmod.parameters[12:15]
amp6,m6,sig6 = gmod.parameters[15:18]
amp7,m7,sig7 = gmod.parameters[18:21]
amp8,m8,sig8 = gmod.parameters[21:24]
amp9,m9,sig9 = gmod.parameters[24:27]
amp10,m10,sig10 = gmod.parameters[27:30]
amp11,m11,sig11 = gmod.parameters[30:33]
amp12,m12,sig12 = gmod.parameters[33:36]
amp13,m13,sig13 = gmod.parameters[36:39]
amp14,m14,sig14 = gmod.parameters[39:42]

print(amp1,m1,sig1)
print(amp2,m2,sig2)
print(amp3,m3,sig3)
print(amp4,m4,sig4)
print(amp5,m5,sig5)
print(amp6,m6,sig6)
print(amp7,m7,sig7)
print(amp8,m8,sig8)
print(amp9,m9,sig9)
print(amp10,m10,sig10)
print(amp11,m11,sig11)
print(amp12,m12,sig12)
print(amp13,m13,sig13)
print(amp14,m14,sig14)

#Subtract emission lines
for j in range(len(waveem)):
    
    if(waveem[j]>=m1-5*sig1 and waveem[j]<=m1+5*sig1):
        fluxem[j] -= gauss1d(waveem[j],amp1,m1,sig1)
    if(waveem[j]>=m2-5*sig2 and waveem[j]<=m2+5*sig2):
        fluxem[j] -= gauss1d(waveem[j],amp2,m2,sig2)
    if(waveem[j]>=m3-5*sig3 and waveem[j]<=m3+5*sig3):
        fluxem[j] -= gauss1d(waveem[j],amp3,m3,sig3)
    if(waveem[j]>=m4-5*sig4 and waveem[j]<=m4+5*sig4):
        fluxem[j] -= gauss1d(waveem[j],amp4,m4,sig4)
    if(waveem[j]>=m5-5*sig5 and waveem[j]<=m5+5*sig5):
        fluxem[j] -= gauss1d(waveem[j],amp5,m5,sig5)
    if(waveem[j]>=m6-5*sig6 and waveem[j]<=m6+5*sig6):
        fluxem[j] -= gauss1d(waveem[j],amp6,m6,sig6)
    if(waveem[j]>=m7-5*sig7 and waveem[j]<=m7+5*sig7):
        fluxem[j] -= gauss1d(waveem[j],amp7,m7,sig7)
    if(waveem[j]>=m8-5*sig8 and waveem[j]<=m8+5*sig8):
        fluxem[j] -= gauss1d(waveem[j],amp8,m8,sig8)
    if(waveem[j]>=m9-5*sig9 and waveem[j]<=m9+5*sig9):
        fluxem[j] -= gauss1d(waveem[j],amp9,m9,sig9)
    if(waveem[j]>=m10-5*sig10 and waveem[j]<=m10+5*sig10):
        fluxem[j] -= gauss1d(waveem[j],amp10,m10,sig10)
    if(waveem[j]>=m12-5*sig12 and waveem[j]<=m12+5*sig12):
        fluxem[j] -= gauss1d(waveem[j],amp12,m12,sig12)
    if(waveem[j]>=m14-5*sig14 and waveem[j]<=m14+5*sig14):
        fluxem[j] -= gauss1d(waveem[j],amp14,m14,sig14)
    
    #These emission lines are more complex/messy: need better template
    #For now fill these bins with gaussian noise
    if(waveem[j]>=m1-5*sig1 and waveem[j]<=m1+5*sig1):
        mu = np.mean(fluxem[j-100:j])
        std = np.std(fluxem[j-100:j])
        fluxem[j] = np.random.normal(mu,std)
    if(waveem[j]>=m11-5*sig11 and waveem[j]<=m11+5*sig11):
        mu = np.mean(fluxem[j-100:j])
        std = np.std(fluxem[j-100:j])
        fluxem[j] = np.random.normal(mu,std)
    if(waveem[j]>=m13-5*sig13 and waveem[j]<=m13+5*sig13):
        mu = np.mean(fluxem[j-100:j])
        std = np.std(fluxem[j-100:j])
        fluxem[j] = np.random.normal(mu,std)

#Smooth spectrum with savitzky-golay filter
wlen = 40
order = 11
fluxstdsavgol = sc.savgol_filter(fluxem,wlen,order)

hstl = [1528,2375,3356,4327,5307]
hstf1 = [38e-18,12e-18,5.5e-18,3.1e-18,1.7e-18]
dhstf1 = [2e-18,1e-18,0.4e-18,0.2e-18,0.1e-18]
hstf2 = [35e-18,13.3e-18,5.6e-18,2.8e-18,1.57e-18]
dhstf2 = [2e-18,0.6e-18,0.2e-18,0.1e-18,0.04e-18]
hstf1 = np.array(hstf1)
hstf2 = np.array(hstf2)
dhstf1 = np.array(dhstf1)
dhstf2 = np.array(dhstf2)
hstf1/=1e-17
hstf2/=1e-17
dhstf1/=1e-17
dhstf2/=1e-17

plt.figure()
plt.plot(waveem,fluxstdsavgol/1e-17,'r-')
# plt.plot(x_g,modtot/1e-17,'k-')
plt.errorbar(hstl,hstf1,yerr=dhstf1,fmt='ro',\
              label="HST photometry: 05-12-2015")
plt.errorbar(hstl,hstf2,yerr=dhstf2,fmt='bo',\
              label="HST photometry: 24-03-2016")
plt.show()

# Z = np.column_stack((waveem,fluxstdsavgol))
# np.savetxt("masked_smoothed_spec.dat",Z,fmt='%s',delimiter='   ')

# #Sensitivity curve
# hdu = fits.open("sens_ngc1313x2.fits")
# sens = hdu[0].data
# hdr = hdu[0].header
# deltaL = hdr['CD1_1']
# lamda_i = hdr['CRVAL1'] 
# Nlamda = len(sens)
# lamda_f = lamda_i + Nlamda*deltaL
# wvlen = np.arange(lamda_i, lamda_f, deltaL)[1:]

# plt.figure()
# plt.plot(wvlen,sens,'k-')
# plt.tick_params(axis='both', which='major', labelsize=12)
# plt.xlabel("Wavelength [Angstrom]",fontsize=12)
# plt.ylabel("Sensitivity [arb. units]",fontsize=12)
# plt.show()

# wavemin = 3500
# wavemax = 6000
# Mpc = 3.086e22 #Mpc in meters
# distance = 3.95*Mpc*100
# mjd_optical = [55127.0,55176.0,55177.0,55178.0,55179.0,55180.0,\
#                 55185.0,55186.0,55187.0,55189.0]
# fname = ["cbscience_n1_calib.fits","cbscience_n2_calib.fits",\
#           "cbscience_n3_calib.fits","cbscience_n4_calib.fits",\
#           "cbscience_n5_calib.fits","cbscience_n6_calib.fits",\
#           "cbscience_n8_calib.fits","cbscience_n9_calib.fits",\
#           "cbscience_n10_calib.fits","cbscience_n11_calib.fits"]
# lumin = []

# # plt.figure()
# for file2 in range(len(fname)):
    
#     hdu = fits.open(fname[file2])
#     fobs = hdu[0].data
#     header = hdu[0].header
#     deltaL = header['CD1_1']
#     lamda_i = header['CRVAL1'] 
#     Nlamda = len(fobs)
#     lamda_f = lamda_i + Nlamda*deltaL
#     wave = np.arange(lamda_i, lamda_f, deltaL)
    
#     fobs = fobs[wave>wavemin]
#     wave = wave[wave>wavemin]
#     fobs = fobs[wave<wavemax]
#     wave = wave[wave<wavemax]
    
#     #mask bad pixels
#     badpixmin1 = 5567.5
#     baxpixmax1 = 5588.0
#     wsize = 100
    
#     wmin = badpixmin1-wsize*deltaL
#     arr1 = fobs[wave>wmin]
#     xarr1 = wave[wave>wmin]
#     arr1 = arr1[xarr1<badpixmin1]
#     xarr1 = xarr1[xarr1<badpixmin1]
#     wmax = baxpixmax1+wsize*deltaL
#     arr2 = fobs[wave>baxpixmax1]
#     xarr2 = wave[wave>baxpixmax1]
#     arr2 = arr2[xarr2<wmax]
#     xarr2 = xarr2[xarr2<wmax]
#     arr = np.hstack((arr1,arr2))
#     xarr = np.hstack((xarr1,xarr2))    
#     muarr = np.median(arr)
#     stdarr = np.std(arr)
#     xarrinterp = np.arange(badpixmin1,baxpixmax1,deltaL)
#     yinterp = np.random.normal(muarr,stdarr,len(xarrinterp))
        
#     badpixmin2 = 5882.0
#     badpixmax2 = 5920.0
#     wmin2 = badpixmin2-wsize*deltaL
#     arr1b = fobs[wave>wmin2]
#     xarr1b = wave[wave>wmin2]
#     arr1b = arr1b[xarr1b<badpixmin2]
#     xarr1b = xarr1b[xarr1b<badpixmin2]
#     wmax2 = badpixmax2+wsize*deltaL
#     arr2b = fobs[wave>badpixmax2]
#     xarr2b = wave[wave>badpixmax2]
#     arr2b = arr2b[xarr2b<wmax2]
#     xarr2b = xarr2b[xarr2b<wmax2]
#     arrB = np.hstack((arr1b,arr2b))
#     xarrB = np.hstack((xarr1b,xarr2b))    
#     muarrB = np.median(arrB)
#     stdarrB = np.std(arrB)
#     xarrinterpB = np.arange(badpixmin2,badpixmax2,deltaL)
#     yinterpB = np.random.normal(muarrB,stdarrB,len(xarrinterpB))
    
#     index = 0
#     index2 = 0
#     for j in range(len(wave)):
#         if(wave[j]>badpixmin1 and wave[j]<baxpixmax1):
#             fobs[j] = yinterp[index]
#             index += 1
            
#         if(wave[j]>badpixmin2 and wave[j]<badpixmax2):
#             fobs[j] = yinterpB[index2]
#             index2 += 1
    
#     # Fit the spectrum and calculate the fitted flux values
#     cont = models.Polynomial1D(10)
#     g1 = models.Gaussian1D(amplitude=2e-16, mean=3731.98, stddev=2.7)
#     g2 = models.Gaussian1D(amplitude=1e-17, mean=4690.01, stddev=3.0)
#     g3 = models.Gaussian1D(amplitude=3.5e-17, mean=4865.95, stddev=2.5)
#     g4 = models.Gaussian1D(amplitude=3e-17, mean=4963.64, stddev=2.4)
#     g5 = models.Gaussian1D(amplitude=8.5e-17, mean=5011.64, stddev=2.38)
#     g6 = models.Gaussian1D(amplitude=1.5e-17, mean=4344.19, stddev=2.6)
#     g7 = models.Gaussian1D(amplitude=1e-17, mean=4172.76, stddev=3.5)

#     g_emission = g1 + g2 + g3 + g4 + g5 + g6 + g7

#     g_cont = cont
#     g_total = g_emission + g_cont
#     fit_g = fitting.LevMarLSQFitter()

#     # Fit the model to the data
#     g_tot = fit_g(g_total, wave, fobs, maxiter = 1000)
#     x_g = np.linspace(np.min(wave), np.max(wave), len(wave))
#     modtot = g_tot(x_g)
#     # cov_diag = np.sqrt(np.diag(fit_g.fit_info['param_cov']))
                
#     m1,amp1,std1 = g_tot.mean_0.value,g_tot.amplitude_0.value,\
#                     g_tot.stddev_0.value   
#     m2,amp2,std2 = g_tot.mean_1.value,g_tot.amplitude_1.value,\
#                     g_tot.stddev_1.value
#     m3,amp3,std3 = g_tot.mean_2.value,g_tot.amplitude_2.value,\
#                     g_tot.stddev_2.value
#     m4,amp4,std4 = g_tot.mean_3.value,g_tot.amplitude_3.value,\
#                     g_tot.stddev_3.value
#     m5,amp5,std5 = g_tot.mean_4.value,g_tot.amplitude_4.value,\
#                     g_tot.stddev_4.value
#     m6,amp6,std6 = g_tot.mean_5.value,g_tot.amplitude_5.value,\
#                     g_tot.stddev_5.value
#     m7,amp7,std7 = g_tot.mean_6.value,g_tot.amplitude_6.value,\
#                     g_tot.stddev_6.value
    
#     # print(amp1,m1,std1)
#     # print(cov_diag[0],cov_diag[1],cov_diag[2])
#     # print(amp2,m2,std2)
#     # print(cov_diag[3],cov_diag[4],cov_diag[5])
#     # print(amp3,m3,std3)
#     # print(cov_diag[6],cov_diag[7],cov_diag[8])
#     # print(amp4,m4,std4)
#     # print(cov_diag[9],cov_diag[10],cov_diag[11])
#     # print(amp5,m5,std5)
#     # print(cov_diag[12],cov_diag[13],cov_diag[14])
#     # print(amp6,m6,std6)
#     # print(cov_diag[15],cov_diag[16],cov_diag[17])
#     # print(amp7,m7,std7)
#     # print(cov_diag[18],cov_diag[19],cov_diag[20])
    
#     gem1 = gauss1d(wave,amp1,m1,std1)
#     gem2 = gauss1d(wave,amp2,m2,std2)
#     gem3 = gauss1d(wave,amp3,m3,std3)
#     gem4 = gauss1d(wave,amp4,m4,std4)
#     gem5 = gauss1d(wave,amp5,m5,std5)
#     gem6 = gauss1d(wave,amp6,m6,std6)
#     gem7 = gauss1d(wave,amp7,m7,std7)
    
#     gem_tot = gem1 + gem2 + gem3 + gem4 + gem5 + gem6 + gem7
#     fcont_sub = fobs - gem_tot
    
#     zeros = np.zeros(len(wave))
    
#     plt.subplot(211)
#     plt.plot(wave,fobs)
#     plt.plot(wave,zeros,'k--')
#     # plt.plot(wave,modtot,'r-')
#     plt.subplot(212)
#     plt.plot(wave,fcont_sub)
#     plt.plot(wave,zeros,'k--')
    
#     fluxcontinuum = integrate.simps(fcont_sub,wave)
#     lum = (4.0*np.pi*distance**2)*(fluxcontinuum)
#     print(mjd_optical[file2],lum)
#     lumin.append(lum)
    
# plt.show()
        
# Z = np.column_stack((mjd_optical,lumin))
# np.savetxt("continuum_luminosities.dat",Z,fmt='%s',delimiter='   ')



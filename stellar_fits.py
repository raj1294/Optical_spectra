#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 11:19:51 2024

@author: erc_magnesia_raj
"""
# Model comparison
import numpy as np
import extinction
from extinction import ccm89
import matplotlib.pyplot as plt
from kapteyn import kmpfit
import glob
from astropy.io import fits
from scipy import integrate

waveem,fluxstdsavgol = np.loadtxt("masked_smoothed_spec.dat",\
                                  skiprows=0,unpack=True)
    
Mpc = 3.086e22 #Mpc in meters
Rsun = 6.957e8 #Solar radius in meters
Msun = 1.99e30 #Solar mass in kg

ebv = 0.13 #Fix E(B-V) to value known from MUSE spectroscopy
Rv = 3.1 #Adopt Rv=3.1 
Av = ebv*Rv
distance = 3.95 #Fix distance of NGC 1313 X-2 to 3.95 Mpc
#Window of spectrum to fit
wmin = 3500
wmax = 6500
#Grid of surface gravities
sgrav = ["g00","g05","g10","g15","g20","g25",\
          "g30","g35","g40","g45","g50"]

fname,surftemp,surfgrav,radius,rerr,chistatgrid = [[],[],[],[],[],[]]

for kgrav in range(len(sgrav)):
    for file in sorted(glob.glob("Castelli-Kurucz/ckm05*.fits")):
                
        fname.append(file)
        
        temp = float(file.split("_")[1].split(".")[0])
        surftemp.append(temp)
        surfgrav.append(sgrav[kgrav])
                
        hdu = fits.open(file)
        wavemod = hdu[1].data['WAVELENGTH']
        fluxmod = hdu[1].data[sgrav[kgrav]]
        
        fluxmod = fluxmod[wavemod>wmin]
        wavemod = wavemod[wavemod>wmin]
        fluxmod = fluxmod[wavemod<wmax]
        wavemod = wavemod[wavemod<wmax]
        
        fluxstdsavgol = fluxstdsavgol[waveem>wmin]
        waveem = waveem[waveem>wmin]
        fluxstdsavgol = fluxstdsavgol[waveem<wmax]
        waveem = waveem[waveem<wmax]
        
        fluxcontinuum = integrate.simpson(fluxstdsavgol,waveem)
        
        #Interpolate model flux so that it can be compared with the data
        fluxmodinterp = np.interp(waveem,wavemod,fluxmod)
        
        #Construct stellar model
        def stellar_model(Avext,Rvext,rstar,dist,wmod,fmod):
            
            wmod = np.float64(wmod)
            fmod = np.float64(fmod)
            fextinct = extinction.apply(ccm89(wmod, Avext, Rvext,unit='aa'),\
                                        fmod)
            fextinct = (((rstar*Rsun)/(dist*Mpc))**2)*fextinct
            
            return fextinct
        
        def smod(p, x):
            
            rstar = p
            y = stellar_model(Av,Rv,rstar,distance,waveem,fluxmodinterp)
            return y
        
        #Residual statistic
        def residuals(p, data):
            
            x,y = data
            rstar = p
            resid = (y - smod(p,x))
            return resid 
        
        #Initial guess for radius
        rstarinit = 1.0
        paramsinitial = [rstarinit]
        fitobj = kmpfit.Fitter(residuals=residuals,data=(waveem,fluxstdsavgol))
        fitobj.fit(params0=paramsinitial)                                    
        chi2 = fitobj.chi2_min
        dof = fitobj.dof
        
        chistatgrid.append(np.log10(chi2))
            
        radbest = fitobj.params[0]
        radbesterr = fitobj.stderr[0]   
        fbest = smod(radbest,waveem)
        
        residmod = fitobj.residuals(p=fitobj.params[0],\
                                    data=(waveem,fluxstdsavgol))
        radius.append(radbest)
        rerr.append(radbesterr)

chistatgrid,surftemp,surfgrav,\
    radius,rerr,fname = zip(*sorted(zip(chistatgrid,surftemp,surfgrav,\
                                        radius,rerr,fname)))
indmax = 10

# lab = ["Teff: 8500.0 K, log g: 15.0","Teff: 8750.0 K, log g: 15.0",\
#         "Teff: 9000.0 K, log g: 20.0","Teff: 9250.0 K, log g: 20.0",\
#         "Teff: 9000.0 K, log g: 15.0","Teff: 8250.0 K, log g: 15.0",\
#         "Teff: 9500.0 K, log g: 20.0","Teff: 8750.0 K, log g: 20.0",\
#         "Teff: 8000.0 K, log g: 10.0","Teff: 8250.0 K, log g: 10.0"]
lab = ["Teff: 11750.0 K, log g: 20.0","Teff: 11500.0 K, log g: 20.0",\
        "Teff: 11250.0 K, log g: 20.0","Teff: 14000.0 K, log g: 20.0",\
        "Teff: 11000.0 K, log g: 20.0","Teff: 10750.0 K, log g: 20.0",\
        "Teff: 10500.0 K, log g: 20.0","Teff: 10250.0 K, log g: 20.0",\
        "Teff: 10000.0 K, log g: 20.0","Teff: 9750.0 K, log g: 20.0"]

hstl = [1528,2375,3356,4327,5307]
hstf1 = [38e-18,12e-18,5.5e-18,3.1e-18,1.7e-18]
dhstf1 = [2e-18,1e-18,0.4e-18,0.2e-18,0.1e-18]
hstf2 = [35e-18,13.3e-18,5.6e-18,2.8e-18,1.57e-18]
dhstf2 = [2e-18,0.6e-18,0.2e-18,0.1e-18,0.04e-18]

# hstl = [1528,2375,3356]
# hstf1 = [38e-18,12e-18,5.5e-18]
# dhstf1 = [2e-18,1e-18,0.4e-18]
# hstf2 = [35e-18,13.3e-18,5.6e-18]
# dhstf2 = [2e-18,0.6e-18,0.2e-18]

hstf1 = np.array(hstf1)
hstf2 = np.array(hstf2)
dhstf1 = np.array(dhstf1)
dhstf2 = np.array(dhstf2)
hstf1/=1e-17
hstf2/=1e-17
dhstf1/=1e-17
dhstf2/=1e-17

plt.figure()
plt.step(waveem,fluxstdsavgol/1e-17,where='mid',color='k',\
          label="Smoothed GMOS-S spectrum")
plt.errorbar(hstl,hstf1,yerr=dhstf1,fmt='ro',\
              label="HST photometry: 05-12-2015")
plt.errorbar(hstl,hstf2,yerr=dhstf2,fmt='bo',\
              label="HST photometry: 24-03-2016")
plt.tick_params(axis='both', which='major', labelsize=18)
for res in range(indmax):
    
    label = lab[res] + ", Radius: (" +\
            str(round(radius[res],1)) + " ± " +\
            str(round(rerr[res],1)) + r") $R_{\odot}$"
    
    print(surftemp[res],surfgrav[res],\
          radius[res],rerr[res])
        
    hdu = fits.open(fname[res])
    
    waveorig = hdu[1].data['WAVELENGTH']
    fluxorig = hdu[1].data[surfgrav[res]]

    wavemod = hdu[1].data['WAVELENGTH']
    fluxmod = hdu[1].data[surfgrav[res]]
    fluxmod = fluxmod[wavemod>wmin]
    wavemod = wavemod[wavemod>wmin]
    fluxmod = fluxmod[wavemod<wmax]
    wavemod = wavemod[wavemod<wmax]
    
    fluxmodinterp = np.interp(waveem,wavemod,fluxmod)
    fluxmodextinct = stellar_model(Av,Rv,radius[res],\
                                    distance,waveem,fluxmodinterp)
    fluxmodplot = stellar_model(Av,Rv,radius[res],\
                                distance,waveorig,fluxorig)
    
    # plt.plot(waveem,fluxmodextinct/1e-17,label=label)
    plt.plot(waveorig,fluxmodplot/1e-17,label=label)
    
    residmod = (fluxstdsavgol - fluxmodextinct)/1e-17
    
plt.legend(loc="best")
plt.ylabel(r"Flux [$10^{-17}$ erg cm$^{-2}$ s$^{-1}$ A$^{-1}$]",\
            fontsize=18)
plt.xlabel("Wavelength [Angstrom]",fontsize=18)
plt.xlim(1000,6600)
# plt.ylim(-1.6,5.0)
plt.show()

# zeroline = np.zeros(100)
# xzeroline = np.linspace(3100,6500,100)

# plt.subplot(212)
# plt.plot(waveem,residmod/1e-17)
# plt.plot(xzeroline,zeroline,'r--')
# plt.tick_params(axis='both', which='major', labelsize=14)

# plt.xlim(3200,6500)
# plt.xlabel("Wavelength [Angstrom]",fontsize=14)
# plt.ylabel("Data - Model",fontsize=14)
# plt.subplots_adjust(hspace=0)
# plt.show()

# ##Roche lobe radius: limits on orbital separation and companion mass
# import matplotlib.colors as colors

# M1 = 1.4*Msun
# Rstarmin = 55*Rsun
# Rstarmax = 75*Rsun
# nm2 = 100
# naorb = 100
# necc = 100
# M2 = Msun*np.linspace(1.0,30.0,nm2)
# aorb = Rsun*np.linspace(2,200,naorb)
# ecc = np.linspace(0.1,0.95,necc)
# m2,Aorb,Ecc = np.meshgrid(M2,aorb,ecc)
# Rlgridmin = np.zeros((len(m2),len(Aorb),len(ecc)))
# Rlgridmax = np.zeros((len(m2),len(Aorb),len(ecc)))
# frot = 0.1

# ##Filling factor of Roche Lobe
# fmin = 0.9 
# fmax = 1.1

# m2limit,Aorblimit,ecclimit = [[],[],[]]
# m2limit_allowed,Aorblimit_allowed,ecclimit_allowed = [[],[],[]]

# for k1 in range(len(m2)):
#     for k2 in range(len(Aorb)):
#         for k3 in range(len(ecc)):
        
#             #Classic RL radius (Eggleton 1983)
#             qratio = m2[k1][k2][k3]/M1
            
#             Rlgridmin[k1][k2][k3] =\
#                 Aorb[k1][k2][k3]*(1-Ecc[k1][k2][k3])*\
#                 (0.49*qratio**(2./3.))/\
#                 (0.6*qratio**(2./3.) + np.log(1 + qratio**(1./3.)))
        
#             Rlgridmax[k1][k2][k3] =\
#                 Aorb[k1][k2][k3]*(1+Ecc[k1][k2][k3])*\
#                 (0.49*qratio**(2./3.))/\
#                 (0.6*qratio**(2./3.) + np.log(1 + qratio**(1./3.)))
            
#             ##Periastron
#             if(Rstarmax<fmin*Rlgridmin[k1][k2][k3] or\
#                 Rstarmax>fmax*Rlgridmin[k1][k2][k3]):
#                 m2limit.append(m2[k1][k2][k3])
#                 Aorblimit.append(Aorb[k1][k2][k3])
#                 ecclimit.append(Ecc[k1][k2][k3])
#             if(Rstarmin>fmin*Rlgridmax[k1][k2][k3] and\
#                 Rstarmin<fmax*Rlgridmax[k1][k2][k3]):
#                 m2limit_allowed.append(m2[k1][k2][k3])
#                 Aorblimit_allowed.append(Aorb[k1][k2][k3])
#                 ecclimit_allowed.append(Ecc[k1][k2][k3])
            
#             ##Apastron
#             # if(Rstarmin<fmin*Rlgridmax[k1][k2][k3] or\
#             #     Rstarmin>fmax*Rlgridmax[k1][k2][k3]):
#             #     m2limit.append(m2[k1][k2][k3])
#             #     Aorblimit.append(Aorb[k1][k2][k3])
#             #     ecclimit.append(Ecc[k1][k2][k3])
#             # if(Rstarmax>fmin*Rlgridmax[k1][k2][k3] and\
#             #     Rstarmax<fmax*Rlgridmax[k1][k2][k3]):
#             #     m2limit_allowed.append(m2[k1][k2][k3])
#             #     Aorblimit_allowed.append(Aorb[k1][k2][k3])
#             #     ecclimit_allowed.append(Ecc[k1][k2][k3])

#             # #Refined RL radius
#             # #(for non-synchronous rotation and eccentric orbits)
#             # rlegg = (0.49*qratio**(2./3.))/\
#             #         (0.6*qratio**(2./3.) + np.log(1 + qratio**(1./3.)))
#             # rleggmin = Aorb[k1][k2][k3]*(1-Ecc[k1][k2][k3])*rlegg
#             # rleggmax = Aorb[k1][k2][k3]*(1+Ecc[k1][k2][k3])*rlegg
            
#             # Apot = (frot**2)*(1 + Ecc[k1][k2][k3])**4/\
#             #                  (1 - Ecc[k1][k2][k3])**3
#             # # Apot = (frot**2)*(1 + Ecc[k1][k2][k3])**4/\
#             # #                  (1 + Ecc[k1][k2][k3])**3

#             # if (np.log10(qratio)>=0 and np.log10(Apot)<=-0.1):
#             #     rl = rlegg*(1.226 - 0.21*Apot - 0.15*(1-Apot)*\
#             #          np.exp((0.25*Apot - 0.3)*(np.log10(qratio))**1.55))
            
#             # if (np.log10(qratio)<=0 and np.log10(Apot)<=-0.1):
#             #     rl = rlegg*(1 + 0.11*(1-Apot) - 0.05*(1-Apot)*\
#             #          np.exp(-(0.5*(1+Apot) + np.log10(qratio))**2))
    
#             # if (np.log10(qratio)<=0 and np.log10(Apot)>=-0.1 and\
#             #     np.log10(Apot)<=0.2):
                
#             #     g0 = 0.9978 - 0.1229*np.log10(Apot) -\
#             #          0.1273*(np.log10(Apot))**2
#             #     g1 = 0.001 + 0.02556*(np.log10(Apot))
#             #     g2 = 0.0004 + 0.0021*(np.log10(Apot))
                
#             #     rl = rlegg*(g0 + g1*(np.log10(qratio)) +\
#             #                 g2*(np.log10(qratio))**2)
    
#             # if (np.log10(qratio)>=0 and np.log10(Apot)>=-0.1 and\
#             #     np.log10(Apot)<=0.2):
                
#             #     h0 = 1.0071 - 0.0907*np.log10(Apot) -\
#             #          0.0495*(np.log10(Apot))**2
#             #     h1 = -0.004 - 0.163*(np.log10(Apot)) -\
#             #          0.214*(np.log10(Apot))**2
#             #     h2 = 0.00022 - 0.0108*(np.log10(Apot)) -\
#             #         0.02718*(np.log10(Apot))**2
                
#             #     rl = rlegg*(h0 + h1*(np.log10(qratio)) +\
#             #                 h2*(np.log10(qratio))**2)
            
#             # if(np.log10(qratio)<=0 and np.log10(Apot)>=0.2):
                
#             #     i0 = (6.3014*(np.log10(Apot))**(1.3643))/\
#             #          (np.exp(2.3644*(np.log10(Apot))**0.70748) -\
#             #           1.4413*np.exp(-0.0000184*(np.log10(Apot))**(-4.5693)))
#             #     i1 = np.log10(Apot)/\
#             #          (0.0015*np.exp(8.84*(np.log10(Apot))**0.282) + 15.78)
#             #     i2 = (1 + 0.036*np.exp(8.01*(np.log10(Apot))**0.879))/\
#             #          (0.105*np.exp(7.91*(np.log10(Apot))**0.879))
#             #     i3 = 0.991/(1.38*np.exp(-0.035*(np.log10(Apot))**0.76) +\
#             #          23.0*np.exp(-2.89*(np.log10(Apot))**0.76))
                
#             #     rl = rlegg*(i0 + i1*np.exp(-i2*(np.log10(qratio) + i3)**2))
            
#             # if(np.log10(qratio)>=0 and np.log10(Apot)>=0.2):
                
#             #     j0 = (1.895*(np.log10(Apot))**0.837)/\
#             #          (np.exp(1.636*(np.log10(Apot))**0.789) - 1)
#             #     j1 = (4.3*(np.log10(Apot))**0.98)/\
#             #          (np.exp(2.5*(np.log10(Apot))**0.66) + 4.7)
#             #     j2 = (1.)/(8.8*np.exp(-2.95*(np.log10(Apot))**0.76) +\
#             #          1.64*np.exp(-0.03*(np.log10(Apot))**0.76))
#             #     j3 = 0.256*(np.exp(-1.33*(np.log10(Apot))**2.9))*\
#             #          (5.5*np.exp(1.33*(np.log10(Apot))**2.9) + 1)
                
#             #     rl = rlegg*(j0 + j1*np.exp(-j2*(np.log10(qratio))**j3))
                
#                 # Rlgrid[k1][k2] = 0

# m2limit = np.array(m2limit)
# Aorblimit = np.array(Aorblimit)
# ecclimit = np.array(ecclimit)
# Aorblimit_allowed = np.array(Aorblimit_allowed)
# m2limit_allowed = np.array(m2limit_allowed)
# ecclimit_allowed = np.array(ecclimit_allowed)
# m2limit/=Msun
# Aorblimit/=Rsun
# m2limit_allowed/=Msun
# Aorblimit_allowed/=Rsun

# fig = plt.figure()
# # ax = plt.axes(projection="3d")
# ax = plt.axes()
# scatt2 = ax.scatter(Aorblimit,m2limit,\
#                     c='g',marker='.',label="Excluded")
# scatt = ax.scatter(Aorblimit_allowed,m2limit_allowed,\
#                    c=ecclimit_allowed,cmap=plt.cm.magma,label="Allowed")
# ax.set_xlabel("Orbital separation ($R_{\odot}$)", fontsize=14)
# ax.set_ylabel("Companion mass ($M_{\odot}$)", fontsize=14)
# # ax.set_zlabel("Eccentricity", fontsize=14)
# ax.tick_params(axis='both', which='major', labelsize=14)
# cbar = fig.colorbar(scatt, ax=ax)
# cbar.set_label(r'Eccentricity',rotation=90,fontsize=14,\
#                labelpad=22)
# cbar.ax.tick_params(labelsize=14)
# plt.legend(loc="upper right")
# plt.show()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 11:18:03 2024

@author: erc_magnesia_raj
"""
import numpy
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models,fitting
from scipy import integrate

Mpc = 3.086e24
dist = 3.95

#Compute line luminosity and the associated uncertainty
def line_lumin(peak, width, peak_err, width_err):
    flux = peak * width * np.sqrt(2*np.pi)
    flux_err = flux*np.sqrt((peak_err/peak)**2 + (width_err/width)**2)
    lumin = (4.0*np.pi*(dist*Mpc)**2)*flux
    lumin_err = (4.0*np.pi*(dist*Mpc)**2)*flux_err
    return lumin, lumin_err

#Speed of light
c = 3e8

dates = ["23-10-2009",\
          "11-12-2009","12-12-2009","13-12-2009",\
          "14-12-2009","15-12-2009","20-12-2009",\
          "21-12-2009","22-12-2009","24-12-2009"]
fname = ["cbscience_n1_calib.fits","cbscience_n2_calib.fits",\
          "cbscience_n3_calib.fits","cbscience_n4_calib.fits",\
          "cbscience_n5_calib.fits","cbscience_n6_calib.fits",\
          "cbscience_n8_calib.fits","cbscience_n9_calib.fits",\
          "cbscience_n10_calib.fits","cbscience_n11_calib.fits"]
mjd_optical = [55127.0,55176.0,55177.0,55178.0,55179.0,55180.0,\
               55185.0,55186.0,55187.0,55189.0]
LheII,LheIIerr = [[],[]]
nrow = 5
ncol = 2
c1 = 0
c2 = 0
count = 0
index = 0

vox3a,vox3b,vheii,vULX = [[],[],[],[]]
dvox3a,dvox3b,dvheii,dvULX = [[],[],[],[]]
vdispox3a,vdispox3b,vdispheii,vdispulx = [[],[],[],[]]
dvdispox3a,dvdispox3b,dvdispheii,dvdispulx = [[],[],[],[]]

lamdaw1,lamdaw2,lamdaw3,lamdaw4 = [[],[],[],[]]
dlamdaw1,dlamdaw2,dlamdaw3,dlamdaw4 = [[],[],[],[]]
fwhmw1,fwhmw2,fwhmw3,fwhmw4 = [[],[],[],[]]
dfwhmw1,dfwhmw2,dfwhmw3,dfwhmw4 = [[],[],[],[]]

fig,ax = plt.subplots(nrows=nrow,ncols=ncol)
for file in range(len(fname)):
    
    hdu = fits.open(fname[file])
    flux = hdu[0].data
    header = hdu[0].header
    deltaL = header['CD1_1']
    lamda_i = header['CRVAL1']
    ncomb = header['NCOMBINE']
    Nlamda = len(flux)
    lamda_f = lamda_i + Nlamda*deltaL
    wave = np.arange(lamda_i, lamda_f, deltaL)
    
    wmin = 4700
    wmax = 5200
    fluxem = flux[wave>wmin]
    waveem = wave[wave>wmin]
    fluxem = fluxem[waveem<wmax]
    waveem = waveem[waveem<wmax]
    dwave = waveem[1]-waveem[0]   
    
    #Rest-frame wavelengths
    heii = 4686.0
    hbeta = 4861.363
    ox3_1 = 4958.911
    ox3_2 = 5006.843
    
    #Initialise fitting parameters
    mu1 = 4865.6
    mu2 = 4964.0
    mu3 = 5011.0
    sig1 = 1.0
    sig2 = 1.0
    sig3 = 1.0
    amp1 = 1e-17
    amp2 = 1e-17
    amp3 = 1e-17
    
    cont = models.Polynomial1D(1)
    g1 = models.Gaussian1D(amplitude=amp1, mean=mu1, stddev=sig1)
    g2 = models.Gaussian1D(amplitude=amp2, mean=mu2, stddev=sig2)
    g3 = models.Gaussian1D(amplitude=amp3, mean=mu3, stddev=sig3)
        
    #Fit the three nebular emission lines
    g_emission = g1 + g2 + g3 
    g_cont = cont
    
    g_total = g_emission + g_cont    
    fit_g = fitting.LevMarLSQFitter()
    g = fit_g(g_total, waveem, fluxem, maxiter = 1000)
    fit_errs = np.sqrt(np.diag(fit_g.fit_info['param_cov']))
    
    wmin2 = 4640
    wmax2 = 4720
    fluxem2 = flux[wave>wmin2]
    waveem2 = wave[wave>wmin2]
    fluxem2 = fluxem2[waveem2<wmax2]
    waveem2 = waveem2[waveem2<wmax2]
    dwave2 = waveem2[1]-waveem2[0]   
    
    #Rest-frame wavelengths
    heii = 4686.0
    
    #Initialise fitting parameters
    mu4 = 4689.0
    sig4 = 2.0
    amp4 = 5e-18
    if(file==6):
        sig4 = 0.5
        amp4 = 1e-18
    
    cont2 = models.Polynomial1D(2)
    g4 = models.Gaussian1D(amplitude=amp4, mean=mu4, stddev=sig4)
        
    #Fit the three nebular emission lines
    g_emission2 = g4
    g_cont2 = cont2
    
    g_total2 = g_emission2 + g_cont2    
    fit_g2 = fitting.LevMarLSQFitter()
    g2 = fit_g2(g_total2, waveem2, fluxem2, maxiter = 1000)
    if((fit_g2.fit_info['param_cov']) is not None):
        fit_errs2 = np.sqrt(np.diag(fit_g2.fit_info['param_cov']))
    elif(fit_g2.fit_info['param_cov'] is None):
        np.zeros(3)
    
    amp1 = g.parameters[0]
    amp1_err = fit_errs[0]
    best_w1 = g.parameters[1]
    best_w1err = fit_errs[1]
    best_fwhm1 = g.parameters[2]
    best_fwhm1err = fit_errs[2]
    
    amp2 = g.parameters[3]
    amp2_err = fit_errs[3]
    best_w2 = g.parameters[4]
    best_w2err = fit_errs[4]
    best_fwhm2 = g.parameters[5]
    best_fwhm2err = fit_errs[5]
    
    amp3 = g.parameters[6]
    amp3_err = fit_errs[6]
    best_w3 = g.parameters[7]
    best_w3err = fit_errs[7]
    best_fwhm3 = g.parameters[8]
    best_fwhm3err = fit_errs[8]
    
    amp4 = g2.parameters[0]
    amp4_err = fit_errs2[0]
    best_w4 = g2.parameters[1]
    best_w4err = fit_errs2[1]
    best_fwhm4 = g2.parameters[2]
    best_fwhm4err = fit_errs[2]
    
    # print(mjd_optical[file],best_w2,best_w2err,best_fwhm2,best_fwhm2err)
    
    heii_lumin,heii_luminerr =\
        line_lumin(amp4,best_fwhm4,amp4_err,best_fwhm4err)
    # heiierr1 = line_lumin(waveem2,amp4+amp4_err,best_w4,\
    #                        best_fwhm4+best_fwhm4err) -\
    #            line_lumin(waveem2,amp4,best_w4,best_fwhm4)
    # heiierr2 = line_lumin(waveem2,amp4,best_w4,best_fwhm4) -\
    #            line_lumin(waveem2,amp4-amp4_err,best_w4,\
    #                       best_fwhm4-best_fwhm4err)
    # heii_luminerr = 0.5*(heiierr1 + heiierr2)
    
    LheII.append(heii_lumin)
    LheIIerr.append(heii_luminerr)
    
    lamdaw1.append(best_w1)
    lamdaw2.append(best_w2)
    lamdaw3.append(best_w3)
    lamdaw4.append(best_w4)
    
    dlamdaw1.append(best_w1err)
    dlamdaw2.append(best_w2err)
    dlamdaw3.append(best_w3err)
    dlamdaw4.append(best_w4err)
    
    fwhmw1.append(best_fwhm1)
    fwhmw2.append(best_fwhm2)
    fwhmw3.append(best_fwhm3)
    fwhmw4.append(best_fwhm4)
    
    dfwhmw1.append(best_fwhm1err)
    dfwhmw2.append(best_fwhm2err)
    dfwhmw3.append(best_fwhm3err)
    dfwhmw4.append(best_fwhm4err)
    
    #Non-relativistic Doppler shift
    v1 = c*(1 - hbeta/best_w1) 
    dv1 = c*(hbeta/best_w1)*(best_w1err/best_w1)    
    v1_disp = c*(1 - hbeta/(best_w1+best_fwhm1)) -\
              c*(1 - hbeta/(best_w1-best_fwhm1))
    q1 = c*hbeta/(best_w1+best_fwhm1)**2
    q2 = c*hbeta/(best_w1-best_fwhm1)**2
    dv1_disp = np.sqrt((q1**2)*(best_fwhm1err**2) +\
                            (q2**2)*(best_fwhm1err**2) +\
                            (q1**2)*(best_w1err**2) + (q2**2)*(best_w1err**2))
              
    # print("Velocity [Hbeta] ", v1/1000-470," ± ",dv1/1000)
    # print("Velocity dispersion [Hbeta] ",v1_disp/1000,\
    #       " ± ",dv1_disp/1000)
    
    v2 = c*(1 - ox3_1/best_w2) 
    dv2 = c*(ox3_1/best_w2)*(best_w2err/best_w2)
    v2_disp = c*(1 - ox3_1/(best_w2+best_fwhm2)) -\
              c*(1 - ox3_1/(best_w2-best_fwhm2))
    q1 = c*ox3_1/(best_w2+best_fwhm2)**2
    q2 = c*ox3_1/(best_w2-best_fwhm2)**2
    dv2_disp = np.sqrt((q1**2)*(best_fwhm2err**2) +\
                      (q2**2)*(best_fwhm2err**2) +\
                      (q1**2)*(best_w2err**2) + (q2**2)*(best_w2err**2))
    
    # print("Velocity [OIII4959] ",v2/1000-470," ± ",dv2/1000)
    # print("Velocity dispersion [OIII4960] ",v2_disp/1000,\
    #       " ± ",dv2_disp/1000)

    v3 = c*(1 - ox3_2/best_w3) 
    dv3 = c*(ox3_2/best_w3)*(best_w3err/best_w3)
    v3_disp = c*(1 - ox3_2/(best_w3+best_fwhm3)) -\
              c*(1 - ox3_2/(best_w3-best_fwhm3))
    q1 = c*ox3_2/(best_w3+best_fwhm3)**2
    q2 = c*ox3_2/(best_w3-best_fwhm3)**2
    dv3_disp = np.sqrt((q1**2)*(best_fwhm3err**2) +\
                      (q2**2)*(best_fwhm3err**2) +\
                      (q1**2)*(best_w3err**2) + (q2**2)*(best_w3err**2))

    print("Velocity [OIII5007] ",v3/1000-470," ± ",dv3/1000)
    print("")
    # print("Velocity dispersion [OIII5008] ",v3_disp/1000,\
    #       " ± ",dv3_disp/1000)
    # print("")
    
    v4 = c*(1 - heii/best_w4) 
    v4_disp = c*(1 - heii/(best_w4+best_fwhm4)) -\
              c*(1 - heii/(best_w4-best_fwhm4))    

    # v4axis = c*(1-hbeta/best_w1)/1000 #in km/s
        
    # v4limx1 = c*(1-ox3_1/4945)/1000 #in km/
    # v4limx2 = c*(1-ox3_1/4980)/1000 #in km/s

    v4limx1 = c*(1-ox3_2/4990)/1000 #in km/
    v4limx2 = c*(1-ox3_2/5028)/1000 #in km/s

    # print("Velocity [HeII4686] ",v4/1000," ± ",dv4/1000)
    # print("Velocity dispersion [HeII4686] ",v4_disp/1000,\
    #       " ± ",dv4_disp/1000)
    # print("")

    vulx = v4 - (1./3.)*(v1 + v2 + v3)
    dvulx = (1./3.)*np.sqrt(dv1**2 + dv2**2 + dv3**2)
    vulx_disp = v4_disp
        
    # print("Velocity [ULX] ",vulx/1000," ± ",dvulx/1000)
    # print("Velocity dispersion [ULX] ",vulx_disp/1000)
    # print("")
    
    vox3a.append(v2/1000)
    vox3b.append(v3/1000)
    vheii.append(v4/1000)
    vULX.append(vulx/1000)
    
    vdispox3a.append(v2_disp/1000)
    vdispox3b.append(v3_disp/1000)
    vdispheii.append(v4_disp/1000)
    
    dvox3a.append(dv2/1000)
    dvox3b.append(dv3/1000)
    dvheii.append(0)
    dvULX.append(dvulx/1000)
    
    dvdispox3a.append(dv2_disp/1000)
    dvdispox3b.append(dv3_disp/1000)
    dvdispheii.append(0)
    
    c1 = count 
    if(count==nrow):
        c2+=1
        c1=0
        count=0
    
    lab = r"[OIII] 5007: " + dates[index]
    # lab = r"[OIII] 4959: " + dates[index]
    # lab = r"[HeII] 4686: " + dates[index]
    
    ax[c1][c2].step(waveem,fluxem/1e-17,label=lab)
    ax[c1][c2].plot(waveem,g(waveem)/1e-17,'r--')
    # ax[c1][c2].step(waveem2,fluxem2/1e-17,label=lab)
    # ax[c1][c2].plot(waveem2,g2(waveem2)/1e-17,'r--')
    ax[c1][c2].tick_params(axis='both', which='major', labelsize=12)
    ax[c1][c2].set_xlim(4990,5028)
    ax[c1][c2].set_ylim(-0.1,6.2)
    
    if(c1==0):
        secax = ax[c1][c2].twiny()
        secax.tick_params(axis='both', which='major', labelsize=12)
        secax.set_xlim(v4limx1,v4limx2)
        secax.set_xlabel('Velocity [km/s]',fontsize=12)

    if(c1==2 and c2==0):
        ax[c1][c2].set_ylabel(\
        r"Flux [10$^{-17}$ erg cm$^{-2}$ s$^{-1}$ $\AA$ $^{-1}$]",\
                    fontsize=12)

    if(c1==2 and c2==1):
        ax[c1][c2].set_ylabel(\
        r"Flux [10$^{-17}$ erg cm$^{-2}$ s$^{-1}$ $\AA$ $^{-1}$]",\
                    fontsize=12)

    if(c1==nrow-1 and c2==0):
        ax[c1][c2].set_xlabel(r"Wavelength [$\AA$]",fontsize=12)
        
    if(c1==nrow-1 and c2==1):
        ax[c1][c2].set_xlabel(r"Wavelength [$\AA$]",fontsize=12)

    if(c1<nrow-1 and c2==0):
        ax[c1][c2].set_xticks([])
    if(c1<nrow-1 and c2==1):
        ax[c1][c2].set_xticks([])

    ax[c1][c2].legend(loc="upper left")
        
    count += 1
    index += 1

plt.subplots_adjust(hspace=0)
plt.subplots_adjust(wspace=0.3)
plt.show()

mjd_optical = np.array(mjd_optical)
vheii = np.array(vheii)
vULX = np.array(vULX)
dvheii = np.array(dvheii)
dvULX = np.array(dvULX)
vdispheii = np.array(vdispheii)
dvdispheii = np.array(dvdispheii)
LheII = np.array(LheII)
LheIIerr = np.array(LheIIerr)

mask = np.ones(len(mjd_optical),dtype=bool)
# mask[[-1,-2,-5]] = False
mask[[8]] = False
mjdsheii = mjd_optical[mask]
vheii = vheii[mask]
vULX = vULX[mask]
dvheii = dvheii[mask]
dvULX = dvULX[mask]
vdispheii = vdispheii[mask]
dvdispheii = dvdispheii[mask]
LheII = LheII[mask]
LheIIerr = LheIIerr[mask]

Z = np.column_stack((mjd_optical,lamdaw1,dlamdaw1,fwhmw1,dfwhmw1))
np.savetxt("emission_line_fits1.dat",Z,fmt='%s',delimiter='   ')
Z = np.column_stack((mjd_optical,lamdaw2,dlamdaw2,fwhmw2,dfwhmw2))
np.savetxt("emission_line_fits2.dat",Z,fmt='%s',delimiter='   ')
Z = np.column_stack((mjd_optical,lamdaw3,dlamdaw3,fwhmw3,dfwhmw3))
np.savetxt("emission_line_fits3.dat",Z,fmt='%s',delimiter='   ')
Z = np.column_stack((mjd_optical,lamdaw4,dlamdaw4,fwhmw4,dfwhmw4))
np.savetxt("emission_line_fits4.dat",Z,fmt='%s',delimiter='   ')

vsystemic = 470
vox3a = np.array(vox3a)
vox3b = np.array(vox3b)
vheii = np.array(vheii)

plt.figure()
plt.subplot(321)
plt.errorbar(mjd_optical,vox3a-vsystemic,\
             yerr=dvox3a,fmt='ko',label=r"[OIII] 5007")
plt.errorbar(mjd_optical,vox3b-vsystemic,\
             yerr=dvox3b,fmt='bo',label=r"[OIII] 4959")
plt.errorbar(mjdsheii,vheii-vsystemic,\
             yerr=dvheii,fmt='ro',label=r"[HeII] 4686")
# plt.errorbar(mjdsheii,vULX,yerr=dvULX,fmt='go',label=r"ULX")
plt.tick_params(axis='both', which='major', labelsize=14)
plt.xticks([])
plt.xlabel("Time [MJD]",fontsize=14)
plt.xlim(55115,55195)
plt.ylim(-280,-20)
plt.legend(loc="best")
plt.ylabel("Velocity [km/s]",fontsize=14)

plt.subplot(323)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.errorbar(mjd_optical,vdispox3a,yerr=dvdispox3a,fmt='ko',\
              label=r"[OIII] 5007")
plt.errorbar(mjd_optical,vdispox3b,\
              yerr=dvdispox3b,fmt='bo',label=r"[OIII] 4959")
plt.errorbar(mjdsheii,vdispheii,\
              yerr=dvdispheii,fmt='ro',label=r"[HeII] 4686")
plt.subplots_adjust(hspace=0)
plt.xlim(55115,55195)
plt.ylim(90,1000)
plt.ylabel(r"$v_{disp}$ [km/s]",fontsize=14)
plt.xlabel("Time [MJD]",fontsize=14)
plt.legend(loc="best")
# plt.show()

##Swift-XRT LC
mjd_xray,rate,rate_err = [[],[],[]]
with open("ngc1313x2_swiftlc.qdp") as fi:
    for line in fi:
        line = line.split()
        if(len(line)>0):
            if(line[0]!='!' and line[0]!='READ' and line[0]!='!MJD'\
                and line[0]!='NO'):
                mjd_xray.append(float(line[0]))
                mjderr = 0.5*(float(line[1])+float(line[2]))
                rate.append(float(line[3]))
                rate_err.append(0.5*(abs(float(line[4]))+abs(float(line[5]))))

mjd_xray,rate,rate_err = zip(*sorted(zip(mjd_xray,rate,rate_err)))
mjd_xray = np.array(mjd_xray)
rate = np.array(rate)
rate_err = np.array(rate_err)

rate = rate[mjd_xray>55115]
rate_err = rate_err[mjd_xray>55115]
mjd_xray = mjd_xray[mjd_xray>55115]
rate = rate[mjd_xray<55195]
rate_err = rate_err[mjd_xray<55195]
mjd_xray = mjd_xray[mjd_xray<55195]

_,cont_lumin = np.loadtxt("continuum_luminosities.dat",\
                          skiprows=0,unpack=True)

# plt.figure()
plt.subplot(322)
plt.errorbar(mjdsheii,LheII/1e35,\
             yerr=LheIIerr/1e35,fmt='ro')
plt.tick_params(axis='both', which='major', labelsize=14)
plt.xlim(55115,55195)
plt.ylabel(r"L$_{HeII}$ [10$^{35}$ erg s$^{-1}$]",fontsize=14)
plt.xticks([])
plt.subplot(324)
plt.plot(mjd_optical,cont_lumin/1e37,'go')
plt.ylim(0.5,3.5)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.ylabel(r"L$_{c}$ [10$^{37}$ erg s$^{-1}$]",fontsize=14)
plt.xlim(55115,55195)
plt.xticks([])
plt.subplot(326)
plt.errorbar(mjd_xray,rate,yerr=rate_err,fmt='mo')
plt.tick_params(axis='both', which='major', labelsize=14)
plt.ylabel(r"$\it{Swift}$ count rate [s$^{-1}$]",fontsize=14)
plt.subplots_adjust(hspace=0)
plt.xlim(55115,55195)
plt.xlabel("Time [MJD]",fontsize=14)
plt.ylim(0.01,0.27)
plt.subplots_adjust(wspace=0.5)
plt.show()

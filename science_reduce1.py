import numpy as np
import sys
import copy
from pyraf import iraf
from pyraf.iraf import gemini, gemtools, gmos, onedspec
import fileSelect as fs
import os, glob

def gmos_ls_proc():

    '''
    GMOS Data Reduction Cookbook companion script to the chapter:
       "Reduction of Longslit Spectra with PyRAF"

    PyRAF script to:
    Process GMOS spectra for NGC 1313 X-2

    The names for the relevant header keywords and their expected values are
    described in the DRC chapter entitled "Supplementary Material"

    Perform the following starting in the parent work directory:
        cd /path/to/work_directory

    Place the fileSelect.py module in your work directory. Now execute this
    script from the unix prompt:
        python gmos_ls_proc.py
    '''

    print ("### Begin Processing GMOS/Longslit Images ###")
    print ("###")
    print ("=== Creating MasterCals ===")

    # This whole example depends upon first having built an sqlite3 database of metadata:
    dbFile='/home/irafuser/ngc1313x2/data/obsLog.sqlite3'

    # From the work_directory:
    # Create the query dictionary of essential parameter=value pairs.
    # Select bias exposures within ~2 months of the target observations:
    qd = {'Full':{'use_me':1,
          'Instrument':'GMOS-S','CcdBin':'2 2','RoI':'Full',
          'Disperser':'B600+_%','CentWave':'456.0','AperMask':'1.0arcsec',
          'Object':'NGC 1313 X-2%',
          'DateObs':'2009-12-09:2009-12-12'}
         }
           
    print (" --Creating Bias MasterCal--")

    # Set the task parameters.
    gemtools.gemextn.unlearn()
    gmos.gbias.unlearn()
    biasFlags = {'logfile':'biasLog.txt',\
    'rawpath':'/home/irafuser/ngc1313x2/data','fl_vardq':'yes','verbose':'no'}
    regions = ['Full']
    
    #Generate and stack bias files
    for r in regions:

        SQL = fs.createQuery('bias', qd[r])
        biasFiles = fs.fileListQuery(dbFile, SQL, qd[r])
                
        if len(biasFiles) > 1:
            with open('bias.lis', 'w') as f:
                [f.write(x+'\n') for x in biasFiles]
            gmos.gbias('@bias.lis', 'MCbiasFull.fits', **biasFlags)

    qd['Full'].update({'DateObs':'*'})
    gmos.gireduce.unlearn()
    gmos.gsflat.unlearn()

    flatFlags = {
        'fl_over':'yes','fl_trim':'yes','fl_bias':'yes','fl_dark':'no',
        'fl_fixpix':'no','fl_oversize':'no','fl_vardq':'yes','fl_fulldq':'yes',
        'rawpath':'/home/irafuser/ngc1313x2/data','fl_inter':'no','fl_detec':'yes',
        'function':'spline3','order':'13,11,28',
        'logfile':'gsflatLog.txt','verbose':'no'
        }
     
    for r in regions:
        
        qr = qd[r]
        flatFiles = fs.fileListQuery(dbFile, fs.createQuery('gcalFlat', qr), qr)
        
        print('MCbias'+r)
        if len(flatFiles) > 0:
            gmos.gsflat (','.join(str(x) for x in flatFiles), 'MCflat456'+r,
                    bias='MCbias'+r, **flatFlags)

    print ("=== Processing Science Files ===")
    print (" -- Performing Basic Processing --")

    #Bias subtraction on arc exposures
    gmos.gsreduce.unlearn()
    sciFlags = {
        'fl_over':'yes','fl_trim':'yes','fl_bias':'yes','fl_gscrrej':'yes',
        'fl_dark':'no','fl_flat':'yes','fl_gmosaic':'yes','fl_fixpix':'no',
        'fl_gsappwave':'yes','fl_oversize':'no',
        'fl_vardq':'yes','fl_fulldq':'yes','rawpath':'/home/irafuser/ngc1313x2/data',
        'fl_inter':'no','logfile':'gsreduceLog.txt','verbose':'no'
    }
    arcFlags = copy.deepcopy(sciFlags)
    arcFlags.update({'fl_flat':'no','fl_vardq':'no','fl_fulldq':'no'})
    stdFlags = copy.deepcopy(sciFlags)
    stdFlags.update({'fl_fixpix':'yes','fl_vardq':'no','fl_fulldq':'no'})

    print ("  - Arc exposures -")
    r = 'Full'
    qr = qd[r]
    arcFiles = fs.fileListQuery(dbFile, fs.createQuery('arc', qr), qr)
    print(len(arcFiles))
    
    if len(arcFiles) > 0:
        gmos.gsreduce (','.join(str(x) for x in arcFiles),
                       bias='MCbias'+r, **arcFlags)

    #Bias subtraction and flat fielding on science exposures
    print ("  - Science exposures -")
    r = 'Full'
    qr = qd[r]
    sciFiles = fs.fileListQuery(dbFile, fs.createQuery('sciSpec', qd[r]), qd[r])
    
    if len(sciFiles) > 0:
        gmos.gsreduce(','.join(str(x) for x in sciFiles), bias='MCbias'+r,
                  flatim='MCflat456'+r, **sciFlags)

    print (" -- Determine wavelength calibration --")
    gmos.gswavelength.unlearn()

    waveFlags = {
        'coordlist':'gmos$data/CuAr_GMOS.dat','fwidth':6,'nsum':50,
        'function':'chebyshev','order':5,
        'fl_inter':'no','logfile':'gswaveLog.txt','verbose':'no'
        }
        
#    The fit to the dispersion relation should be performed interactively.
#    Here we will use a previously determined result.
#    Need to select specific wavecals to match science exposures.

    prefix = 'gsS20091023S0'
    for arc in ['079']:
        gmos.gswavelength (prefix+arc, **waveFlags)

    print (" -- Performing Advanced Processing --")
    print (" -- Combine exposures, apply dispersion, subtract sky --")

    gmos.gstransform.unlearn()
    transFlags = {
        'fl_vardq':'yes','interptype':'linear','fl_flux':'yes',
        'lambda1':'3100.0','lambda2':'10000.0','dx':'0.45',
        'logfile':'gstransLog.txt'
    }

#     Process the science targets.
#     Use a dictionary to associate science targets with Arcs and sky regions.

    prefix = "gs"
    sciTargets = {
        'NGC 1313 X-2':{'arc':'gsS20091023S0079','sky':'850:1060,1280:1520'}
    }
    
    for r in regions:
    
        qs = qd[r]
        arcFiles = fs.fileListQuery(dbFile, fs.createQuery('arc', qs), qs)[0]
        arcFiles = 'gs' + str(arcFiles)

        sciFiles = fs.fileListQuery(dbFile, fs.createQuery('sciSpec', qs), qs)
        sciFiles = ','.join(prefix+str(x) for x in sciFiles)
        gmos.gstransform(sciFiles, wavtraname=arcFiles, **transFlags)
        
        #Stack and background subtract
            
if __name__ == "__main__":
    gmos_ls_proc()


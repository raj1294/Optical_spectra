#Reduce Swift/UVOT data and extract light-curves
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import glob, os
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS

day = 86400

stringmv,stringopen,stringrm,stringseg,\
stringdetect = [[],[],[],[],[]]

#ULX location
raulx = 202.4108333
deculx = 58.4183333

#Window
#Top left
rabox1 = 202.4202182
decbox1 = 58.4275381
#Bottom left
rabox2 = 202.4203006
decbox2 = 58.4126201
#Top right
rabox3 = 202.3922212
decbox3 = 58.4274915
#Bottom right
rabox4 = 202.3918206
decbox4 = 58.4127684

#SNR threshold (image detection)
snr = 1.0

for file in glob.glob("0*/uvot/image/sw*sk*img*.gz"):
    
    imagefile = file.split("/")[-1] 
    
    path = file.split("/")
    pathdir = path[0] + "/" + path[1] + "/" + path[2] + "/"
    
    obsiduvot = file.split("sw")[1].split("sk.img.gz")[0].split("u")[0]
    instuv = file.split("_sk.img.gz")[0].split("sw")[1].split(obsiduvot)[1]
    Expfile = "sw" + obsiduvot + instuv + "_ex.img.gz"
    
    #Move files
    commmv1 = "cp " + file + " new/"
    commmv2 = "cp " + pathdir + Expfile + " new/"
    stringmv.append(commmv1)
    stringmv.append(commmv2)
    
    #Open files
    commopen1 = "open new/" + imagefile 
    commopen2 = "open new/" + Expfile 
    stringopen.append(commopen1)
    stringopen.append(commopen2)
    
    #Delete .gz files
    commrm = "rm -f new/*.gz"
    stringrm.append(commrm)

    #Cut out a window of the Swift/UVOT image to 
    #more accurately capture the background local to the source
    hdu = fits.open(file)
    header = hdu[1].header    
    wcs_image = WCS(fits.getheader(file, 1))    
    sky_pos1 = [rabox1,decbox1]
    sky_pos2 = [rabox2,decbox2]
    sky_pos3 = [rabox3,decbox3]
    sky_pos4 = [rabox4,decbox4]
    xbox1, ybox1 = wcs_image.world_to_pixel_values([sky_pos1])[0].astype(int)
    xbox2, ybox2 = wcs_image.world_to_pixel_values([sky_pos2])[0].astype(int)
    xbox3, ybox3 = wcs_image.world_to_pixel_values([sky_pos3])[0].astype(int)
    xbox4, ybox4 = wcs_image.world_to_pixel_values([sky_pos4])[0].astype(int)
    
    outimagefile = "sw" + obsiduvot + instuv + "_img_filt.fits"
    outexpfile = "sw" + obsiduvot + instuv + "_exp_filt.fits"
    
    #Segment image 
    newimagefile = imagefile[0:-3] 
    newexpfile = Expfile[0:-3]
    
    commcpimage = "imcopy '" + newimagefile + "[" + str(xbox2) + ":" +\
    str(xbox4) + "," + str(ybox2) + ":" + str(ybox1) + "]'" +\
    " " + outimagefile
    #Segment exposure map 
    commcpexpimage = "imcopy '" + newexpfile + "[" + str(xbox2) + ":" +\
    str(xbox4) + "," + str(ybox2) + ":" + str(ybox1) + "]'" +\
    " " + outexpfile
    
    stringseg.append(commcpimage)
    stringseg.append(commcpexpimage)
    
    #Run source detection algorithm
    
    #Output souce file
    outsrcfile = "detect_" + obsiduvot + ".fits"
    commdetect = "uvotdetect plotsrc=no infile=" + outimagefile + " " +\
                 "outfile=" + outsrcfile + " expfile=" + outexpfile +\
                 " threshold=" + str(snr) + " clobber=yes"
    stringdetect.append(commdetect)
            
np.savetxt("move.sh",stringmv,fmt='%s',delimiter=' ')
np.savetxt("open.sh",stringopen,fmt='%s',delimiter=' ')
np.savetxt("remove.sh",stringrm,fmt='%s',delimiter=' ')

np.savetxt("segment.sh",stringseg,fmt='%s',delimiter=' ')
os.system("mv segment.sh new/")

np.savetxt("detect.sh",stringdetect,fmt='%s',delimiter=' ')
os.system("mv detect.sh new/")

os.system("chmod u+x *.sh")
os.system("chmod u+x new/*.sh")

#Extract UVOT light-curve
timeuvot,fluxuvot,fluxerruvot = [[],[],[]]
for detectfile in glob.glob("detect*.fits"):
    
    #Reference band (source LC)
    hdulistref = fits.open(detectfile)
    dataref = hdulistref[1].data
    hdr = hdulistref[1].header
    
    tstart = hdr['TSTART']
    tstop = hdr['TSTOP']
    mjdref = hdr['MJDREFI'] + hdr['MJDREFF']   
    mjdstart = mjdref + tstart/day
    mjdstop = mjdref + tstop/day
    timeobs = 0.5*(mjdstart + mjdstop)
        
    rasrc = dataref['RA']  
    decsrc = dataref['DEC']
    fluxsrc = dataref['FLUX']
    fluxerr = dataref['FLUX_ERR']
    
    culx = SkyCoord(raulx,deculx,unit="deg",frame='icrs')                            
    csrc = SkyCoord(rasrc,decsrc,unit="deg",frame="icrs")    
    separation = culx.separation(csrc)
    sepdeg = separation.deg
    
    sepdegmin = np.min(sepdeg)
    fluxsrc = fluxsrc[sepdeg<=sepdegmin]
    fluxerr = fluxerr[sepdeg<=sepdegmin]
    rasrc = rasrc[sepdeg<=sepdegmin]
    decsrc = decsrc[sepdeg<=sepdegmin]
    
    timeuvot.append(timeobs)
    
    if(len(rasrc)==1):
        
        rasrc = rasrc[0]
        decsrc = decsrc[0]
        fluxsrc = fluxsrc[0]
        fluxerr = fluxerr[0]
        
        fluxuvot.append(fluxsrc)
        fluxerruvot.append(fluxerr)

timeuvot = np.array(timeuvot)
fluxuvot = np.array(fluxuvot)
fluxerruvot = np.array(fluxerruvot)

plt.errorbar(timeuvot-np.min(timeuvot),fluxuvot,yerr=fluxerruvot,fmt='k.')
plt.show()

        
                
        

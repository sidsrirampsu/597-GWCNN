import numpy as np
import os
from LightPipes import *
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
import random
from numpy.lib.index_tricks import index_exp
np.random.seed(7561)

# Number of images to be generated
Ntot=500

# Defining variables: Wavelength of 1064 Nanometer, 40*40 mm2 sqaure grid with 128x128 pixels
wavelength = 1064*nm
size = 40*mm
Npix = 128
w0=3*mm
LG=False

# Maximum number of modes to be generated 
mode_max=6
mode_m=np.random.randint(0,mode_max,Ntot)
mode_n=np.random.randint(0,mode_max,Ntot)

# Noise distribution
mean = 0
sigma = np.random.uniform(0.05,0.9, Ntot)

# Offset 
x_offset = np.random.randint(-30,30, Ntot)
y_offset = np.random.randint(-30,30, Ntot)

#The Begin command generates a field with amplitude 1.0 and phase zero, a plane wave. 
#So, all the 128x128 elements of array contain the complex number: 1.0 + j0.0
F0=Begin(size,wavelength,Npix)

# Catalog is the record of all the images and it's parameters generated
catalog = open("sample_catalog.txt","w")
m_list = []
n_list = []

for num in range(Ntot):
   nn=mode_n[num]
   mm=mode_m[num]
   x_off=x_offset[num] 
   y_off=y_offset[num] 
   F1=GaussBeam(F0, w0, LG=LG, n=nn, m=mm)
   Iimg=Intensity(F1,1) #Intensity is calculated and normalized to 255 (2 -> 255, 1 -> 1.0, 0 -> not normalized)
   
   # Adding Noise
   gauss = np.random.normal(mean,sigma[num],Iimg.shape)
   gauss_img = gauss.reshape(Iimg.shape)
   noisyIimg = Iimg + gauss_img
   
   # Index creates unique IDs for each image 
   index = str(int(num +1)).zfill(5)
   noiseFile = f'{index}_HG_{nn}_{mm}'
   fname = f'{noiseFile}.png'
   plt.imsave(fname, noisyIimg, cmap='gray')

   # Adding offset 
   im = Image.open(fname)
   offset_im = ImageChops.offset(im, x_off, y_off)

   # Naming the images before saving
   sigma_value = format(sigma[num], '.2f')
   catalog.write("%s %d %d %s %d %d \n"%(index,nn,mm,sigma_value,x_off,y_off))

   n_list.append(nn)
   m_list.append(mm)
   
   if x_off < 0 and y_off < 0:
      x_off = str(abs(x_off)).zfill(2)
      y_off = str(abs(y_off)).zfill(2)
      filename=f'{noiseFile}_{sigma_value}_n{x_off}_n{y_off}.png'
   elif x_off >= 0 and y_off >= 0:
      x_off = str(x_off).zfill(2)
      y_off = str(y_off).zfill(2)
      filename=f'{noiseFile}_{sigma_value}_p{x_off}_p{y_off}.png'
   elif x_off >= 0 and y_off < 0:
      x_off = str(x_off).zfill(2)
      y_off = str(abs(y_off)).zfill(2)
      filename=f'{noiseFile}_{sigma_value}_p{x_off}_n{y_off}.png'
   elif x_off < 0 and y_off >= 0:
      x_off = str(abs(x_off)).zfill(2)
      y_off = str(y_off).zfill(2)
      filename=f'{noiseFile}_{sigma_value}_n{x_off}_p{y_off}.png'

      
   os.rename(fname, filename)
   offset_im.save(filename)

plt.figure(figsize=(10,20)) 
plt.suptitle('Distribution of dataset')
plt.subplot(2,1,1)

plt.xlabel('TEM m')
plt.ylabel('TEM n')
sc = plt.scatter(m_list, n_list, c=sigma, cmap=cm)
cbar = plt.colorbar(sc)
cbar.mappable.set_clim(vmin=0.05,vmax=0.9)

catalog.close()
            

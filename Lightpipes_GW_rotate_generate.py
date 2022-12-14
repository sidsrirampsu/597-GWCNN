import numpy as np
import os
from LightPipes import *
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
import random
from numpy.lib.index_tricks import index_exp
np.random.seed(7561)

# Number of images to be generated
Ntot=100

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
   noiseFile = f'{index}_HG_{mm}_{nn}'
   fname = f'{noiseFile}.png'
   plt.imsave(fname, noisyIimg, cmap='gray')

   # Adding offset 
   im = Image.open(fname)
   offset_im = ImageChops.offset(im, x_off, y_off)
   rotate_im = offset_im.rotate(90)
   # Naming the images before saving
   sigma_value = format(sigma[num], '.2f')
   catalog.write("%s %d %d %s %d %d \n"%(index,mm,nn,sigma_value,y_off,x_off))

   if x_off < 0 and y_off < 0:
      x_off = str(abs(x_off)).zfill(2)
      y_off = str(abs(y_off)).zfill(2)
      filename=f'{noiseFile}_{sigma_value}_n{y_off}_n{x_off}.png'
   elif x_off >= 0 and y_off >= 0:
      x_off = str(x_off).zfill(2)
      y_off = str(y_off).zfill(2)
      filename=f'{noiseFile}_{sigma_value}_p{y_off}_p{x_off}.png'
   elif x_off >= 0 and y_off < 0:
      x_off = str(x_off).zfill(2)
      y_off = str(abs(y_off)).zfill(2)
      filename=f'{noiseFile}_{sigma_value}_n{y_off}_p{x_off}.png'
   elif x_off < 0 and y_off >= 0:
      x_off = str(abs(x_off)).zfill(2)
      y_off = str(y_off).zfill(2)
      filename=f'{noiseFile}_{sigma_value}_p{y_off}_n{x_off}.png'

      

   os.rename(fname, filename)
  
   rotate_im.save(filename)

   
catalog.close()
            

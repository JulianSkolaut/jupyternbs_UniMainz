#!/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import pickle
import skimage
from skimage.io import imsave
from scipy.optimize import curve_fit


def readjust_contrast(image):
  """
takes in an image; returns it with pixel values between 0 and 1
  """
  image = image - np.min(image)
  return image / np.max(image)

def raveled_polynomial_image(xydata, *args, xovery=1):
  """
takes in flattend xy mesh; unflattens it; runs polynomial_image on it; returns
a flattend image

can be used for optimization
  """
  x, y = xydata
  x = x.reshape((int((len(x)*xovery)**.5), int((len(x)/xovery)**.5)))
  y = y.reshape((int((len(y)*xovery)**.5), int((len(y)/xovery)**.5)))
  return polynomial_image((x, y), *args).ravel()
  

def polynomial_image(xy, *args):
  """
takes in an xy mesh and polynomial coefficients 'args' and returns a polynomial
background of degree 'len(args)**.5' according to
http://gwyddion.net/documentation/user-guide-en/leveling-and-background.html#polynomial-level
  """
  X, Y = xy
  image = np.zeros(np.shape(X))
  n = int(len(args)**.5)
  a = []
  for i in range(n):
    a += [[*args[i*n:(i+1)*n]]]
  for j in range(n):
    for k in range(n):
      image += X**j * Y**k * a[j][k]
  return image

def get_polynomial_background(image, n=3, small_size=512):
  """
takes in an image; scales it down to a size 'small_size' x >'small_size' with a
fixed ratio; fits a polynomia background onto the image using
scipy.optimize.curve_fit for optimization and polynomial_image for creating the
background image; then returns a background of the full scale using the afore
determined fit parameters.

make sure contrast is readjusted before subtracting
  """
  # resizing ##################################################################
  # get the largest dimension and determine the scale factor
  #image = skimage.exposure.rescale_intensity(image)
  image = readjust_contrast(image)
  scale = np.max(np.shape(image)) / small_size
  small = skimage.transform.resize(image, (np.shape(image)[0]//scale,
                                           np.shape(image)[1]//scale))
  # readjust contrast to be save
  small = skimage.exposure.rescale_intensity(small)

  # background fitting ########################################################
  a = np.zeros(int((n+1)**2))
  x = np.arange(np.shape(small)[0])
  y = np.arange(np.shape(small)[1])
  x, y = np.meshgrid(x, y)
  xovery = np.shape(small)[0]/np.shape(small)[1]
  para, pcov = curve_fit(
    lambda xy, *args: raveled_polynomial_image(xy, *args, xovery=xovery),
    (x.ravel(), y.ravel()), small.ravel(), p0=a)
  # returning #################################################################
  x = np.arange(np.shape(image)[0])/scale
  y = np.arange(np.shape(image)[1])/scale
  x, y = np.meshgrid(x, y)
  return(polynomial_image((x,y), *para))

def remove_polynomial_background(image, n=3, small_size=512):
  image = readjust_contrast(image)
  bg = get_polynomial_background(image, n, small_size)
  new_image = image - bg
  return readjust_contrast(new_image)

# stack = None
# with open('360dw_drift_corrected_stack.pickle', 'rb') as f:
#   stack = pickle.load(f)

# image = stack[0]
# image = readjust_contrast(image)

# new_image = remove_polynomial_background(image, n=3, small_size=512)

# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(image)
# ax[1].imshow(new_image)
# plt.show()

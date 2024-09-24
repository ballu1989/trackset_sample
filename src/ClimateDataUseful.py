#=============================================================
#                       INFORMATION
#=============================================================
"""

  Set of class to read and manipulate climate data
  initially ERA5 and CESM

  It also includes the calling class that provides
  standardised output whatever is the climate data used

--------------------------------------------------------------
"""
__author__      = "Nicolas Bruneau"
__copyright__   = "Copyright 2021, Reask"
__version__     = "Trackset V22"

#=============================================================
#                   LOAD STANDARD MODULES
#=============================================================

import os, sys, operator, functools
import datetime, time, glob
import xesmf  as xe
import xarray as xr
#from   netCDF4 import Dataset, num2date
import matplotlib.pyplot as plt
import pandas as pd
xr.set_options(keep_attrs=True)
import itertools
from   abc import ABCMeta, abstractmethod
from   scipy import ndimage as nd
import numpy as np

#=============================================================
#               MANIPULATE SHAPE OF ARRAY    
#=============================================================

dict_month_name = {1:'Jan', 2:'Feb', 3:'Mar',  4:'Apr',  5:'May',  6:'Jun', \
                   7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec' }

def multDim( datadim ) :
    'Multiply elements of a tuple / list'
    return functools.reduce( operator.mul, datadim, 1 )

def FlattenArray( data, nkeep=1 ) :
    'Convert any array size into a flatten over other dimension, ...'
    datadim = data.shape     ## dimension data
    lendim  = len(datadim)   ## number of dimension
    return np.reshape( data, datadim[0:nkeep]+(multDim(datadim[nkeep::]),) )

def FullArray( data, dim ) :
    'Convert any flatten array back to its original size'
    return np.reshape( data, dim )

def Check( data, nkeep=1 ) :
    if len(data.shape) <= 1 : raise ValueError( 'Data needs to have at least 2 dimensions' )
    if len(data.shape) >= 3 : dataV = FlattenArray( data, nkeep=nkeep )
    return dataV


#=============================================================
#                         SCALER
#=============================================================

class SCALER( object ) :

  def __init__( self, Data = None, nKeepAxis=1 ) :
      '''
      Remove mean and std normalisation
      '''
      ## Checks
      if Data is None         : raise ValueError( 'Data field is required' )
      Data = np.array(Data); 
      self.dim = Data.shape;
      Data = Check( Data, nkeep=nKeepAxis )
      self.dimstatic = self.dim[nKeepAxis::]
      self.nKeepAxis = nKeepAxis

      ## Set-Up Scaling
      self._mean_ = np.nanmean( Data, axis=tuple(range(nKeepAxis)) )
      self._std_  = np.nanstd ( Data, axis=tuple(range(nKeepAxis)) )
      self._std_[(self._std_==0)] = 1.

      ## WHERE SCALING IS NAN
      ind = np.isnan(self._std_)
      self._std_ [ind] = 1.
      self._mean_[ind] = 0.

  def Decode( self, data ) :
      'PCA to Estimated Data'
      dataV = Check( data, nkeep=self.nKeepAxis )
      dum = dataV * self._std_ + self._mean_
      return np.reshape( dum, data.shape[0:self.nKeepAxis]+self.dimstatic )

  def Encode( self, data ) :
      'Raw Data to Scaled Data per features (second dimension)'
      dataV = Check( data, nkeep=self.nKeepAxis )
      dum = (dataV-self._mean_) / self._std_
      return np.reshape( dum, data.shape[0:self.nKeepAxis]+self.dimstatic )








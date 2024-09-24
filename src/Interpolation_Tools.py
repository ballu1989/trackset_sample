#=============================================================
#                       INFORMATION
#=============================================================
"""

  Deal with the climate data

--------------------------------------------------------------
"""
__author__      = "Nicolas Bruneau"
__copyright__   = "Copyright 2021, Reask"
__className__   = "Class_Climate.py"

#=============================================================
#                   LOAD STANDARD MODULES
#=============================================================

import sys,time
import numpy as np
from abc import ABCMeta, abstractmethod
from src.Useful import multDim, FlattenArray, FlattenArray_KeepLast, haversine
import xarray as xr
import bisect
from scipy.interpolate import RegularGridInterpolator
import scipy.ndimage as ndimage
from src.Fctn_Gaussian_Filter import *
 
#=============================================================
#               CESM HYBRID VERTICAL INTERPOLATION 
#=============================================================

import Ngl

def Vertical_Interp( dvar, var="U", levels=[200,850] ) :

    ## stretching and interpolation parameter
    hyam  = dvar["hyam"][:].values                   ## get stretching properties
    hybm  = dvar["hybm"][:].values                   ## get stretching properties
    P0mb  = 0.01*dvar["P0"].values                   ## convert from Pa to milliBar
    intyp = 1                                        ## interpolation type: 1=linear, 2=log, 3=log-log
    kxtrp = False                                    ## extrapolation out of pressure level

    ## case where you don't aggregate files and attributes variables
    ## are one dim and not 2 dimension with repetition
    special=False
    if len(hyam.shape) == 1 :
       special=True
       P0mb = np.array([P0mb])
       hyam = hyam[np.newaxis,:]
       hybm = hybm[np.newaxis,:]

    ## check hyam, etc are constant along time
    dha = np.abs( hyam - hyam[0,:] )
    dhb = np.abs( hybm - hybm[0,:] )
    dP0 = np.diff( P0mb )
    if special : dP0 = 0
    if np.max(dha) > 0 or np.max(dhb) > 0 or np.max(dP0) > 0 :
       print("hyam or hybm is time variable. stop"); sys.exit()

    ## Interpolate the ensemble member
    t0   = time.time()
    Tnew = Ngl.vinth2p( dvar[var], hyam[0] , hybm[0],\
            levels, dvar.PS, intyp, P0mb[0], 1, kxtrp )
    Tnew[Tnew==1e30] = np.nan             ## fill gaps

    ## store in new xarray
    dst = xr.Dataset( { 'time' : ( [ 'time' ], dvar.time.values ), \
                        'lat'  : ( [ 'lat'  ], dvar.lat.values  ), \
                        'lon'  : ( [ 'lon'  ], dvar.lon.values  ), \
                        'level': ( [ 'level'], levels,  ), \
                         var   : ( [ 'time', 'level', 'lat', 'lon' ], Tnew ) } )

    ## copy attrivutes
    for iVar in [  var, ] :
        for iatt in dvar[iVar].attrs.keys() :
            dst[iVar].attrs[iatt] = dvar[iVar].attrs[iatt]

    return dst

#=============================================================
#               LOAD OTHER DATA (TOPO AND COAST)
#=============================================================

class DATA2D_to_INTERPOLATE( object ) :

      def __init__( self, MYXR=None, PATH_DATA=None, var=None, method='linear', fillvalue=np.nan, shiftdays=0, \
                    daily=False, extension = None, JD=True, periodic=False ) :

          ## Load data 
          if MYXR is not None : self.data = MYXR.copy()
          else : self.data = xr.open_mfdataset( "{0}".format(PATH_DATA) )

          ## shift day (useful to recentre monthly data)
          if shiftdays != 0 : self.data['time'] = self.data.time + np.timedelta64(shiftdays,"D")

          ## rename lat lon
          try : self.data = self.data.rename( {'longitude':'lon','latitude':'lat'} )
          except: pass
          ## daily average
          if daily : self.data = self.data.resample(time='1D').mean()
          ## subselect
          if extension is not None :
             self.data = self.data.sel( lon=slice(extension[0],extension[1]) )
             if self.data.lat[0] > self.data.lat[-1] :
                print( extension[3],extension[2] )
                self.data = self.data.sel( lat=slice(extension[3],extension[2]) )   ## era5 at are reverse
             else : self.data = self.data.sel( lat=slice(extension[2],extension[3]) )
          ## reverse lat if needed
          if self.data.lat[0] > self.data.lat[-1] :
             self.data = self.data.reindex(lat=list(reversed(self.data.lat)))
          ## load data in memory
          self.data = self.data.load() ## put data in memory for faster access    
          self.lon  = self.data.lon.values; self.dlon = np.abs(np.mean(np.diff(self.lon)))
          self.lat  = self.data.lat.values; self.dlat = np.abs(np.mean(np.diff(self.lat)))
          self.var  = self.data[var].values 
          self.meth = method
          ## add a lon if periodic
          if periodic :
             self.lon  = np.append( [self.lon[-1]-360.,], np.append( self.lon, [self.lon[0]+360.,] ) )
             self.var  = np.append( self.var[...,-1][...,np.newaxis], np.append( self.var, self.var[...,0][...,np.newaxis], axis=-1 ), axis=-1 )
          ## look if time variable
          if JD : self.JD = 1
          else  : self.JD = 0
          try    : 
              self.Tref = self.data.time[0].values
              #print( "REFERNCE interpolation", self.Tref )
              self.time = ((self.data.time-self.Tref) / np.timedelta64(1,'D')).values + self.JD
          except : self.time = None
          if self.lat[-1] < self.lat[0] : print("Climate_Data class not implemented for latitude going form N to S" ); sys.exit()
          self.fillvalue = fillvalue
          self.Create_Interpolator()

      def Create_Interpolator( self ) :
          if self.time is None : points = ( self.lat, self.lon )
          else : points = ( self.time, self.lat, self.lon )
          #print( self.var.shape, self.lon.shape,self.lon)
          interp = RegularGridInterpolator( points, self.var, method = self.meth,
                                            bounds_error = False, fill_value = self.fillvalue )
          self.interpolator = interp

      def get_values( self, time=None, lon=None, lat=None ) :
          '''extract values in a quick (not fully precise way)'''
          if time is None : newpoints = np.array( [lat, lon] ).T
          else : newpoints = np.array( [time, lat, lon]).T
          return self.interpolator( newpoints ).T

      def get_average_around( self, time=None, lon=None, lat=None, dx=1 ) :
          store = []
          Latspace = np.arange( -dx, dx+0.00001, self.dlat )
          Lonspace = np.arange( -dx, dx+0.00001, self.dlon ) 
          #print( "WHICH SPACE YOU SCAN", self.dlat, self.dlon, Lonspace.shape, Latspace.shape )
          for jLat in Latspace :
              for iLon in Lonspace :
                  if time is None : newpoints = np.array( [      lat+jLat, lon+iLon]).T
                  else :            newpoints = np.array( [time, lat+jLat, lon+iLon]).T
                  store.append( self.interpolator( newpoints ).T )
          store = np.array(store)
          return( np.nanmean(store, axis=0) )


#=============================================================
#              CLIMATE DATA 
#=============================================================

class Climate_Data( object ) :

   var2load = [ "u", "v", "sst" ]  

   def __init__( self, PATH_DATA, season, method='linear', smooth=0, fillvalue=np.nan ) : 

       ## Load all data - assuming they are on same grid after preprocessing
       self.data = xr.open_mfdataset( "{0}/DAILY_Season_{1}_*.nc".format(PATH_DATA,season) )
       self.data = self.data.load() ## put data in memory for faster access   
       self.lon  = self.data.lon.values
       self.lat  = self.data.lat.values
       self.time = self.data.time.values
       self.levs = self.data.level.values
       self.smooth = smooth
       self.meth = method
       self.fillvalue = fillvalue
       if self.lat[-1] < self.lat[0] : print("Climate_Data class not implemented for latitude going form N to S" ); sys.exit() 
       self.Create_Interpolator()

   def Create_Interpolator( self ) :
       '''create the interpolator once and then it could be called as many time as we want'''
       self.interpoltors = {}; self.varnames = []
       points = ( np.arange(self.time.size), self.lat, self.lon  )
       for iVar in [ 'u', 'v', 'sst' ] :
           ndim = len( self.data[ iVar ].shape )
           if ndim == 4 : nlev = 2; levs = [ 200, 850, ]
           else         : nlev = 1; levs = [ None, ]
           for ilev in range(nlev) :
               if levs[ilev] is None : dum = self.data[iVar].values; mylab = "weather_{0}".format(iVar)
               else : dum = self.data.sel( level=levs[ilev] )[iVar].values; mylab = "weather_{0}_{1}hPa".format(iVar,levs[ilev])
               #plt.imshow( dum[10,::-1] ); plt.show()
               ## smooth data
               #if self.smooth > 0 : dum = ndimage.gaussian_filter( dum, sigma=(0,self.smooth,self.smooth), order=0, mode="nearest" )
               if self.smooth > 0 : dum = filter_nan_gaussian_conserving( dum, self.smooth ) 
               #plt.imshow( dum[10,::-1] ); plt.show()
               interp = RegularGridInterpolator( points, dum, method=self.meth,
                                                 bounds_error = False, fill_value = self.fillvalue )
               self.interpoltors[mylab] = interp
               self.varnames.append( mylab )

   def get_values( self, time, lon, lat ) :
       '''extract values in a quick (not fully precise way)'''
       newpoints = np.array( [time, lat, lon]).T
       outputs   = np.zeros( [newpoints.shape[0],len(self.varnames)] )
       for iV, varname in enumerate(self.varnames) : 
           outputs[:,iV] = self.interpoltors[varname]( newpoints )
       outputs[np.isnan(outputs)] = 0.
       return outputs 


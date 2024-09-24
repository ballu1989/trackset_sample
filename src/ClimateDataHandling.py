#=============================================================
#                       INFORMATION
#=============================================================
"""

  Set of class to read and manipulate climate data
  initially ERA5 and CESM

  It also includes the calling class that provides
  standardised output whatever is the climate data used


  SOME WEIRD PATTERN WITH REGRIDDER AND PERIODIC
  LIKE REVERSING THE LATITUDE MAKES INTERPOLATION WRONG...
  DIDN@T FIND A CLEAR PATTERN


--------------------------------------------------------------
"""
__author__      = "Nicolas Bruneau"
__copyright__   = "Copyright 2021, Reask"
__version__     = "Trackset V22"

#=============================================================
#                   LOAD STANDARD MODULES
#=============================================================

import os, sys, datetime, time, glob
import xesmf  as xe
import xarray as xr
#from   netCDF4 import Dataset, num2date
import matplotlib.pyplot as plt
from   src.Useful import *
import pandas as pd
xr.set_options(keep_attrs=True)
import itertools
from   abc import ABCMeta, abstractmethod
from   scipy import ndimage as nd
from skimage.restoration import inpaint
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
import cv2
from pandas.tseries.offsets import DateOffset

#=============================================================
#                      INTERNAL FCTNS
#=============================================================

def open_xarray( mypath, renameVars=None, concat_dim=None, chunk=None, engine=None ) :
    'open the files as xarray - easier to manipulate'
    if "*" in mypath or isinstance( mypath, list):
           da = xr.open_mfdataset( mypath, combine='nested', concat_dim=concat_dim, chunks=chunk, engine=engine )
    else : da = xr.open_dataset  ( mypath, chunks=chunk, engine=engine ) 
    ## check unique time
    _, index = np.unique( da['time'], return_index=True)
    try : da = da.isel( time=index )
    except : pass
    try : 
      if renameVars is not None : da = da.rename( renameVars )
    except: pass
    return da

def xr_get_mask( da, var2D, axis=['time',]  ) :
    'from a variable [ time, lat, lon], evaluate the [lat,lon] mask)'
    da[ "mask" ] = xr.where( ~np.isnan(da[var2D].mean(dim=axis,skipna=False)), 1, 0)
    return da 
     
def np_get_mask( data, axis=0 ): 
    '''0 where NaN, 1 otherwise'''
    mask = np.mean( data, axis=axis )
    return np.squeeze( (~np.isnan( mask )).astype(np.int) )

def np_set_mask( data, mask ) :
    dmask = mask.copy().astype(np.float); 
    dmask[mask==0] = np.nan
    data  = data*dmask
    return data

def fill_NaN( Data, indNaN=None, NN=True, inpainting=None ) :
    '''
    Fill mask with NN and apply another mask
      inpainting could be cv2.INPAINT_TELEA, cv2.INPAINT_NS
    '''
    dum = np.copy( Data ); myDIM  = dum.shape
    for idx in itertools.product( *[range(s) for s in myDIM[0:-2] ] ):
        test = np.copy(Data[idx].astype(np.float32)); mask = np.isnan(test)
        if np.sum(mask) > 0 : 
           ## first inpaint
           if inpainting is not None : 
              test = cv2.inpaint( test, mask.astype(np.uint8), 3, inpainting )
              mask = np.isnan(test) ## update before the NN next level filling
           ## second file corner gaps (can be large in antarctica) with NN     
           if NN :
              indices = nd.distance_transform_edt(mask, return_distances=False, return_indices=True)
              test = test[ tuple(indices) ]
        if indNaN is not None : test[indNaN==0] = np.nan
        dum[idx] =  test
    return dum

def Get_hPaLevel( zlevel, zpress_lev=850 ) :
    'Index of nearest wanted level'
    return np.argmin( np.abs(zpress_lev-zlevel) ) 

def subselect_da( da, LonMin, LonMax, LatMin, LatMax, shiftLong ) :
    lat0 = da.lat.values[0]; lat1 = da.lat.values[-1]
    if lat0 < lat1 : CresLatDir = True
    else           : CresLatDir= False
    if da.lon.values[0] >= 0 :
       if shiftLong : da = da.assign_coords( lon=(((da.lon + 180)% 360)-180)).roll(lon=(da.dims['lon'] // 2), roll_coords=True )
    da = da.sel( lon = slice( LonMin, LonMax ) )
    if CresLatDir : da = da.sel( lat = slice( LatMin, LatMax ) )   ## latitude starts from northPole
    else :          da = da.sel( lat = slice( LatMax, LatMin ) )
    return da, CresLatDir

def Downscaling_Data( da, aggreg_mode, Lon_dst, Lat_dst, method="bilinear", fout=None, varplot=None ):
    'change resolution by interpolation of binning-average'
    ## interpolation 
    if aggreg_mode == "interpolation" :
       ## Create destination framework    
       dst = xr.Dataset( { 'lat'  : ( ['lat' ], Lat_dst ), \
                           'lon'  : ( ['lon' ], Lon_dst ), \
                         } )
       ## periodic conditions
       if da.lon.max()-da.lon.min() > 358 : myper = True; strper = '_periodic'
       else: myper = False; strper = ''
       
       ## create interpolator
       if fout is not None : 
          if   len(da.lon.shape) == 1 : nLon = da.lon.size;     nLat = da.lat.size
          elif len(da.lon.shape) == 2 : nLon = da.lon.shape[1]; nLat = da.lat.shape[0]
          else : ErrorDisplay( "Issue in Downscaling_Data. Lon / Lat needs to be 1D or 2D" ) 
          mymeth = method
          #if extrap_method is not None:mymeth+="_{0}".format(extrap_method)
          fout = "{0}_{1}_from_{2}x{3}_to_{4}x{5}{6}_Lon_{7:.2f}E_{8:.2f}E_Lat_{9:.2f}N_{10:.2f}N.nc".format( fout, mymeth, \
                                         nLon, nLat, Lon_dst.size, Lat_dst.size, strper,\
                                         Lon_dst[0],Lon_dst[-1], Lat_dst[0],Lat_dst[-1],)

       ## create regridder
       try :    regridder = xe.Regridder( da, dst, method=method, periodic=myper, reuse_weights=True , filename=fout )
       except : regridder = xe.Regridder( da, dst, method=method, periodic=myper, reuse_weights=False, filename=fout )
       ## interpolate
       dst = regridder(da)

       ## check figure
       if varplot is not None : 
          dum = len(dst[varplot].values.shape)
          if   dum==2 : plt.imshow( dst[varplot].values );
          elif dum==3 : plt.imshow( dst[varplot].values[0] ); 
          elif dum==4 : plt.imshow( dst[varplot].values[0,0] );
          plt.colorbar(); 
          if fout is not None : plt.savefig( fout.replace(".nc",".png"), dpi=200 )
          else : plt.show()
          plt.close()

    ## or aggregation
    elif aggreg_mode == "aggregation" :
         if np.min(Lon_dst) < da.lon.min() : ErrorDisplay( "EDGE WARNING: LON is outside the range of dataframe ({0:.2f} < {1:.2f})". format( np.min(Lon_dst), da.lon.min().values ))
         if np.max(Lon_dst) > da.lon.max() : ErrorDisplay( "EDGE WARNING: LON is outside the range of dataframe ({0:.2f} > {1:.2f})". format( np.max(Lon_dst), da.lon.max().values ))
         ## bin data and average instead of interpolating
         print(da)
         a = da.groupby_bins('lat', Lat_dst ).mean(skipna=True) 
         b =  a.groupby_bins('lon', Lon_dst ).mean(skipna=True).transpose()
         b =  b.assign_coords( { "lon_bins": (Lon_dst[1::]+Lon_dst[0:-1])/2, \
                                 "lat_bins": (Lat_dst[1::]+Lat_dst[0:-1])/2, } )
         dst = b.rename( name_dict={"lon_bins":"lon","lat_bins":"lat"} )
         if "level" not in dst.dims : 
             if "ensemble" in dst.dims : dst = dst.transpose( "time", "ensemble", "lat", "lon" )
             else :  dst = dst.transpose( "time","lat", "lon" )
         else : dst = dst.transpose( "time", "ensemble", "level", "lat", "lon" )

    ## otherwise
    else : ErrorDisplay( "interpolation or aggregation only available for aggreg_mode (given {0})".format(aggreg_mode) )
    dst = dst.assign_attrs( reask_downscaling_mode = aggreg_mode )

    return dst


def Get_Most_Accurate_CellArea( da ) :
    dLon = np.mean(np.diff( da.lon.values ) )
    dLat = np.mean(np.diff( da.lat.values ) )
    print( "  - CELL AREA [deg]: dLon, dLat", dLon, dLat )
    area = np.zeros( [da.lat.size, da.lon.size] )
    ## in latitude it's constant whatever longitude
    dy = np.zeros( [da.lat.size, da.lon.size])
    for iLat in range( da.lat.size-1 ) :
        for iLon in range( da.lon.size ) : 
            dy[iLat,iLon] = haversine( da.lon.values[ iLon ], da.lat.values[ iLat   ], \
                                       da.lon.values[ iLon ], da.lat.values[ iLat+1 ]  )

    dx = np.zeros( [da.lat.size,da.lon.size])     
    for iLat in range( da.lat.size ) :
        for iLon in range( da.lon.size-1 ) :
            dx[iLat, iLon] = haversine( da.lon.values[ iLon   ], da.lat.values[ iLat ], \
                                        da.lon.values[ iLon+1 ], da.lat.values[ iLat ]  )
    ## boundary
    dx[:,-1] = dx[:,-2]
    dy[-1,:] = dy[-2,:]       
    print( "  - CELL AREA [ km]: dLon", dx.min(), dx.max() )
    print( "  - CELL AREA [ km]: dLat", dy.min(), dy.max() )
    return dx, dy 


def extend_bnd( dum ) :
    dum[..., 0,:] = dum[..., 1,:]
    dum[...,-1,:] = dum[...,-2,:]
    dum[...,:, 0] = dum[...,:, 1]
    dum[...,:,-1] = dum[...,:,-2]
    return dum

def Compute_Spatial_derivatives( data, list_var, dx=None, dy=None ) :
    if dx is None or dy is None : dx, dy = Get_Most_Accurate_CellArea( data )
    for i in list_var :
        dum = data[i]
        gradx = np.zeros_like( dum.values ) + np.nan
        gradx[...,1:-1,1:-1] = ( dum.values[...,1:-1,2::] - dum.values[...,1:-1,0:-2] ) / ( dx[1:-1,0:-2]+dx[1:-1,1:-1] )
        gradx = extend_bnd(gradx)
        grady = np.zeros_like( dum.values ) + np.nan
        grady[...,1:-1,1:-1] = ( dum.values[...,2::,1:-1] - dum.values[...,0:-2,1:-1] ) / ( dy[0:-2,1:-1]+dy[1:-1,1:-1] )
        grady = extend_bnd(grady)

        ## add to structure
        data = data.assign( gradx = (("time","lat","lon",),gradx) )
        data = data.assign( grady = (("time","lat","lon",),grady) )
        data = data.rename( { "gradx":"gradx_{0}".format(i), "grady":"grady_{0}".format(i)} )
    return data 

#=============================================================
#                    GENERAL CLIMATE DATA
#=============================================================

class CLIMATEdata( object ):

  mapping   = None 
  renameVar = None

  def __init__( self, PATH_TO_DATA=None, VAR_TO_LOAD=None, ENS_LIST=None, NAMESTR=None, MONTH=None, FILLNA = False ) :
        ''' init variables and checks'''

        ## Set Path to data
        if PATH_TO_DATA is None : ErrorDisplay( "PATH_TO_DATA is required" )
        else : self.pdata = PATH_TO_DATA
        self.VAR_TO_LOAD = VAR_TO_LOAD.copy()
        self.NAMESTR = NAMESTR
        self.MONTH   = MONTH
        self.FILLNA  = FILLNA

        ## Set variables to load
        if self.mapping  is None : ErrorDisplay( "mapping variable needs to be defined in the class" )
        elif self.VAR_TO_LOAD is None : self.variables = self.mapping.ID.values; self.varout = self.variables.copy()
        else :
            self.varout    = np.squeeze( np.array(self.VAR_TO_LOAD).copy() )
            ## Handle shear
            if  "shr"  in self.VAR_TO_LOAD : self.VAR_TO_LOAD.append( "u" ); self.VAR_TO_LOAD.append( "v" ); self.VAR_TO_LOAD.remove( "shr" )
            if "ushr"  in self.VAR_TO_LOAD : self.VAR_TO_LOAD.append( "u" ); self.VAR_TO_LOAD.remove( "ushr" )
            if "vshr"  in self.VAR_TO_LOAD : self.VAR_TO_LOAD.append( "v" ); self.VAR_TO_LOAD.remove( "vshr" )
            if "vorti" in self.VAR_TO_LOAD : self.VAR_TO_LOAD.append( "u" ); self.VAR_TO_LOAD.append( "v" ); self.VAR_TO_LOAD.remove( "vorti" )
            if "u_steering" in self.VAR_TO_LOAD : self.VAR_TO_LOAD.append( "u" ); self.VAR_TO_LOAD.remove( "u_steering" )
            self.variables = np.array( list(set(self.VAR_TO_LOAD)) ) 
        print( " - LOADING", self.variables, "TO GET", self.varout ) 
        
        ## Load Xarra
        self.ENS_LIST = ENS_LIST
        self.Load_Data()
        if "shr" in self.varout or "ushr" in self.varout or "vshr" in self.varout :
            self.Compute_Shear()
        if "vorti" in self.varout :
            self.Compute_Vorticity()
        if "u_steering" in self.varout or "v_steering" in self.varout :
            self.Compute_Steering()

  @abstractmethod
  def Load_Data( self ) :
      'Define the Data you want to load depending of the set used'
      pass

  def Compute_Steering( self ) :
      uuu = self.datas["u"].copy()
      indu850 = Get_hPaLevel( uuu.level.values, zpress_lev=850 )
      indu200 = Get_hPaLevel( uuu.level.values, zpress_lev=200 )
      uuu['u_steering'] = 0.2 * uuu.u[...,indu200,:,:] + 0.8 * uuu.u[...,indu850,:,:]
      uuu = uuu.drop( ['u',"level"] )

      ## store in dictionnary
      self.datas['u_steering'] = uuu


  def Compute_Shear( self ) :
      'Compute Wind shear'
      uuu = self.datas["u"].copy()
      indu850 = Get_hPaLevel( uuu.level.values, zpress_lev=850 )
      indu200 = Get_hPaLevel( uuu.level.values, zpress_lev=200 )
      uuu['ushr'] = uuu.u[...,indu200,:,:] - uuu.u[...,indu850,:,:]

      vvv = self.datas["v"].copy()
      indv850 = Get_hPaLevel( uuu.level.values, zpress_lev=850 )
      indv200 = Get_hPaLevel( uuu.level.values, zpress_lev=200 )
      vvv['vshr'] = vvv.v[...,indv200,:,:] - vvv.v[...,indv850,:,:]

      ## ADD SHEAR
      uuu = uuu.assign( shr = np.sqrt( uuu.ushr**2+vvv.vshr**2) )
      uuu = uuu.assign( ushr = uuu.ushr )
      uuu = uuu.assign( vshr = vvv.vshr )
      uuu = uuu.drop( ['u',"level"] )
      
      ## store in dictionnary
      self.datas['shr'] = uuu
    
  def Compute_Vorticity( self ):

      uuu = self.datas["u"].copy()
      vvv = self.datas["v"]

      dx, dy = Get_Most_Accurate_CellArea( uuu )

      vorti  = np.zeros_like( uuu.u.values ) + np.nan
      vorti[...,1:-1,1:-1]  = ( vvv.v.values[...,1:-1,2::] - vvv.v.values[...,1:-1,0:-2] ) / ( dx[1:-1,0:-2]+dx[1:-1,1:-1] )
      vorti[...,1:-1,1:-1] -= ( uuu.u.values[...,2::,1:-1] - uuu.u.values[...,0:-2,1:-1] ) / ( dy[0:-2,1:-1]+dy[1:-1,1:-1] )
      
      ## boundary conditions (just simple propagation)
      vorti[..., 0,:] = vorti[..., 1,:]
      vorti[...,-1,:] = vorti[...,-2,:]
      vorti[...,:, 0] = vorti[...,:, 1]
      vorti[...,:,-1] = vorti[...,:,-2]

      ## add to structure
      uuu = uuu.assign( vorti = (("time","ensemble","level","lat","lon",),vorti) )
      uuu = uuu.drop( ['u',] )
      self.datas['vorti'] = uuu

  def Fill_Missing( self, fill_vars=[ "u", "v", "sst" ], mask_vars={"sst":None}, inpaint_Method=cv2.INPAINT_NS ):
      '''U and V field might have MISSING values over large mountains for some Level'''
      for iV in fill_vars :
          if iV in self.datas.keys() : ## only if needed
             if iV in mask_vars.keys() : mask = mask_vars[iV]
             else : mask=None
             dum = self.datas[iV][iV].values  #[0:3,0:2]
             t0 = time.time()
             tmp = fill_NaN( dum, indNaN=mask, inpainting=inpaint_Method )
             self.datas[iV][iV].values = tmp
             t1=time.time()
             print( "  -> Fill NaN", iV, (t1-t0)/60., "min")

  def Set_Mask( self, mask_vars={"sst":None} ) :
      for iV in mask_vars.keys() :
          tmp = np_set_mask( self.datas[iV][iV].values, mask=mask_vars[iV] )
          self.datas[iV][iV].values = tmp

#=============================================================
#                    ERA5 CLIMATE DATA
#=============================================================

class ERA5data( CLIMATEdata ) :

    mapping = pd.DataFrame( { "ID": [ "sst", "msl", "u", "v", "r", ], \
                        "NAME_VAR": [ "sst", "msl", "u", "v", "r", ], \
                        "FOLD_VAR": [ "sea_surface_temperature", "mean_sea_level_pressure", \
                                      "u_component_of_wind", "v_component_of_wind", "relative_humidity",  ] }   )
    renameVar = { 'latitude':'lat', 'longitude':'lon'}

    def Load_Data( self ) :
        self.ENS_LIST = ['RAN',]
        self.datas = {}
        for iV in self.variables :

            ## Manage variable name between set to be homogenous
            ind  = self.mapping['ID'] == iV
            var_read = self.mapping['NAME_VAR'][ind].values[0]
            dir_read = self.mapping['FOLD_VAR'][ind].values[0]

            ## add renaming if needed
            renameVars = self.renameVar
            if iV != var_read : renameVars[var_read] = iV

            ## open netcdf
            if self.NAMESTR is None : prename = "{0}/*_{1}.nc".format( self.pdata, dir_read )
            else : prename = "{0}/{1}*_{2}.nc".format( self.pdata, self.NAMESTR, dir_read )
            print( prename )
            da = open_xarray( prename, renameVars )
            #da = da.sel( time=slice(None,"2022-01-01"))
            print("whoooo")
            print(da)
            
            da = da.expand_dims( "ensemble", axis=1 )
            da = da.assign_coords( ensemble=self.ENS_LIST )

            ## store in dictionnary
            da = da.rename( { var_read: iV} )
            self.datas[iV] = da[ [iV,] ]


class C3S_FCTdata( CLIMATEdata ) :

    mapping = pd.DataFrame( { "ID": [ "sst", "msl", "u", "v", ], \
                        "NAME_VAR": [ "sst", "msl", "u", "v", ], \
                        "FOLD_VAR": [ "surface", "surface", "levels", "levels", ] } )

    renameVar = { 'latitude':'lat', 'longitude':'lon', "number": "ensemble", }#"step":"time"}
    def Load_Data( self ) :
        self.datas = {}
        for iV in self.variables :

            ## Manage variable name between set to be homogenous
            ind  = self.mapping['ID'] == iV
            var_read = self.mapping['NAME_VAR'][ind].values[0]
            dir_read = self.mapping['FOLD_VAR'][ind].values[0]

            ## add renaming if needed
            renameVars = self.renameVar
            if iV != var_read : renameVars[var_read] = iV

            ## open netcdf
            prename = "{0}/{2}_ecmwf_Exp5_{1}.grib".format( self.pdata, dir_read, self.MONTH )
            da = open_xarray( prename, renameVars, engine="cfgrib" )
            try : da = da.stack(z=("time", "step"))  ## flatten forecast time and leadtime
            except : print("cannot flatten along both time and step: probably forecast mode for current year")

            ## play with coordinates
            try    : da = da.rename_dims({"z":"time"}).reset_index("z").set_coords( "time" )
            except : da = da.drop(["time",]).rename_dims({"step":"time"}) 
            
            da = da.assign_coords( time=da.valid_time )
            da = da.drop( ["step","valid_time"] )
            a = pd.to_datetime(da.time) + DateOffset(months=-1) ## valid_time is the end of the integrated period and not the begining
            da = da.assign(time=a)

            ## reorder dimension
            if "isobaricInhPa" in da.dims : da = da.rename( {"isobaricInhPa":'level',} )
            if "level" not in da.dims : 
                da = da.transpose( "time", "ensemble", "lat", "lon" )
            else :
                da['level'] = da.level.astype(int)
                da = da.transpose( "time", "ensemble", "level", "lat", "lon" )

            ## rename ensemble
            da['ensemble'] = [ "MEMBER_{0:02d}".format(aa) for aa in da['ensemble'].values ]
            da = da.sel(ensemble=da.ensemble.isin(self.ENS_LIST) )

            ## store in dictionnary
            da = da.rename( { var_read: iV} )
            self.datas[iV] = da[ [iV,] ]


#=============================================================
#                  CESM LENS2 CLIMATE DATA
#=============================================================

class LENS2data( CLIMATEdata ) :

    mapping = pd.DataFrame( { "ID": [ "sst", "msl", "u", "v" ], \
                        "NAME_VAR": [ "SST", "PSL", "U", "V" ], }   )

    def Load_Data( self ) :
        if not isinstance(self.ENS_LIST, list): self.ENS_LIST = [ self.ENS_LIST, ]
        self.datas = {}
        for iV in self.variables :

            ## Manage variable name between set to be homogenous
            ind  = self.mapping['ID'] == iV
            var_read = self.mapping['NAME_VAR'][ind].values[0]

            ## select files and order properly the ensemble
            allfiles = sorted(glob.glob("{0}/ENS.*/{1}*.nc".format( self.pdata, var_read )))
            files2read = []
            for iENS in self.ENS_LIST :
                for thisfile in allfiles :
                    dum = thisfile.split( "_ens." )[-1].split( ".nc" )[0]
                    if dum == iENS : files2read.append( thisfile )

            ## open netcdf
            #print( files2read)
            da = open_xarray( files2read, renameVars=None, concat_dim=["ensemble",] )#, chunk={'lon':36,'lat':24} )
            da = da.assign_coords( ensemble=self.ENS_LIST)
            if "level" not in da.dims : da = da.transpose( "time", "ensemble", "lat", "lon" )
            else : da = da.transpose( "time", "ensemble", "level", "lat", "lon" ) 
            print( "-> LENS2 {0} ENSEMBLE LOADED".format( da.ensemble.size) )

            ## convert 365-day calendar to normal calendar
            help_date = np.vectorize( lambda x: np.datetime64(x) )
            da = da.assign_coords( time=help_date( da.time.values ) )
            da.time.attrs = {}
            da.time.attrs['long_name'] = 'time'

            ## store in dictionnary
            da = da.rename( { var_read: iV} )
            self.datas[iV] = da[ [iV,] ]

            ## add attributes
            da['lon'].attrs = {"units":"degrees_east", "long_name":"longitude"} 
            da['lat'].attrs = {"units":"degrees_north", "long_name":"latitude"}


#=============================================================
#               PRE-PROCESSED CLIMATE DATA
#=============================================================

class PREPROCdata( CLIMATEdata ) :

    mapping = pd.DataFrame( { "ID": [ "sst", "msl", "u", "v" ], }   )

    def Load_Data( self ) :
        try    : self.ENS_LIST
        except : self.ENS_LIST = ['RAN',]
        self.datas = {}
        for iV in self.variables :

            ## Manage variable name between set to be homogenous
            ind  = self.mapping['ID'] == iV
            var_read = self.mapping['ID'][ind].values[0]

            ## open netcdf
            if self.NAMESTR is None : prename = "{0}/*_{1}.nc".format( self.pdata, var_read )
            else : prename = "{0}/{1}*_{2}.nc".format( self.pdata, self.NAMESTR, var_read )
            print( prename )
            da = open_xarray( prename, renameVars=None )

            ## store in dictionnary
            self.datas[iV] = da[ [iV,] ]


#=============================================================
#                    GENERAL CLIMATE DATA
#=============================================================

class ZARRdata( object ):

  def __init__( self, PATH_TO_DATA=None, VAR_TO_LOAD=None, LEVELS=[200,850], ENSEMBLE=None, FILLNA=False  ) :
        ''' init variables and checks'''

        ## Set Path to data
        if PATH_TO_DATA is None : ErrorDisplay( "PATH_TO_DATA is required" )
        else : self.pdata = PATH_TO_DATA
        self.variables = VAR_TO_LOAD.copy()
        self.levels = LEVELS
        self.FILLNA = FILLNA

        ## Load data
        self.datas = {}
        for iV in self.variables :
            if LEVELS is not None : dir_var = iV.split("_")[0]
            else : dir_var = iV
            print( "{0}/{1}.zarr".format(self.pdata,dir_var) )
            da = xr.open_zarr( "{0}/{1}.zarr".format(self.pdata,dir_var) )
            if ENSEMBLE is not None : 
               da = da.sel( ensemble=da.ensemble.isin(ENSEMBLE) )
            if iV in [ "u", "v", ] :
               for iL in self.levels :
                   varname = "{0}_{1}hpa".format(iV,iL)
                   print(da)
                   self.datas[varname] = da[ [varname,] ]
            else : self.datas[iV] = da[ [iV,] ] 
        for i in self.datas.keys():
            ## put nan in member that did not exist
            if self.FILLNA :
               da = self.datas[i] 
               da = da.where( (da["ensemble"].isin([ "MEMBER_{0:02d}".format(i) for i in range(25)]))&(da['time.year']<2017) | (da['time.year']>=2017)  , other = np.nan ) 
               self.datas[i] = da


## DEVELOP A CLAS TO BRING ZARRdata object to CLIMATEdata object
## mostly developped for the CS3 data where after filling gap in
## ZARR database, the shape are slightly different from reading 
## directly grib or nc
class ZARR2CLIMATEdata( CLIMATEdata ) :
    mapping ={}; Lon0=25 
    def Load_Data( self ) :
        thisvar = []; levels = [200,850] 
        for iV in self.variables :
            if iV in [ "u", "v", ] :
               for iL in levels : 
                   thisvar.append( "{0}_{1}hpa".format( iV, iL ) )
            else : thisvar.append( iV )
        climdata = ZARRdata( PATH_TO_DATA = "{0}_{1}/".format( self.pdata, self.MONTH), VAR_TO_LOAD=thisvar, LEVELS=None, ENSEMBLE=self.ENS_LIST, FILLNA=self.FILLNA )

        ## concat data diff level
        for iV in self.variables :
            if iV in [ "u", "v", ] : 
               toconcat = []
               for iL in levels :
                   varname = "{0}_{1}hpa".format( iV, iL )
                   toconcat.append( climdata.datas[varname].rename({varname:iV}) )
               da = xr.concat( toconcat, dim="level" ) 
               da = da.assign_coords( level=levels )
               da = da.transpose( "time", "ensemble", "level", "lat", "lon" ) 

               climdata.datas[ iV ] = da.chunk( chunks="auto" )
               for iL in levels : 
                   varname = "{0}_{1}hpa".format( iV, iL )
                   del climdata.datas[varname]

        for i in climdata.datas.keys():
            da = climdata.datas[i]
            da = da.isel( lon = slice(0,-1) )
            da = da.assign_coords( lon=da.lon.where(da.lon<360,da.lon-360)).roll(lon=np.sum(da.lon.values>=360), roll_coords=True )
            ## put nan in member that did not exist
            #if self.FILLNA :
            #   da = da.where( (da["ensemble"].isin([ "MEMBER_{0:02d}".format(i) for i in range(25)]))&(da['time.year']<2017) | (da['time.year']>=2017)  , other = np.nan )
            #   da = da.assign_ccords( ENSNUM = [ int( str(i).split("_")[1]) for i in da["ensemble"].values] ) 
            #   da = da.where( (da["ENSNUM"]<25)&(da.time==slice(None,"2016-12-31")) | (da.time==slice("2016-12-31", None)), other = np.nan  )
            climdata.datas[i] = da

        ## keep in memory
        self.datas = climdata.datas

        







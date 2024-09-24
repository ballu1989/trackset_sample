import numpy as np
from scipy.interpolate import interp2d
import scipy.spatial as spatial
from src.Useful import *
import matplotlib.pyplot as plt
import xesmf  as xe
import xarray as xr
from src.ClimateDataHandling import *

##=================================================================================
##  NEW VERSION FOR GRID DEFINITION
##  use pre-generated zarr coastline
##  mainly used to bin spatially data now
##  periodicity is not handle at the edge but the centre is chosen where no cyclone happens
##  a way to handle it is to had the first and last bins in longitude assuming they overlap 
##  as a lat/lon point will be only in one
##=================================================================================

class Generate_Grids( object ) :

    def __init__( self, da_msk=None, display=True ) : 

       self.fmsk = da_msk  
       self.display = display
       if self.fmsk is None: ErrorDisplay( "Generate_Grids needs a xarray with the lsm mask definition" ) 
       self.lon = self.fmsk.lon.values
       self.lat = self.fmsk.lat.values
       self.msk = self.fmsk.lsm.values.astype(float)
       self.Build_Grids()

    def Build_Grids( self ) :

       self.lon2D, self.lat2D = np.meshgrid( self.lon, self.lat )
       self.nLon = self.lon.size
       self.nLat = self.lat.size
       self.dlon = np.mean( np.diff(self.lon) )
       self.dlat = np.mean( np.diff(self.lon) )
       print( self.dlon, self.dlat)

       self.lon_edge = np.array( (self.lon-self.dlon/2).tolist() + [self.lon[-1]+self.dlon/2,] ) 
       self.lat_edge = np.array( (self.lat-self.dlat/2).tolist() + [self.lat[-1]+self.dlat/2,] )

       if self.display :
          print("\n ===============================") 
          print(  " GRID PROPERTIES [Generate_Grids class]" )
          print(  "  -> nLon, nLat    :", self.nLon, self.nLat )
          print(  "  -> dLon, dLat    : {0:.2f}, {1:.2f}".format( self.dlon, self.dlat ) )
          print(  "  -> Lon min/max   : {0:.2f} - {1:.2f}".format( min(self.lon), max(self.lon) ) )
          print(  "  -> Lat min/max   : {0:.2f} - {1:.2f}".format( min(self.lat), max(self.lat) ) )
          print(  "  -> Edge Dimension:", self.lon_edge.shape, self.lat_edge.shape )

    def Compute_Distance_to_Region( self, regionLONs, regionLATs ) :
        LonLnd = self.lon2D.ravel()
        LatLnd = self.lat2D.ravel()
        d2lnd  = np.zeros_like( self.lon2D.ravel() )
        for i in range( self.lon2D.ravel().size ) :
            dd = haversine( regionLONs, regionLATs, LonLnd[i], LatLnd[i] )
            d2lnd[i] = np.nanmin( dd )
        self.d2lnd = np.reshape( d2lnd, self.lon2D.shape )
        return self.d2lnd 


##=================================================================================
##  ORIGINAL CLASS TO HANDLE GRIDS
##  with the move to zarr archive and better handling of periodicity in longitude
##  migrating slightly the code
##=================================================================================

class Gridded_Domain(object) :

   def __init__( self, Lonmin=-180, Lonmax=180, Latmin= -90, Latmax= 90, \
                       Resolution = 1, MaskFile = None, \
                       MaskType='interpolation', MaskMeth='bilinear', MaskThreshold = 1, fSuffix='' ) :

       print( "Gridded_Domain is mostly RETIRED (since move to zarr storage)")
       self.Lonmin = Lonmin; self.Lonmax = Lonmax
       self.Latmin = Latmin; self.Latmax = Latmax
       self.fSuffix = fSuffix
       self.Res = Resolution; self.file_msk = MaskFile; self.msk_thr = MaskThreshold
       self.Build_Grids()
       if self.file_msk is not None : self.ComputeMask( mtype=MaskType, meth=MaskMeth )
       #self.CreateTreeNeighbors()

   def Build_Grids(self) :
       self.lon_edge = np.arange( self.Lonmin-self.Res/2., self.Lonmax+self.Res/2.+0.0001, self.Res )
       self.lat_edge = np.arange( self.Latmin-self.Res/2., self.Latmax+self.Res/2.+0.0001, self.Res )
       self.lon = ( self.lon_edge[1::] + self.lon_edge[0:-1] ) / 2.
       self.lat = ( self.lat_edge[1::] + self.lat_edge[0:-1] ) / 2.
       self.lon2D, self.lat2D = np.meshgrid( self.lon, self.lat )
       self.nLon = self.lon.size
       self.nLat = self.lat.size

       ## some periodic longtitude for plots
       self.lon385E = monobound( periodic( np.copy(self.lon), xmin=  25. ))
       self.lon180E = monobound( periodic( np.copy(self.lon), xmin=-180. ))
       self.lon385E_edge = monobound( periodic( np.copy(self.lon_edge), xmin=  25. ) )
       self.lon180E_edge = monobound( periodic( np.copy(self.lon_edge), xmin=-180. ) )
       self.lon180E_2D, self.lat180E_2D = np.meshgrid( self.lon180E, self.lat )
       self.lon385E_2D, self.lat385E_2D = np.meshgrid( self.lon385E, self.lat )

   def ComputeMask(self, mtype="interpolation", meth='bilinear') :
       self.fmsk  = xr.open_dataset( self.file_msk )
       try : self.fmsk  = self.fmsk.rename( { 'latitude':'lat', 'longitude':'lon'} )
       except : pass
       if mtype=="interpolation" : self.fmski = Downscaling_Data( self.fmsk, mtype, self.lon, self.lat, fout="../interp_weights/Weights_Interpolation_LSM{0}".format(self.fSuffix), method=meth )
       else : self.fmski = Downscaling_Data( self.fmsk, mtype, self.lon_edge, self.lat_edge )
       a = xr.where( self.fmski.lsm>self.msk_thr, 1.0, 0.0 )
       self.fmski = self.fmski.assign(lsm=a)
       self.msk   = self.fmski.variables['lsm' ].values
       if a.shape == 3 : self.msk = self.msk[0]
       #self.msk_interp_f = interp2d( lon_msk, lat_msk, msk )
       #self.msk = self.msk_interp_f( self.lon, self.lat )
       #self.msk[self.msk>=self.msk_thr] = 1

   def ComputeDist2Coast( self ) :
       #find loand point
       LonLnd = self.lon2D[self.msk==1].ravel()
       LatLnd = self.lat2D[self.msk==1].ravel()
       d2lnd  = np.zeros_like( self.lon2D.ravel() ) 
       for i in range( self.lon2D.ravel().size ) : 
           dd = haversine( LonLnd, LatLnd, self.lon2D.ravel()[i], self.lat2D.ravel()[i])
           d2lnd[i] = np.nanmin( dd )
       self.d2lnd = np.reshape( d2lnd, self.lon2D.shape )
       #self.d2lnd[self.d2lnd==0] = np.nan
       #plt.imshow( self.d2lnd[::-1] );plt.colorbar();plt.show()

   def CreateTreeNeighbors( self ) :
       self.points = np.array( [self.lon2D.ravel(), self.lat2D.ravel()] ).T
       self.NNtree = spatial.cKDTree(self.points)

   def NumberOfLandNeighbors( self, radius, mask=False) :
       nLandNeighbors = np.zeros_like(self.points[:,0], dtype=float)
       dum = self.NNtree.query_ball_point( self.points, radius )
       for i in range( nLandNeighbors.size ):
           tmp = self.msk.ravel()[ dum[i] ]
           nLandNeighbors[i] = np.sum(tmp)/len(dum[i])
       aa = np.reshape( nLandNeighbors, self.lon2D.shape ) 
       if mask : aa[self.msk==1] = np.nan
       return aa

   def quick_raw_interp( self, data, gridnew, mask=True ) :
       ## data needs to be on the present grid while gridnew is the new grid
       print( "THIS IS A STANDARD INTERPOLATION AND NOT A PROPER FOR LAT/LON - MIGHT CREATE ISSUES" )
       interp_f = interp2d( self.lon, self.lat, data )
       datanew  = interp_f( gridnew.lon, gridnew.lat )
       if mask : datanew[ gridnew.msk==1] = np.nan
       return datanew



#=============================================================
#                       INFORMATION
#=============================================================
"""

  Some functions to manage domain to plot

--------------------------------------------------------------
"""
__author__      = "Nicolas Bruneau"
__copyright__   = "Copyright 2021, Reask"
__version__     = "Trackset V22"

#=============================================================
#                   LOAD STANDARD MODULES
#=============================================================

import numpy  as np
import os, sys, datetime, time
os.environ["CARTOPY_USER_BACKGROUNDS"] = "/mydisk/storage/DATA/BACKGROUND/"

## plotting libraries
import matplotlib.pyplot as plt
from   matplotlib.font_manager import FontProperties
font0  = FontProperties(); font = font0.copy(); font.set_weight('bold'); font.set_size('large')
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec

## cartopy library for earth plot
import cartopy.crs as ccrs
from   cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from   cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature
import xarray as xr

#from   src.ClimateDataHandling import *
from   src.Useful import *

#=============================================================
#                     INTERNAL FUNCTIONS
#=============================================================

def Get_XY_PCOLOR( LON, LAT ) :
    'transform centre to bottom left edge'
    dx = np.nanmean( np.diff(LON) )
    LON = LON - dx/2
    LON = np.concatenate( [LON,[LON[-1]+dx]])
    dy = np.nanmean( np.diff(LAT) )
    if dy < 0 : LAT = LAT + dy/2; LAT = np.concatenate( [[LAT[0]-dy],LAT])
    else :      LAT = LAT - dy/2; LAT = np.concatenate( [LAT,[LAT[-1]+dy]])
    return( LON, LAT )

class My_Geo_Config( object ) :
    def __init__( self ) :
       self.name = "Create a geogrpahical config for plotting"

    def Get_Config( self, region ) :
       self.GeoRegion = region
       self.shiftLong = False;

       if region == 'AT':
          self.LonMin = -100; self.LonMax = 20; self.dLon = 10
          self.LatMin =    5; self.LatMax = 60; self.dLat = 10
          self.shiftLong = True;
       elif region == 'ATfct':
          self.LonMin = -110; self.LonMax = -60; self.dLon = 10
          self.LatMin =   16; self.LatMax = 52; self.dLat = 10
          self.shiftLong = True;
       elif region == 'ATlse':
          self.LonMin = -100; self.LonMax = -65; self.dLon = 10
          self.LatMin =   24; self.LatMax = 46; self.dLat = 10
          self.shiftLong = True;   
       elif region == 'ATold':
          self.LonMin = -100; self.LonMax = -1; self.dLon = 10
          self.LatMin =    0; self.LatMax = 60; self.dLat = 10
          self.shiftLong = True;
       elif region == 'WP':
          self.LonMin =  90; self.LonMax = 170; self.dLon = 10
          self.LatMin =   0; self.LatMax =  65; self.dLat = 10
          self.shiftLong = False;
       elif region == 'EP':
          self.LonMin = 140; self.LonMax = 280; self.dLon = 20
          self.LatMin =   0; self.LatMax =  60; self.dLat = 10
          self.shiftLong = True;   
       elif region == 'NI':
          self.LonMin = 40; self.LonMax = 105; self.dLon = 10
          self.LatMin =   0; self.LatMax = 40; self.dLat = 10
          self.shiftLong = False;
       elif region == 'SI' :
          self.LonMin =  30; self.LonMax = 150; self.dLon = 10
          self.LatMin = -65; self.LatMax =   0; self.dLat = 10
          self.shiftLong = False;
       elif region == 'SP' :
          self.LonMin = 100; self.LonMax = 220; self.dLon = 10
          self.LatMin = -70; self.LatMax =   0; self.dLat = 10
          self.shiftLong = False;
       elif region == 'NHreduced' :
          self.LonMin =  30; self.LonMax =  20; self.dLon = 20
          self.LatMin = -10; self.LatMax =  60; self.dLat = 10
          self.shiftLong = False;
       elif region == 'SHreduced' :
          self.LonMin =  30; self.LonMax = 250; self.dLon = 20
          self.LatMin = -60; self.LatMax =  10; self.dLat = 10
          self.shiftLong = False;
       elif region == 'GLOBE360' :
          self.LonMin =   0; self.LonMax = 359; self.dLon = 30
          self.LatMin = -70; self.LatMax =  70; self.dLat = 30
          self.shiftLong = False;
       elif region == 'GLOBE385' :
          self.LonMin =  25; self.LonMax = 384; self.dLon = 30
          self.LatMin = -70; self.LatMax =  70; self.dLat = 30
          self.shiftLong = False;
       elif region =='US_NA': #US NA only
          self.LonMin =  -100; self.LonMax = -50; self.dLon = 30
          self.LatMin = 10; self.LatMax =  50; self.dLat = 30
          self.shiftLong = False
   
       else: 
          ErrorDisplay( "region not implemented in My_Geo_Config (given {0})".format(region) )

       if region != 'GLOBE360' :  
          self.LonMin385E = periodic( [self.LonMin,], xmin=25 )[0]
          self.LonMax385E = periodic( [self.LonMax,], xmin=25 )[0]
       else : 
          self.LonMin385E = self.LonMin 
          self.LonMax385E = self.LonMax

    def subselect_latitude( self, da ) :
        da, self.CresLatDir = subselect_da( da, self.LonMin, self.LonMax, self.LatMin, \
                               self.LatMax, self.shiftLong)
        return da

    def Get_Grid_Edge( self, da ) :
        self.Lon, self.Lat = Get_XY_PCOLOR( da.lon.values, da.lat.values )
        return self.Lon, self.Lat 



#=============================================================
#              FUNCTION FIGURES PLOT / CARTOPY
#=============================================================

## Define Projection
#Proj_shift = ccrs.PlateCarree( central_longitude=180)
#Proj_real  = ccrs.PlateCarree( central_longitude=0. )

def set_plot_region( ax, Proj_real, LonMin=230, LonMax=305, LatMin=15, LatMax=60, dLon=10, dLat=5, \
                     land=None, ocean=None, res='50m', xtop=False ) :

    ax.coastlines( resolution=res, zorder=10000 ); 
    LonTicks = np.arange( -180  , LonMax+0.001, dLon ) 
    LatTicks = np.arange( np.round(LatMin), LatMax+0.001, dLat )
    ax.set_xticks(LonTicks, crs=Proj_real); ax.set_xticklabels(LonTicks); 
    if xtop : ax.xaxis.tick_top()
    ax.set_yticks(LatTicks, crs=Proj_real); ax.set_yticklabels(LatTicks); 
    lon_formatter = LongitudeFormatter(); ax.xaxis.set_major_formatter(lon_formatter)
    lat_formatter = LatitudeFormatter() ; ax.yaxis.set_major_formatter(lat_formatter)
    if land is not None :
       land_feat = cfeature.NaturalEarthFeature( 'physical', 'land', scale=res, \
                                    edgecolor='face', alpha=land["alpha"], facecolor=land["fc"] )
       ax.add_feature(land_feat, zorder=50)
    if ocean is not None :
       ocean_feat = cfeature.NaturalEarthFeature( 'physical', 'ocean', scale=res, \
                                    edgecolor='face', alpha=ocean["alpha"], facecolor=ocean["fc"] )
       ax.add_feature( ocean_feat )
    ax.grid(linewidth=0.5, color='black', alpha=0.6, linestyle='--', zorder=10)
    return ax


def wrapper_maps_TC( region, myfigsize=None, domain=None, landmask=True ) :

    ## DEFINE REGION
    if domain is None :
       geoReg = My_Geo_Config()
       geoReg.Get_Config( region )
       Lonmin = geoReg.LonMin385E; Lonmax = geoReg.LonMax385E; dLon = geoReg.dLon
       Latmin = geoReg.LatMin; Latmax = geoReg.LatMax; dLat = geoReg.dLat
    else :
       Lonmin, Lonmax, Latmin, Latmax, dLon, dLat = domain 

    ww = Lonmax-Lonmin; hh = Latmax-Latmin

    ## DEFINE PROJECTION
    myProjR = ccrs.PlateCarree( central_longitude=0. )
    if region in ["NI", "AT",  ] : myProjS = ccrs.PlateCarree( central_longitude=0. )
    elif "reduced" in region :     myProjS = ccrs.PlateCarree( central_longitude=205. )
    else :                         myProjS = ccrs.PlateCarree( central_longitude=180. )

    ## land sea fetures
    landfeat = {"alpha":0.5, "fc": np.array([0.2,0.2,0.2]) }


    ## FIGURE
    if myfigsize is not None :  figw = myfigsize; print( figw,0.9*figw*hh/ww ); fig = plt.figure( figsize = ( figw,0.9*figw*hh/ww ) )
    else :    
       if "reduced" in region : figw = 12; fig = plt.figure( figsize = ( figw,1.3*figw*hh/ww ) )
       else :                   figw =  8; print( figw,0.9*figw*hh/ww, hh, ww ); fig = plt.figure( figsize = ( figw,0.9*figw*hh/ww ) )
    gs  = gridspec.GridSpec( 1, 1 ); gs.update( wspace=0.1, hspace=0.0, top = 0.92, bottom = 0.08, left = 0.06, right = 0.8 )
    ax0 = plt.subplot( gs[0], projection=myProjS )
    if landmask : ax0 = set_plot_region( ax0, myProjR, LonMin=Lonmin, LonMax=Lonmax, LatMin=Latmin, LatMax=Latmax, dLon=dLon, dLat=dLat, land=landfeat )
    else        : ax0 = set_plot_region( ax0, myProjR, LonMin=Lonmin, LonMax=Lonmax, LatMin=Latmin, LatMax=Latmax, dLon=dLon, dLat=dLat, ocean=landfeat ) 
    ax0.set_extent( [Lonmin,Lonmax,Latmin,Latmax], crs=myProjR )
      
    return fig, ax0, myProjR









#=============================================================
#                       INFORMATION
#=============================================================
"""

  Handling subregions based on polygons stored 
  in shapefile format

--------------------------------------------------------------
"""
__author__      = "Nicolas Bruneau"
__copyright__   = "Copyright 2021, Reask"
__version__     = "Trackset V22"


#=============================================================
#                      USER PARAMETERS
#=============================================================


#=============================================================
#                   LOAD STANDARD MODULES
#=============================================================

## general modules
import numpy  as np
import pandas as pd 
import os, sys, datetime, time, glob

## plotting libraries
import matplotlib.pyplot as plt
from   matplotlib.font_manager import FontProperties
font0  = FontProperties(); font = font0.copy(); font.set_weight('bold'); font.set_size('large')
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec

## geopandas tools
import geopandas as gpd
from   shapely.geometry import Point, Polygon
import shapely.speedups
shapely.speedups.enable()
from pyproj import CRS


#=============================================================
#                     INTERNAL FUNCTIONS
#=============================================================

def min_distance( point, ashape ):
    return ashape.boundary.distance( point )

 
#=============================================================
#                       CORE DRIVER
#=============================================================


class SubRegions( object ) :

   def __init__( self, shp_file = None, display = False, figcheck=False ) :  

       if shp_file is None : print( "shp_file parameter is required" )
       self.geopolys = gpd.read_file( shp_file )
       self.geopolys = self.geopolys.sort_values(by=["id",]).reset_index(drop=True)
       self.display = display
       if self.display:
          print( shp_file ) 
          print( self.geopolys )
          print( self.geopolys.shape )
       self.figcheck = figcheck
       ## check unicity of the row
       self.regionIDs  = self.geopolys.id.unique()
       self.Geometries = self.geopolys.Name.unique()
       if self.Geometries.size != self.geopolys.shape[0] :
          print( self.Geometries.size, "vs", self.geopolys.shape[0] )
          print( "WE NEED UNICITY OF THE POLYGONS / NAMES IN KML file:" )
          print( kml_file )
          sys.exit()
       ## Map ID to Name via dictionary   
       self.Create_Mapping()

   def Create_Mapping( self ) :
       self.ID2Name_map = { -2: "NOT DEFINED" }
       self.Name2ID_map = { "NOT DEFINED": -2 };
       for name in self.Geometries :
           poly  = self.geopolys.loc[ self.geopolys.Name==name ]
           self.ID2Name_map[ poly.id.values[0] ]  = name
           self.Name2ID_map[ name   ]  = poly.id.values[0]
  
   def find_which_polygon( self, LON, LAT, mask=None, mapping=None ) :
       ind = LON > 180; LON[ind] -= 360
       ## create Points
       myPts = gpd.points_from_xy( LON.ravel(), LAT.ravel() )
       ## initialise data
       self.which_region = np.zeros_like( LON, dtype=int )-2;
       for name in self.Geometries :
           poly  = self.geopolys.loc[ self.geopolys.Name==name ]
           index = poly.index.values[0]; 
           thispoly = Polygon( poly.loc[index, 'geometry'] )
           region   = np.reshape( myPts.within( thispoly ), LON.shape )
           self.which_region[ region ] = poly.id.values[0]

       ## reallocate region in case of available mapping
       if mapping is not None :
          for i in mapping.keys() :
              ind = self.which_region == i 
              self.which_region[ind] = mapping[i]

       ## store infos 
       self.Regions  = np.array(sorted( self.ID2Name_map.keys() ))
       self.nRegions = self.Regions.size

       ## mask data
       if mask is not None :
          self.which_region *= mask
          self.mask = mask

       ## some figure to check
       if self.figcheck :    
          mycolour = np.zeros_like(self.which_region)+np.nan
          neworder = np.unique( self.which_region[~np.isnan(self.which_region)])
          np.random.shuffle( neworder )
          for i,v in enumerate( neworder.tolist() ) :
              ind = self.which_region == i
              mycolour[ind] = v
              #print( i, v, self.ID2Name_map[v], np.sum(ind) )    

          mycolour = self.which_region
          try    : cb = plt.imshow( mycolour, vmin = self.Regions.min(), vmax = self.Regions.max() ); 
          except : cb = plt.scatter( LON, LAT, c=mycolour, vmin = self.Regions.min(), vmax = self.Regions.max(), cmap=plt.cm.jet) 
          ind = self.which_region == -1 
          plt.plot( LON[ind], LAT[ind], 'ks')
          tickval = []; ticklab = []
          for i in self.Regions : 
              tickval.append( i ) 
              ticklab.append( self.ID2Name_map[i] )
          cbar = plt.colorbar( cb, ticks=tickval )
          cbar.ax.set_yticklabels( ticklab )
          plt.show()
       return self.which_region

   def distance2polygon( self, LON, LAT ) :
       myPts = gpd.points_from_xy( LON.ravel(), LAT.ravel() )
       myPts = gpd.GeoSeries( myPts ); myPts.crs=4326
       ## initialise data
       self.dist2poly = {}
       print( self.Geometries )
       for name in self.Geometries :
           poly  = self.geopolys.loc[ self.geopolys.Name==name ]
           index = poly.index.values[0];
           regid = poly.id.values[0]
           thispoly = Polygon( poly.loc[index, 'geometry'] )
           t = gpd.GeoSeries(thispoly); t.crs = 4326 
           # project
           aeqd = CRS(proj='aeqd', ellps='WGS84', datum='WGS84', lat_0=t.centroid.y.values[0], lon_0=t.centroid.x.values[0]).srs
           p = myPts.to_crs(crs=aeqd)
           t = t.to_crs(crs=aeqd)
           dist = p.apply( lambda x: t.distance(x) ).values 
           dist = np.reshape( np.squeeze(dist), LON.shape ) / 1000. #convert to km 
           self.dist2poly[ regid ] = dist
       return self.dist2poly

   def display_mapping( self ) :
       try : 
           for reg in self.ID2Name_map : 
               print(reg, self.ID2Name_map[reg])
       except : print( "YOU NEED TO FIRST LOAD THE find_which_polygons METHOD"); sys.exit()








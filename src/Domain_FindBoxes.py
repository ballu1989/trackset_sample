#=============================================================
#                       INFORMATION
#=============================================================
"""



--------------------------------------------------------------
"""
__author__      = "Nicolas Bruneau"
__copyright__   = "Copyright 2021, Reask"

#=============================================================
#                   LOAD STANDARD MODULES
#=============================================================

import os, sys, operator, functools
import numpy as np
import glob
import pandas as pd 

#=============================================================
#                   INTERNAL USEFUL FCTNS 
#=============================================================

def find_boxes( cutLon, cutLat, Merge, LON, LAT, allBoxOrder=None, returnBasin=False ):
    'find in which cluster are LON/LAT based on merged boxes'
    ## Checks and conversion
    if isinstance(LON, list) : LON = np.array(LON); LAT = np.array(LAT)
    if isinstance(LON,(list, np.ndarray)): clust = np.zeros_like( LON,dtype=np.int ); isarray=True
    else : clust = 0; isarray=False
    if returnBasin and allBoxOrder is None : print('with returnBasin you need a allBoxOrder mapping'); sys.exit()
    ## Loop over boxes
    for iLon in range( len(cutLon)-1 ) :
        i1 = cutLon[iLon]; i2 = cutLon[iLon+1]
        for iLat in range( len(cutLat)-1 ) :
            j1 = cutLat[iLat]; j2 = cutLat[iLat+1]
            name = 'i{0}j{1}'.format( iLon+1, iLat+1 )
            ind  = (LON>i1)*(LON<=i2)*(LAT>j1)*(LAT<=j2);
            if isarray : clust[ind] = Merge[name]
            else :
              if ind : clust = Merge[name]

    ## correction extremes
    clust_max = 0; clust_min = 10000; dum=[]
    for i in Merge.keys() :
        dum.append(Merge[i])
    dum = np.array(dum); 
    clust_max = np.max(dum)
    clust_min = np.min(dum)
 
    if isarray :
       ind = LON >= cutLon[-1]; clust[ind] = clust_max
       ind = LON <= cutLon[ 0]; clust[ind] = clust_min
    else :
       if LON >= cutLon[-1]: clust = clust_max
       if LON <= cutLon[ 0]: clust = clust_min

    if returnBasin :
       if isarray : rbasin = np.zeros_like( clust, dtype='|S2' )
       for i in allBoxOrder.keys() :
           if isarray :
              for j in allBoxOrder[i] :
                  ind =  clust == j
                  rbasin[ind] = i
           else :
              if clust in allBoxOrder[i] : rbasin = i
       return rbasin.astype(str)
    else : return clust





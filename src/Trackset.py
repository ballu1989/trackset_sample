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

## other local lib to load
sys.path.append('..')
from   src.Useful              import *
from   src.Domain_SubRegions   import *

#=============================================================
#                     INTERNAL FUNCTIONS
#=============================================================

def get_season_start( hemis, seas, start_season_month, start_season_day ) :
    if   hemis == 'NH' : T1 = np.datetime64( '{0:04d}-{1:02d}-{2:02d}'.format(seas  ,start_season_month,start_season_day) )
    elif hemis == 'SH' : T1 = np.datetime64( '{0:04d}-{1:02d}-{2:02d}'.format(seas-1,start_season_month,start_season_day) )
    else : ErrorDisplay( "SEASON TIME ONLY DEVELOP FOR SH AND NH (givwn {0})".format(hemis) )
    return T1

def get_season_extended( hemis, seas, start_season_month, start_season_day, plusXmonth=0 ) :
    if start_season_month-1 > start_season_month : ErrorDisplay( "THIS CASE IS NOT IMPLEMENTED IN get_season_extended; need to add a modulo" )
    if start_season_month+1 > 12 : ErrorDisplay( "THIS CASE IS NOT IMPLEMENTED IN get_season_extended; need to add a modulo" )

    if hemis == 'NH' :
       T1 = np.datetime64( '{0:04d}-{1:02d}-{2:02d}'.format(seas  ,start_season_month-1,start_season_day) )
       T2 = np.datetime64( '{0:04d}-{1:02d}-{2:02d}'.format(seas+1,start_season_month+plusXmonth,start_season_day) )
    elif hemis == 'SH' :
       T1 = np.datetime64( '{0:04d}-{1:02d}-{2:02d}'.format(seas-1,start_season_month-1,start_season_day) )
       T2 = np.datetime64( '{0:04d}-{1:02d}-{2:02d}'.format(seas  ,start_season_month+plusXmonth,start_season_day) )
    else :
       ErrorDisplay( "SEASON TIME ONLY DEVELOP FOR SH AND NH (givwn {0})".format(hemis) )
    return T1, T2

def Name_Rate( Bas, iR, Reg ):
    if   Reg ==-1 : colname = Bas 
    elif Reg >= 0 : colname = "{0}_Rate_{1:02d}_Reg{2:02d}".format( Bas, iR, Reg )
    else : ErrorDisplay( "this case should not happen in Name_Rate (given {0})".format(Reg) )
    return colname

def Shifted_Longitude( LON ) :
    LON385E = periodic( np.copy(LON), xmin=   25. ) 
    LON180E = periodic( np.copy(LON), xmin= -180. )
    LON360E = periodic( np.copy(LON), xmin=    0. )
    return LON180E, LON360E, LON385E

def sort_time( df ) :
    return df.sort_values(by =["SID","DATE"] ).reset_index(drop=True)

def update_step( df ) :
    df = sort_time(df)   ## reorder by time first
    if "STEP" in df.columns : df = df.drop( "STEP", axis=1)
    df['STEP'] = df.groupby(['SID',]).cumcount()+1;
    return df

def Compute_nTC( df, Selection='REGION', seas1=1980, seas2=2020 ) :
    df = df.loc[(df.SEASON>=seas1)&(df.SEASON<=seas2)]
    ## only take first point of tracks as genesis
    if df.STEP.unique().size > 1 :
       df = df.drop( "STEP", axis=1)
       df['STEP'] = df.groupby(['SID',]).cumcount()+1;
       df = df.loc[ df.STEP==1 ].copy().reset_index(drop=True)
    ## do the ocunt
    aa = df.groupby( [ 'SEASON', Selection, ] )['SID'].count().unstack(fill_value=0).stack()
    bb = pd.DataFrame({ 'nTCs' : aa}).reset_index()
    cc = bb.pivot(index='SEASON', columns=Selection, values='nTCs').reset_index()
    for iS in range(seas1,seas2+1) :
        if iS not in cc.SEASON.values :
           dum = {'SEASON':iS}
           for iB in cc.columns.tolist() :
               if iB != "SEASON" : dum[iB] = -1
           cc = cc.append( dum, ignore_index=True )
    return cc.sort_values( "SEASON" )

def Compute_nTC_fromRegion( df, Basins2Subregion ) :
    for Bas in Basins2Subregion.keys() : 
        df[ Bas ] = 0 
        for Reg in Basins2Subregion[Bas][1::] :
            df[ Bas ] += df[ Reg ] 
    return df

def from_Subregion_to_Basin( Regions2Basin, df_Regions_Basins ):  
    for Reg in sorted( df_Regions_Basins.REGION.unique() ) :
        try :   mybas = Regions2Basin[Reg][0]
        except: mybas = "NOTALLOCATED"
        ind = df_Regions_Basins.BASIN.isna() & (df_Regions_Basins.REGION == Reg)
        df_Regions_Basins.at[ ind, "BASIN" ] = mybas
    return df_Regions_Basins

#=============================================================
#                       CORE DRIVER
#=============================================================


class TrackSet( object ) :


   def __init__( self, Region_def = None, shp_file = None, df_trk = None, mapping=None ) :  

       if shp_file   is None : ErrorDisplay( "shp_file parameter is required" )
       if Region_def is None : ErrorDisplay( "dictionnary Region_def is required" )
          
       ## set variables
       self.Basins2Region = Region_def
       self.Regions2Basin = inv_dictionary( Region_def )
         
       ## load associated shapefile (this should be coherent with the dictionary in term of IDs)
       self.TCregions = SubRegions( shp_file = shp_file, figcheck=False )
       
       ## if a dataframe is passed allocate it
       if df_trk is not None : self.trk = df_trk

       self.mapping = mapping

   def init_track_pts( self, dataframe_with_trackpoints ) :
       print( "WARNING: init_track_pts function of TrackSet ony designed and tested to pre-process IBTrACS")
       self.trk = dataframe_with_trackpoints
       for i in ['LON385E','LON180E','LON360E', ] :
           self.trk[ i ] = np.nan
       for i in [ "REGION", "BASIN", ] :
           self.trk[ i ] = pd.NA
       ## pertubate lat,lon so it doesn't fall on edge of poly and be rejected
       self.trk.LON += 0.0001 

   def get_longitude( self ):
       ind = self.trk['LON385E'].isna()
       a, b, c  = Shifted_Longitude( self.trk.loc[ind].LON.values )
       self.trk.at[ind,'LON385E'] = c 
       self.trk.at[ind,'LON180E'] = a
       self.trk.at[ind,'LON360E'] = b

   def get_basin_subregion( self ) :
       ind = self.trk["REGION"].isna()
       dum = self.trk.loc[ind]
       self.trk.at[ind,"REGION"] = self.TCregions.find_which_polygon( dum.LON180E+0.0001, dum.LAT+0.0001, mapping=self.mapping )

       ## reallocate region in case of available mapping
       #if self.mapping is not None : 
       #   for i in self.mapping.keys() :
       #       self.trk.at[ self.trk.REGION == i, "REGION"] = self.mapping[i]

       ## Then do the basin category too
       self.trk = from_Subregion_to_Basin( self.Regions2Basin, self.trk )

   def sort_by_time( self ) :
       self.trk = sort_time( self.trk )

   def get_genesis( self, reset_STEP=False ) : 
       if reset_STEP : self.trk = update_step( self.trk )
       self.genesis = self.trk.loc[ self.trk.STEP==1 ].copy().reset_index(drop=True)
       return self.genesis

   def get_nTC( self ) :
       ## Count
       #nTC_basin  = Compute_nTC( self.genesis, Selection='BASIN' , seas1=self.trk.SEASON.min(), seas2=self.trk.SEASON.max() )
       nTC_region = Compute_nTC( self.genesis, Selection='REGION', seas1=self.trk.SEASON.min(), seas2=self.trk.SEASON.max() )
       nTC_basin  = Compute_nTC_fromRegion( nTC_region, self.Basins2Region )
       self.nTCs = pd.merge( nTC_basin, nTC_region )
       ## Rates & Trials
       for Bas, myRegions in self.Basins2Region.items() :
           CountStorms = self.nTCs[ Bas ].copy()
           for iR, Reg in enumerate(myRegions[0:-1]) : ## last one is left over so no need to process it 
               if Reg >= 0 :
                  colname = Name_Rate( Bas, iR, Reg ) 
                  self.nTCs[ colname   ] = self.nTCs[ Reg ] / CountStorms 
                  self.nTCs[ colname.replace("Rate","nTrials") ] = CountStorms
                  CountStorms -= self.nTCs[ Reg ] 

       return self.nTCs



#=============================================================
#                       INFORMATION
#=============================================================
"""

  where to set what to load so each other bit of code
  starts from the same information

  all 3: oversea 0, overland +0.05

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

import sys
sys.path.append('..')
from mysetup import *   ## Load local properties

import os, sys, datetime, time
import numpy as np
#import h5py

from   src.Useful            import *
from   src.Units             import *
from   src.Domain_SubRegions import *

#=============================================================
#                     INTERNAL FUNCTIONS
#=============================================================


#=============================================================
#                       CORE DRIVER
#=============================================================


def load_tracks_file( what, Y1=None, Y2=None, UTC21=False ) : 

    ##-----------------------------------------------------------------
    ## if what is a csv we load it
    if ".csv" in what or ".hdf" in what or ".snp" in what or ".parquet" in what :

       myfiles = sorted( glob.glob(what) )
       df = []
       for myfile in myfiles :
           df.append( read_trk_file( myfile ) )
       df = pd.concat( df, axis=0, ignore_index=True)
       df = df[df['SAMPLE'] != 0]
       df.loc[df['LON'] < 180, 'LON'] = df['LON'] + 360
       
       if UTC21 : 
    #      df = df.loc[df.subregion=="AT"] 
          df["SID"] = df["id"]
          df["SOURCE"] = "ERA5"
          df["ENSEMBLE"] = "RAN"
          df["DAY_IN_SEASON"] = df['days_since_season_start']
          #df["REGION"] = df["subregion"]
          df["OVERLAND_FLAG"] = df.overland
          for i in ["sample", "season", "lon", "lat", "basin", "step"] :
              df[i.upper()] = df[i]
          df[ 'Hours'     ] = df['hours.since.genesis'] 
          df[ 'dCp_next'  ] = df['dCp.prev']
          df[ 'dLON_next' ] = df['dlon.prev']
          df[ 'dLAT_next' ] = df['dlat.prev']
 
          shp_file="{0}/SUBREGIONS/{1}_SubRegions.shp".format( UTC_DIR, hemisphere )
          TCregions = SubRegions( shp_file = shp_file, figcheck=False )
          dum_lon = periodic( df.LON, xmin=-180. )+0.000001; ind = dum_lon > 180; dum_lon[ind] -= 360
          df['REGION'] = TCregions.find_which_polygon( dum_lon, df.LAT+0.000001 )

    ##-----------------------------------------------------------------
       



    ## if TRAINING SET
    elif "IBTrACS_TRAIN" in what :
       df = pd.read_csv( "/fast/dev/utc22_storage/MODEL/DISPLACEMENT/training_data_filled.csv" )
       df["LON"] = periodic( df[ "LON" ], xmin=25)
       df[ "SAMPLE"   ] = 1
       df[ "ENSEMBLE" ] = 1

    ##-----------------------------------------------------------------
    ## if IBTrACS
    elif "IBTrACS" in what :  

       ## load region  
       shp_file="{0}/SUBREGIONS/{1}_SubRegions.shp".format( UTC_DIR, hemisphere )
       TCregions = SubRegions( shp_file = shp_file, figcheck=False )
 
       if   "INTERP"    in what : df = pd.read_csv( "/fast/dev/utc22_storage/MODEL/ibtracs_preprocessed_climate.csv" )
       elif "LINFILLED" in what : df = pd.read_csv( "/fast/dev/utc22_storage/MODEL/ibtracs_preprocessed_VmCpfilled_linear.csv" )
       elif "FILLED"    in what : df = pd.read_csv( "/fast/dev/utc22_storage/MODEL/ibtracs_preprocessed_VmCpfilled.csv" )
       elif "GENESIS"   in what : df = pd.read_csv( "/fast/dev/utc22_storage/MODEL/GENESIS/Genesis_NH_Location_1980-2020_USAwinds.csv" )
       else :                     df = pd.read_csv( "/fast/dev/utc22_storage/MODEL/ibtracs_preprocessed_nointerp.csv" )
       
       df[ "LON"] = periodic( df[ "LON" ], xmin=Lon0)
       df[ "Cp" ] = df[ "USA_PRES" ]
       df[ "Vm" ] = df[ "USA_WIND" ].apply( kt_to_ms )
       df[ "SOURCE"   ] = "IBTrACS"
       if "WMO" in what :
           print( "USE WMO FIELDS")
           df[ "Cp" ] = df[ "WMO_PRES" ]
           ##df[ "Vm" ] = df[ "WMO_WIND" ].apply( kt_to_ms )
           df[ "Vm" ] = df[ "V1min" ].apply( kt_to_ms )
           df[ "SOURCE"   ] = "IBTrACS_WMO"
       df[ "SAMPLE"   ] = 1
       df[ "ENSEMBLE" ] = 1
       df[ "BASIN"    ] = df.IB_BASIN
       df[ "DAY_IN_SEASON" ] = df.days_since_season_start

       if "GENESIS" in what : 
           df["OVERLAND_FLAG"] = 0
           df.at[df.DIST2COAST < 0, "OVERLAND_FLAG"] = 1


       ## ADD REGION
       dum_lon = periodic( df.LON, xmin=-180. )+0.000001; ind = dum_lon > 180; dum_lon[ind] -= 360
       df['REGION'] = TCregions.find_which_polygon( dum_lon, df.LAT+0.000001 )


    ##-----------------------------------------------------------------
    ## DEFAUT UTC MODEL
    elif "UTC_ERA5" in what : 

#       myfiles = sorted(glob.glob( "/fast/dev/utc22_storage/MODEL//STOCHASTIC/all9_test_*_Samples_0001-0025.csv" ))
       myfiles = sorted(glob.glob( "/fast/dev/utc22_storage/MODEL//STOCHASTIC/skRanger910_test_*_Samples_*.csv" ))
       myfiles = sorted(glob.glob( "/fast/dev/utc22_storage/MODEL/TRACKSET/ITERATE_Winds_InCyc_NEW_ERA5-RAN_*_Samples_000*.csvtest" ))
       myfiles = sorted(glob.glob( '/fast/dev//utc22_storage/MODEL//TRACKSET/WINDS_SST30.25_NEW_AT_ERA5_v2.1.0_*_Samples_*.csv' )) 
#       myfiles = [ "/fast/dev/utc22_storage/MODEL/BAYESIAN_ACTIVITY/AT/TC_activity_NH_AT_ERA5_1950-2020_2500samples_ENS-RAN_locations.csv", ]

       print(myfiles)

       df = []
       for myfile in myfiles : 
           dum = csv_or_hdf( myfile )
           #dum = dum[ [ "SID", "SEASON", "REGION", "Cp", "dLON_next", "dLAT_next", "LON", "LAT", "dCp_next" ] ]
           df.append( dum )
       df = pd.concat( df, axis=0, ignore_index=True)
       print(what,"read", df.shape)   

       df = df.loc[(df.SEASON>=1980)*(df.SEASON<=2021)] 

#       shp_file="{0}/SUBREGIONS/{1}_SubRegions.shp".format( UTC_DIR, hemisphere )
#       TCregions = SubRegions( shp_file = shp_file, figcheck=False )
#       dum_lon = periodic( df.LON, xmin=-180. )+0.000001; ind = dum_lon > 180; dum_lon[ind] -= 360
#       df['REGION'] = TCregions.find_which_polygon( dum_lon, df.LAT+0.000001 )
#       df["SID"] = np.arange( df.shape[0])
#       df['OVERLAND_FLAG'] = 0;  df['Cp'] = 0.; df["dLON_next"] = 0.
#       df["dLAT_next"] = 0.;  df['STEP']=1;  df["dCp_next"] = 0.
#       df["BASIN"]="AT"

       print("n STO", df.SID.unique().size, df.SEASON.unique().size, df.ENSEMBLE.unique().size, df.SAMPLE.unique().size )


    else: ErrorDisplay( "{0} NOT IMPLEMENTED IN load_trackset".format( what) )

    ##-----------------------------------------------------------------
    ## SELECTION TIME PERIOD
    if Y1 is not None : df = df.loc[ (df.SEASON>=Y1) ]
    if Y2 is not None : df = df.loc[ (df.SEASON<=Y2) ]

    ##-----------------------------------------------------------------
    ## CREATE EMPTY FIELD IF NECESSARY
    for iField in ["Vm","Rm","NAME"]:
        if iField not in df.columns : df[iField] = pd.NA


    print(df)

    print( " LOAD {0} Tracks over {1} Seasons and {2} Samples :".format( df.SID.unique().size,\
                                             df.SEASON.unique().size, df.SAMPLE.unique().size) , df.BASIN.unique() )

    print( "\n----------------------------------------------------\n" )

    return df


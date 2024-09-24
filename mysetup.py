####################################################################################################################################################################
##
##
## This file is loaded at the beginning of each script 
## so you can specify some paths here
##
####################################################################################################################################################################

import os, datetime, glob
import numpy as np
import sys
sys.path.append('../')
from   src.Useful import inv_dictionary, get_lat_label 

ROOT_DIR = "/fast/dev/"

UTC_DIR = "{0}/utc22_storage/MODEL/".format(ROOT_DIR)
UTC_OUT = "/slow/nico/utc22"

ZARR_MAIN_STORAGE = "{0}/utc22_storage/DATA/ZARR_STORE".format(ROOT_DIR)

## DEFINE PATHS
#SCRATCH_DIR   = '/mydisk/utc22_scratch/'
DATA_DIR      = '{0}/storage/DATA/'.format(ROOT_DIR)
OUTPUT_DIR    = "{0}/utc22_storage/".format(ROOT_DIR)
CLIMCONN_DIR  = "{0}/utc22_storage/CLIMATE_CONNECTORS".format(ROOT_DIR)
GATES_DIR     = "{0}/GATES/".format(UTC_DIR)
FIGS_DIR      = "{0}/utc22_storage/MODEL/FIGURES".format(ROOT_DIR)
FIGS_DIR      = "/home/balaji/Gitlab/trackset-analysis/clients/munichre"
TRACKSET_DIR  = "{0}/TRACKSET/".format(UTC_DIR)
TRACKSET_DIR  = "{0}/TRACKSET/".format(UTC_OUT)

#GATES_DIR     = "{0}/MSAMLIN_GATES/".format(UTC_DIR)
#GATES_DIR      = "/fast/dev/utc22_storage/MSAmlin/GATES_SET2_OFFICIAL"
#GATES_DIR     = "/fast/dev/utc22_storage/SECURIS/GATES_OFFICIAL_SET1_JUNE"
#GATES_DIR     = "/fast/dev/utc22_storage/REASK/GATES"
#GATES_DIR     = "/fast/dev/utc22_storage/LSE_GATES"
ANALYSIS_DIR   = '/fast/dev/utc22_storage/REASK/GENESIS'

## DATA PATH
ERA5_DATA     = "{0}/utc22_storage/DATA/ERA5/".format(ROOT_DIR)
LENS2_DATA    = "{0}/utc22_storage/DATA/LENS2/".format(ROOT_DIR)
C3S_DATA      = "/fast/dev/utc22_storage/DATA/C3S_SeasonalForecast/"
#RAW_ERA5_DATA = "/mydisk/dailydata/ERA5/MONTHLY/"

CMIP6_MRI_ESM1_2_LR = "/fast/dev/utc22_storage/DATA/CMIP6/mpi_esm1_2_lr/"


#RAW_ERA5_DIR  = '{0}/ERA5/MONTHLY/'.format( DATA_DIR )
#PROC_ERA5_DIR = '{0}/ClimateForcing/'.format( SCRATCH_DIR ) 
LANDSEA_MSK   = "/mydisk/dailydata/ERA5/lsm_1279l4_0.1x0.1.grb_v4_unpack.nc"
LANDSEA_MSK   = "{0}/utc22_storage/DATA/COARSEN_0.50_MASK.nc".format(ROOT_DIR)
 
####################################################################################################################################################################

hemisphere = 'SH'      ## The region you actually processes
hemisphere = 'NH'

## Interpolate on final grid
h_interp   = True
aggreg_mode = "interpolation"
#aggreg_mode = "aggregation"

## COASTLINE
dist2coast_nc  = '{0}/storage/DATA/dist2coast.signed.nc'.format(ROOT_DIR)
dist2coast_var = "dist2coast"

## topograhy
topo_nc  = "{0}/utc22_storage/DATA/etopo1_topography.nc".format(ROOT_DIR)
topo_var = "topo"


####################################################################################################################################################################
## TRAINING PERIODS

## FIT THE CLIMATE CONNECTORS ON WHICH PERIOD
Fitting_Year_init = 1980
Fitting_Year_end  = 2020


####################################################################################################################################################################
## CLIMATE CONNECTORS REGIONS

## DOMAIN DEFINITION FOR DIMENSIONALITY REDUCTION
## Lonin, Lonmax, Latmin, Latmax, hscale for plotting
domains  = { 'GLOBAL' : [   0, 360, -65, 65, 0.85 ], \
             'AT'     : [-100,  10,   0, 65, 1.   ], \
             'WP'     : [  70, 200, -20, 60, 1.   ], \
             'NI'     : [  30, 130, -10, 65, 1.1  ], \
             'EP'     : [ 200, 280,   0, 65, 1.05 ], \
             'EA'     : [ 110, 230, -60, 20, 0.97 ], \
             'WA'     : [  40, 160, -60, 20, 1.25 ], \
             'AF'     : [   0,  75, -60, 20, 1.25 ], }

domains  = { 'GLOBAL' : [   0, 360, -65, 65, 0.85 ], \
             'AT'     : [-100,  10,   0, 65, 1.   ], \
             'WP'     : [  70, 170, -20, 60, 1.   ], \
             'NI'     : [  30, 130, -10, 65, 1.1  ], \
             'EP'     : [ 140, 280,   0, 65, 1.05 ], \
             'EA'     : [ 110, 230, -60, 20, 0.97 ], \
             'WA'     : [  40, 160, -60, 20, 1.25 ], \
             'AF'     : [   0, 100, -60, 20, 1.25 ], }


####################################################################################################################################################################
## NESTED REGIONS
## to split regions with qgis: gis.stackexchange.com/questions/130949/is-there-a-way-to-split-polygons-by-several-lines-in-qgis

shp_regions = "{0}/SUBREGIONS/GLOBAL_FINAL_Regions.shp".format( UTC_DIR, hemisphere )

Basins2Subregion = { 'NI': [ -1,  2,  1,    ], \
                     'WP': [ -1,  4,  3,  5 ], \
                     'EP': [ -1,  7,  6,    ], \
                     'AT': [ -1,  9,  8, 10 ], \
                     'EA': [ -1, 14, 15,    ], \
                     'WA': [ -1, 13, 12,    ], \
                     'AF': [ -1, 11,        ]  }

Basins2Subregion = { 'NI': [ -1,  2,  1,    ], \
                     'WP': [ -1,  4,  3,  5 ], \
                     'EP': [ -1,  7,  6,    ], \
                     'AT': [ -1,  9,  8, 10 ], \
                     'EA': [ -1, 14, 15,    ], \
                     'WA': [ -1, 13, 12,    ], \
                     'AF': [ -1, 11,        ]  }

region_mapping_detailed = { 16:15, 66:6,}
region_mapping = { 16:15, 66:6 }   ## 16 and 15 regions are the same on each side of 180E

NH_REGIONS = [ "NI", "WP", "EP", "AT", ]
SH_REGIONS = [ "AF", "WA", "EA", ]


####################################################################################################################################################################
## LOAD HEMISPHERE SPECIFIC INFO

# if hemisphere == 'NH' : from definition_NH import * ; HEMIS_REGIONS = NH_REGIONS 
# if hemisphere == 'SH' : from definition_SH import * ; HEMIS_REGIONS = SH_REGIONS
 
# #start_season = datetime.datetime.strptime( "1901-{0}-{1}".format(start_season_month,start_season_day), "%Y-%m-%d")
# start_season = datetime.datetime( 1901, start_season_month, start_season_day )

# Basin2Subregion  = Basins2Subregion[subregion]

####################################################################################################################################################################
## DEFINE ZARR GRIDS

my_landthres = 0.5   ## [0-1] land / sea threshold
Lon0    = 25
myRes   = 1.0

#myRes   = 0.25

Lon_dst = np.arange( Lon0, Lon0+360.01, myRes )   ## LON grid
Lat_dst = np.arange(  -75.+myRes/2,  75.01, myRes )   ## LAT grid
Naming_Lon = "{0:.1f}W-{1:.1f}W".format( Lon_dst[0], Lon_dst[-1] )
Naming_Lat = "{0}-{1}".format( get_lat_label(Lat_dst[0]), get_lat_label(Lat_dst[-1]) )
ZARR_STORAGE = "{0}/Res_{1:.1f}deg_Lon_{2}_Lat_{3}".format( ZARR_MAIN_STORAGE, myRes, Naming_Lon, Naming_Lat)

def in_season( month, hemisphere ) :
    "Define Season to look at"
    if hemisphere == 'SH': return (month <  4) | (month == 12)
    if hemisphere == 'NH': return (month >= 7) & (month <= 10)

def in_season( da, hemisphere ) :
    "Define Season to look at"
    if hemisphere == 'SH':
       ## create a season field
       seasons = np.zeros(da.time.shape[0], dtype=np.int)
       allseas = np.unique(da['time.year'].values )
       for SEAS in np.arange( np.min(allseas)+1, np.max(allseas)+1) :
           ind = (da['time.year'] == SEAS-1) & (da['time.month']>= start_season_month)
           seasons[ind] = SEAS
           ind = (da['time.year'] == SEAS) & (da['time.month']< start_season_month)
           seasons[ind] = SEAS
       da = da.assign_coords( { "season": ( ("time", ), seasons, ), })
       #da = da.sel( season=da.season !=0)
       da = da.where(da.season!=0, drop=True)
       return (da['time.month'] <  4) | (da['time.month'] == 12),da
    if hemisphere == 'NH': return (da['time.month'] >= 7) & (da['time.month'] <= 10), da


Subregion2Basin = inv_dictionary( Basins2Subregion )
Subregions      = np.array(sorted( Subregion2Basin.keys() ))
Subregions      = Subregions[Subregions>0]
BASINs = np.array(sorted( Basins2Subregion.keys() )) 


## or binning large region
RegionalLatBin = np.arange( -85, 85.001, 10 )
RegionalLonBin = np.arange( Lon0, Lon0+360.01, 20 )
SSTBin = np.array( [-5,20,]+np.arange(21,31,0.5).tolist()+[31,40] )

####################################################################################################################################################################
## Vm model

mypower = 0.5
Penv    = 1013
Penv = 1015

####################################################################################################################################################################
## SOME ELNINO PARAMETERS (only end year: 1969-1970 -> 1970)
## https://ggweather.com/enso/oni.htm
ELNINO   = { 1970:0, 1973:2, 1977:0, 1978:0, 1980:0, 1983:3, 1987:1, 1988:2, 1992:2, 1995:1, 1998:3, 2003:1, 2005:0, 2007:0, 2010:1, 2015:0, 2016:3, 2019:0, }
LANINA   = { 1971:1, 1972:0, 1974:2, 1975:0, 1976:2, 1984:0, 1985:0, 1989:2, 1996:1, 1999:2, 2000:2, 2001:1, 2008:2, 2009:0, 2011:2, 2012:1, 2017:0, 2018:0, }

## LOAD PRECOMPUTED GATE
#gate_file = "{0}/{1}/{2}_Gates_Master.csv".format( OUTPUT_DIR, region, hemisphere )
#dfg = pd.read_csv( gate_file )
#gates = []; gates_name = []
#for i in range( dfg.shape[0] ) :
#    gates_name.append( dfg.Gate[i] )  
#    gates.append( [ (dfg.lon_init[i]%360, dfg.lat_init[i]), (dfg.lon_end[i]%360, dfg.lat_end[i]) ] )
#except : print( "NO GATE TO LOAD IN {0}".format(  gate_file) )

import pandas as pd
## READ ENSO pre-computed index
ENSOcut = 0.35
# ENSO_index = "{0}/{1}".format(DATA_DIR,ENSO_index_file)
# enso_ind = pd.read_csv( ENSO_index )
# enso_ind['season'] = enso_ind['season'].astype(int)
# ind = enso_ind.enso.values > ENSOcut
# yELNINO = enso_ind.season.values[ind].tolist()
# ind = enso_ind.enso.values < -ENSOcut
# yLANINA = enso_ind.season.values[ind].tolist()
#print("ELNINO", yELNINO)
#print("LANINA", yLANINA)

####################################################################################################################################################################
##
## CESM - LENS2 ENSEMBLES
##------------------------

CESM_LENS2_DIR = "/slow/nico/drought/CESM2/"
dum = np.sort( glob.glob( "{0}/PS/*2025*".format(CESM_LENS2_DIR) ) )
dum = np.sort( glob.glob( "{0}/*/SST*.nc".format(LENS2_DATA) ) )

#CESM_LENS2_ENS = []
#for i in dum :
#    #CESM_LENS2_ENS.append( i.split( ".f09_g17.LE2-")[1].split(".cam.h0.PS.")[0] )
#    CESM_LENS2_ENS.append( ".".join( i.split( "." )[-3:-1]) ) 
#print("NUMBER OF CESM LENS2 ENSEMBLE:", len(CESM_LENS2_ENS))

## CESM per BATCH to handle memory
## 3 members are not yet complete "1151.008", "1171.009", "1191.010",
CESM_LENS2_BATCH = [ [ "1011.001", "1031.002", "1051.003", "1071.004", "1091.005", "1111.006", "1131.007", ], \
                     [ "1231.011", "1231.012", "1231.013", "1231.014", "1231.015", "1231.016", "1231.017", "1231.018", "1231.019", "1231.020", ], \
                     [ '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', ], \
                     [ '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', ], \
                     [ '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020', ], \
                     [ '1001.001', '1021.002', '1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', ], \
                     [ '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', ], \
                     [ '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', ], \
                     [ '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', ], \
                     [ '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', ], ] 




CESM_LENS2_BATCH = [ [ "1011.001", "1031.002", "1051.003", "1071.004", "1091.005", ], \
                     [ "1111.006", "1131.007", "1151.008", "1171.009", "1191.010", ], \
                     [ "1231.011", "1231.012", "1231.013", "1231.014", "1231.015", ], \
                     [ "1231.016", "1231.017", "1231.018", "1231.019", "1231.020", ], \
                     [ '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', ], \
                     [ '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', ], \
                     [ '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', ], \
                     [ '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', ], \
                     [ '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', ], \
                     [ '1301.016', '1301.017', '1301.018', '1301.019', '1301.020', ], \
                     [ '1001.001', '1021.002', '1041.003', '1061.004', '1081.005', ], \
                     [ '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', ], \
                     [ '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', ], \
                     [ '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', ], \
                     [ '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', ], \
                     [ '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', ], \
                     [ '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', ], \
                     [ '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', ], \
                     [ '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', ], \
                     [ '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', ], ]


BATCHES_CLIMATES = { "LENS2": CESM_LENS2_BATCH, "ERA5" : [ "RAN", ] }



### DISSOCIATE THE BIOMASS BURNING EMISSION
SBMB_CESM_LENS2 = [ [ "1011.001", "1031.002", "1051.003", "1071.004", "1091.005", ], \
                    [ "1111.006", "1131.007", "1151.008", "1171.009", "1191.010", ], \
                    [ "1231.011", "1231.012", "1231.013", "1231.014", "1231.015", ], \
                    [ "1231.016", "1231.017", "1231.018", "1231.019", "1231.020", ], \
                    [ '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', ], \
                    [ '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', ], \
                    [ '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', ], \
                    [ '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', ], \
                    [ '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', ], \
                    [ '1301.016', '1301.017', '1301.018', '1301.019', '1301.020', ], ]
SBMB_CESM_LENS2 = [ item for sublist in SBMB_CESM_LENS2 for item in sublist ]

BMB_CESM_LENS2 = [ [ '1001.001', '1021.002', '1041.003', '1061.004', '1081.005', ], \
                   [ '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', ], \
                   [ '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', ], \
                   [ '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', ], \
                   [ '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', ], \
                   [ '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', ], \
                   [ '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', ], \
                   [ '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', ], \
                   [ '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', ], \
                   [ '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', ], ]
BMB_CESM_LENS2 = [ item for sublist in BMB_CESM_LENS2 for item in sublist ]
 


C3S_FCT_BATCH = [ ["MEMBER_00","MEMBER_01", "MEMBER_02", "MEMBER_03", "MEMBER_04"], \
                  ["MEMBER_05","MEMBER_06", "MEMBER_07", "MEMBER_08", "MEMBER_09"], \
                  ["MEMBER_10","MEMBER_11", "MEMBER_12", "MEMBER_13", "MEMBER_14"], \
                  ["MEMBER_15","MEMBER_16", "MEMBER_17", "MEMBER_18", "MEMBER_19"], \
                  ["MEMBER_20","MEMBER_21", "MEMBER_22", "MEMBER_23", "MEMBER_24"], \
                  ["MEMBER_25","MEMBER_26", "MEMBER_27", "MEMBER_28", "MEMBER_29"], \
                  ["MEMBER_30","MEMBER_31", "MEMBER_32", "MEMBER_33", "MEMBER_34"], \
                  ["MEMBER_35","MEMBER_36", "MEMBER_37", "MEMBER_38", "MEMBER_39"], \
                  ["MEMBER_40","MEMBER_41", "MEMBER_42", "MEMBER_43", "MEMBER_44"], \
                  ["MEMBER_45","MEMBER_46", "MEMBER_47", "MEMBER_48", "MEMBER_49", "MEMBER_50"], ]

#C3S_FCT_BATCH = [ ["RAN",], ]

#C3S_FCT_BATCH = [ ["MEMBER_00","MEMBER_01", "MEMBER_02", "MEMBER_03", "MEMBER_04"], \
#                  ["MEMBER_05","MEMBER_06", "MEMBER_07", "MEMBER_08", "MEMBER_09"], \
#                  ["MEMBER_10","MEMBER_11", "MEMBER_12", "MEMBER_13", "MEMBER_14"], \
#                  ["MEMBER_15","MEMBER_16", "MEMBER_17", "MEMBER_18", "MEMBER_19"], \
#                  ["MEMBER_20","MEMBER_21", "MEMBER_22", "MEMBER_23", "MEMBER_24"], ]

ENSEMBLE_CLIMATES = { "C3S_FCT": [ item for sublist in C3S_FCT_BATCH    for item in sublist ], \
                      "C3S_FCT_FCT_2022": [ item for sublist in C3S_FCT_BATCH    for item in sublist ], \
                      "LENS2"  : [ item for sublist in CESM_LENS2_BATCH for item in sublist ], \
                      "ERA5_WITH2022"   : [ "RAN", ], \
                      "ERA5_NEW" : [ "RAN", ] ,\
                      "ERA5"   : [ "RAN", ] }




# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: track_eval
#     language: python
#     name: python3
# ---

# # General info
# __Author__ = Reask <br>
# __copyright__ = "Copyright 2024, Reask"
#
# The script processes the sample trackset (120 years) and finds events at targets locations (latitude/longitude) defined by user user-defined <br>
# Following are the items produced by this script: <br>
# 1. landfalling events at user-defined locations and intensity threshold
# 2. Exceedance probability of cyclone intensity
#

#
# ### User input
#

# +
# Input track file
trackfile = 'North_Atlantic_Trackset_Sample_120years.csv'

# LOCATION 1: Miami [Name, Latitude, Longitude]
loc1 = ['Miami', 25.75, -80]

# LOCATION 2: New Orleans [Name, Latitude, Longitude]
loc2 = ['New Orleans', 30, -90]

# Search radius in miles
srad = 300

# Intensity (windspeed in m/s) threshold at Location 1 
# for loc 1
Vmax1 = 33 # Category 1 intensity
# for loc 1
Vmax2 = 33 # Category 1 intensity
# -

# ### Step1: Load Python packges 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import glob as glob
from cartopy.geodesic import Geodesic
import shapely.geometry as sgeom

# ### Step 2: functions

# +
# converting search radius from miles to kilometers
rad = srad * 1.60934

# create dataframe (table) of user locations
df_loc = pd.DataFrame([loc1, loc2]).rename(columns = {0: 'City', 1: 'lat', 2: 'lon'})

# function to calculate distance between track points and user-defined location
def calc_d(lat,lon, plat,plon):
    dist = 111* np.sqrt(((lat-plat)**2) + ((lon-plon)**2))
    return dist

# create search radius circle polygons
geoms = []
gd = Geodesic()

for lon, lat in zip(df_loc['lon'], df_loc['lat']):
        cp = gd.circle(lon=lon, lat=lat, radius=srad*1000)
        geoms.append(sgeom.Polygon(cp))

# function to plot tracks
def plot_tracks(df, loc=None, title=None):

        src_crs = ccrs.PlateCarree()

        fig, ax = plt.subplots(figsize=(12, 12),
                        subplot_kw=dict(projection=src_crs))

        extent = [-100, -67, 20, 45]

        ax.add_feature(cfeature.STATES, zorder=200, linewidth=0.5, edgecolor='k')
        ax.add_geometries(geoms, crs=src_crs, fc=(1,0,0,0.25), ec=(0,0,0,1), lw=2, zorder=200)

        ax.set_extent(extent)
        ax.coastlines('10m', color="grey", linewidth=1, zorder=200, alpha=1)
        ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='grey', zorder=100)

        if loc is not None:  
               ax.scatter(loc[2], loc[1], color = 'r', s=200, zorder=300)
        else:
                ax.scatter(loc1[2], loc1[1], color = 'r', s=200, zorder=300)
                ax.scatter(loc2[2], loc2[1], color = 'r', s=200, zorder=300)
        

        for ii in df.SID.unique() :
                b = df.loc[df.SID==ii].reset_index(drop=True)
                ax.plot(b.LON,b.LAT,color='k',zorder=200)

        if title is not None:
                ax.set_title(title, fontsize=25)


# -

# ### Step 3: Read track file and calculate distance

# +
# read input track csv file
trackset = pd.read_csv(trackfile)

# convert track longitude format form 0/360 to -180/180 
trackset["LON"] = (trackset["LON"] + 180) % 360 - 180

# calculate distance between track points and user location 1
trackset["dist1"] = trackset.apply(lambda row: calc_d( row.LON, row.LAT, loc1[2], loc1[1]), axis=1 )

# calculate distance between track points and user location 1
trackset["dist2"] = trackset.apply(lambda row: calc_d( row.LON, row.LAT, loc2[2], loc2[1]), axis=1 )
# -

# ### Step 4: Filter events

# +
# filter events for location 1 under 1) search radius and 2) intensity threshold
sids_loc1 = trackset.loc[(trackset.dist1 < srad) & (trackset.Vm > Vmax1)]['SID'].unique()
track_loc1 = trackset.loc[trackset.SID.isin(sids_loc1)]
print('No of events passing through ' + loc1[0] + ' is ' + str(sids_loc1.shape[0]))

# filter events for location 2 under 1) search radius and 2) intensity threshold
sids_loc2 = trackset.loc[(trackset.dist2 < srad) & (trackset.Vm > Vmax2)]['SID'].unique()
track_loc2 = trackset.loc[trackset.SID.isin(sids_loc2)]
print('No of events passing through ' + loc2[0] + ' is ' + str(sids_loc2.shape[0]))

# store storm IDs for events that passes through both locations
sids_conditional = track_loc1.loc[(track_loc1.dist2 < srad) & (track_loc1.Vm > Vmax2)]['SID'].unique()
track_conditional = track_loc1.loc[track_loc1.SID.isin(sids_conditional)]

print('No of events passing through ' + loc1[0] + ' and ' + loc2[0] + ' is ' + str(sids_conditional.shape[0]))

# -

# ### Step 5: Plot tracks

plot_tracks(track_loc1, loc1, 'Miami')
plot_tracks(track_loc2, loc2, 'New Orleans')
plot_tracks(track_conditional, None, 'Miami and New Orleans')

# ### Some basic STATS

# ##### 1. Plot probability density for Vm (cyclone windspeed)

# +
# Cut tracks for location 1
df1 = trackset.loc[(trackset.dist1 < srad) & (trackset.Vm > Vmax1)]
df2 = trackset.loc[(trackset.dist2 < srad) & (trackset.Vm > Vmax2)]
df21 = track_conditional.loc[(track_conditional.dist2 < srad) & (track_conditional.Vm > Vmax2)]

# Find max intensity for each event inside search radius
df1 = df1.groupby(['SID'])['Vm'].max().reset_index()
df2 = df2.groupby(['SID'])['Vm'].max().reset_index()
df21 = df21.groupby(['SID'])['Vm'].max().reset_index()

# plot figure
plt.figure(figsize=(12, 8))
df1.Vm.plot.kde(label=loc1[0], lw=2)
df2.Vm.plot.kde(label=loc2[0], lw=2)
df21.Vm.plot.kde(label=loc2[0] + ' | ' + loc1[0], lw=2)
plt.legend(fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.grid()
plt.ylabel('Probability',fontsize = 20)
plt.xlabel('Cyclone windspeed (m/s)',fontsize = 20)
plt.title('Probability of cyclone windspeed within ' + str(srad) + ' miles radius', fontsize = 20)
# -

# #### 2. Plot annual frequency breakdown by hurricane category (saffir simpson scale)

# +
# bins for hurricane category
bins = [33, 43, 50, 58, 70, 500]

category = [1,2,3,4,5]

# label Vm based on above bins
df1["category"] = pd.cut( df1.Vm, bins, labels = category )
df2["category"] = pd.cut( df2.Vm, bins, labels = category )
df21["category"] = pd.cut( df21.Vm, bins, labels = category )

# find counts per category
df1_group = df1.groupby(['category'])['SID'].count().reset_index().rename(columns={'SID': 'rate'})
df2_group = df2.groupby(['category'])['SID'].count().reset_index().rename(columns={'SID': 'rate'})
df21_group = df21.groupby(['category'])['SID'].count().reset_index().rename(columns={'SID': 'rate'})

# calculate annual rate
df1_group['rate'] = df1_group['rate'] / 120
df2_group['rate'] = df2_group['rate'] / 120
df21_group['rate'] = df21_group['rate'] / 120

# merge both tables
df_group = pd.merge(df1_group, df2_group, on='category').rename(columns=({'rate_x': loc1[0], 'rate_y': loc2[0]}))
df_group = pd.merge(df_group, df21_group, on='category').rename(columns=({'rate': loc2[0] + ' | ' + loc1[0]}))

#-------------
# plot figure
#-------------
fig, ax = plt.subplots(figsize=(12, 8))

df_group.plot(kind='bar', x='category', rot=0, ax=ax)
plt.legend(fontsize=20)

plt.tick_params(axis='both', which='major', labelsize=15)

plt.xlabel('Hurricane catgory (saffir-simpson scale)',fontsize = 20)
plt.ylabel('Annual frequency',fontsize = 20)


# -

# #### 3. Exceedance probability

# +
# Trackset period = 120 years
nyears = 120

# ordering winds from max to min values (descending)
Vm1 = df1.sort_values(by='Vm', ascending=False)['Vm'].values
Vm2 = df2.sort_values(by='Vm', ascending=False)['Vm'].values
Vm21 = df21.sort_values(by='Vm', ascending=False)['Vm'].values

# calculate cumulative frequency
rate1 = np.cumsum( np.repeat( 1./nyears, Vm1.size ) )
rate2 = np.cumsum( np.repeat( 1./nyears, Vm2.size ) )
rate21 = np.cumsum( np.repeat( 1./nyears, Vm21.size ) )

# return period = 1/EP
rp1 = 1./rate1
rp2 = 1./rate2
rp21 = 1./rate21

#-------------
# plot figure
#-------------
fig, ax = plt.subplots(figsize=(12, 8))
plt.plot(rp1, Vm1, label=loc1[0], lw=3)
plt.plot(rp2, Vm2, label=loc2[0], lw=3)
plt.plot(rp21, Vm21, label=loc2[0] + ' | ' + loc1[0], lw=3)
ax.set_xscale('log')
ax.set_xticks([2, 5, 10, 25, 50, 75, 120, ])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

plt.legend(fontsize=20)
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=15)

plt.xlabel('Return period (years)',fontsize = 20)
plt.ylabel('Cyclone windspeed (m/s)',fontsize = 20)
plt.title('Exceedance probability', fontsize = 20)

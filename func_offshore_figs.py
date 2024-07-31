import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import glob as glob
from cartopy.geodesic import Geodesic
import shapely.geometry as sgeom
import os
from   src.Useful              import *
from   src.Stats               import *
from   src.Fig_ColorBars       import *
from   src.Fig_Domains         import *
# from   src.Domain_FindBoxes    import *
# from   src.Domain_Grids        import *
# from   src.Domain_SubRegions   import *
from   src.Trackset            import *
from   load_track_data         import *

import seaborn as sns
import pickle
from   src.Fig_ColorBars       import *
reask_standard = ReaskBar.STANDARD( nSubPoints = 256 )
from matplotlib.offsetbox import  OffsetImage, AnnotationBbox 

from func_offshore_figs import *

NINO_NH = [1982, 1986, 1987, 1991, 1994, 1997, 2002, 2004, 2006, 2009, 2015, 2018]
NINA_NH = [1985, 1988, 1995, 1998, 1999, 2000, 2007, 2010, 2011, 2016, 2020, ]


#=============================================================
#                   PLOT EP AT GATES
#=============================================================

colourtheme='white'


def EP( val, nyears ) :
    rate = np.cumsum( np.repeat( 1./nyears, val.size ) )
    out  = np.sort(val)
    return rate, out

def get_EPs( df, what, nyears ) :
    RP, Val = EP( df[what].values, nyears )
    if "Vm" in what : Val = Val[::-1]
    return 1./RP, Val

def Plot_a_Gate(df, NSamp, city=None, what="Vm" , ) :

    im = matplotlib.image.imread("Reask_Logo.jpg")

    models = df.model.unique()
    
    ## SET UP FIGURE  
    fig = plt.figure(figsize=(14,5), dpi=250)
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 3]) 

    ax1  = fig.add_subplot( gs[1])

    ## plot others DarkCyan
    cores = [ ReaskColors["PinkRed"], ReaskColors["FlashyRed"]]
    cc = -1
    for Source in models:
        if "ERA5" in Source: 
            # for iP, Period in enumerate([[1980, 2020]]):
            st = df.loc[(df.model == Source)]
            # nsamp = NSamp[Source]

            if 'ERA5 [1980-2020]' in Source:
                nyears = 41*2500 #(Period[1] - Period[0] + 1) * nsamp
                RP, Vals = get_EPs(st, what, nyears)
                ax1.plot(RP, Vals, color=ReaskColors['Green'], linewidth=2.5,
                        alpha=0.9, zorder=1000, label=f"{Source}")

            if 'ERA5 El-Nino' in Source:
                nyears = 12*2500 #(Period[1] - Period[0] + 1) * nsamp
                RP, Vals = get_EPs(st, what, nyears)
                ax1.plot(RP, Vals, color=ReaskColors['DarkCyan'], linewidth=2.5,
                        alpha=0.9, zorder=1000, label=f"{Source}")

            if 'ERA5 La-Nina' in Source:
                nyears = 11*2500 #(Period[1] - Period[0] + 1) * nsamp
                RP, Vals = get_EPs(st, what, nyears)
                ax1.plot(RP, Vals, color=ReaskColors['PinkRed'], linewidth=2.5,
                        alpha=0.9, zorder=1000, label=f"{Source}")


            elif 'ERA5 AMO[-] [1980-1998]' in Source:
                nyears = 19*2500 #(Period[1] - Period[0] + 1) * nsamp
                RP, Vals = get_EPs(st, what, nyears)
                ax1.plot(RP, Vals, color=ReaskColors['DarkCyan'], linewidth=2.5,
                        alpha=0.9, zorder=1000, label=f"{Source}")
                
            elif 'ERA5 AMO[+] [1999-2020]' in Source:
                nyears = 22*2500 #(Period[1] - Period[0] + 1) * nsamp
                RP, Vals = get_EPs(st, what, nyears)
                ax1.plot(RP, Vals, color=ReaskColors['PinkRed'], linewidth=2.5,
                        alpha=0.9, zorder=1000, label=f"{Source}")
        if "CESM" in Source:
                cc += 1
                for iP, Period in enumerate([[1980, 2100]]):
                    st = df.loc[(df.model == Source)]
                    # nsamp = NSamp[Source]
                    nyears = NSamp #100 * nsamp
                    RP, Vals = get_EPs(st, what, nyears)
                    ax1.plot(RP, Vals, color=cores[cc], linewidth=2.5, alpha=0.9, zorder=1000, label=f"{Source}")

                        
    ## FINALISE FIGURES
    plt.xscale('log')
    ax1.set_facecolor('0.4')

    if colourtheme=='black':
        COLOR="w"
        trans = True
        
        sns.set()
        sns.axes_style({
        'text.color': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        })
        plt.rcParams['text.color'] = COLOR
        plt.rcParams['axes.labelcolor'] = COLOR
        plt.rcParams['xtick.color'] = COLOR
        plt.rcParams['ytick.color'] = COLOR
        plt.rcParams['grid.color'] = COLOR

        #axes
        ax1.tick_params(axis='x', colors=COLOR)
        ax1.tick_params(axis='y', colors=COLOR)
        ax1.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax1.xaxis.label.set_color(COLOR)
        ax1.yaxis.label.set_color(COLOR)

        #plot legend
        lgd = ax1.legend(fontsize=12, loc="upper left", framealpha=1)
        lgd.set_alpha(1)
        frame = lgd.get_frame()
        frame.set_facecolor("#000025")

    elif colourtheme == 'white':
        trans = False
        COLOR="black"

        #axes
        ax1.tick_params(axis='both', which='major', labelsize=12)
        ax1.get_xaxis().set_major_formatter(mticker.ScalarFormatter())

        #plot legend
        plt.legend(fontsize=12, loc="upper left", edgecolor='k', facecolor='0.4')


    imagebox = OffsetImage(im, zoom=0.0225)
    ab = AnnotationBbox(imagebox, (225, 7.5), frameon = False)
    ax1.add_artist(ab)


    #set x label, limits and ticks
    plt.xlabel( "Return Period (Years)", fontsize=13, labelpad=10)
    plt.xlim([1, 500]) 
    ax1.set_xticks([1, 3, 10, 30, 100, 300])

    #titles
    plt.title ("Offshore Gate {0}".format(city), fontsize=14,  color=COLOR)
    if   'Vm' in what:
        plt.ylabel( "Max 1min Wind (m/s)", fontsize=13,  labelpad=10 )
        plt.ylim([0, 100])
         
    plt.grid(True, color='0.7')

    ax2  = fig.add_subplot( gs[0], projection=ccrs.PlateCarree() )

    reask_gates = pd.read_csv('/mnt/slow/nico/utc22/DATA/OFFICIAL_Jan2023/gates_definition/NH_Gates_Master.csv')

    for i in range(len(reask_gates)-1):
        cb = ax2.plot((reask_gates.iloc[i]['lon_init'], reask_gates.iloc[i]['lon_end']), (reask_gates.iloc[i]['lat_init'], reask_gates.iloc[i]['lat_end']),
                    color= 'grey', linestyle='solid', linewidth=2, zorder=10000, alpha=0.7)

    reask_gates_c = reask_gates.loc[reask_gates.Gate == int(city)]

    ax2.plot((reask_gates_c['lon_init'], reask_gates_c['lon_end']), (reask_gates_c['lat_init'], reask_gates_c['lat_end']),
                color= 'r', linestyle='solid', linewidth=2, zorder=10000, alpha=1)

    extent = [-110, -65, 22, 50]

    ax2.coastlines('10m', color="k", linewidth=0.5, zorder=6000, alpha=1)

    ax2.add_feature(cfeature.LAND.with_scale('10m'), facecolor='grey', zorder=100)

    ax2.add_feature(cfeature.OCEAN.with_scale('10m'),zorder=5000, facecolor='w')
        
    ax2.add_feature(cfeature.STATES, zorder=6000, linewidth=0.5, edgecolor='k')

    ax2.set_title('GATE ' + str(city))


    ax2.set_extent(extent)

    plt.show()


def plot_reask_gate(gate):

    reask_gates = pd.read_csv('/mnt/slow/nico/utc22/DATA/OFFICIAL_Jan2023/gates_definition/NH_Gates_Master.csv')

    fig, ax = plt.subplots(figsize=(6, 6),
                    subplot_kw=dict(projection=ccrs.PlateCarree()))
    
    for i in range(len(reask_gates)-1):
        cb = ax.plot((reask_gates.iloc[i]['lon_init'], reask_gates.iloc[i]['lon_end']), (reask_gates.iloc[i]['lat_init'], reask_gates.iloc[i]['lat_end']),
                    color= 'grey', linestyle='solid', linewidth=2, zorder=10000, alpha=0.7)

    # gate = 50

    reask_gates_c = reask_gates.loc[reask_gates.Gate == gate]

    ax.plot((reask_gates_c['lon_init'], reask_gates_c['lon_end']), (reask_gates_c['lat_init'], reask_gates_c['lat_end']),
                color= 'r', linestyle='solid', linewidth=2, zorder=10000, alpha=1)

    extent = [-110, -65, 22, 50]

    ax.coastlines('10m', color="k", linewidth=0.5, zorder=6000, alpha=1)

    ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='grey', zorder=100)

    ax.add_feature(cfeature.OCEAN.with_scale('10m'),zorder=5000, facecolor='w')
        
    ax.add_feature(cfeature.STATES, zorder=6000, linewidth=0.5, edgecolor='k')

    ax.set_title('GATE ' + str(gate))


    ax.set_extent(extent)

    plt.show()


# =============================================================
#   # edit climate file names based off region
# =============================================================

def rename_filenames(region, subregion, file_names):
    if region == 'NORTH_ATLANTIC':
        if subregion == 'AT':
            for i in range(len(file_names)):
                file_names[i] = file_names[i].replace('WINDS', 'GATES_WINDS')
    if region == 'NORTH_INDIAN':
        if subregion == 'NI':
            for i in range(len(file_names)):
                file_names[i] = file_names[i].replace('NH_AT_v2.0.2', 'NH_NI_v2.0.2').replace('WINDS', 'GATES_WINDS')
    elif region == 'PACIFIC':
        if subregion == 'EP':
            for i in range(len(file_names)):
                file_names[i] = file_names[i].replace('NH_AT_v2.0.2', 'NH_EP_v2.0.6').replace('WINDS', 'GATES_WINDS')
        elif subregion == 'WP':
            for i in range(len(file_names)):
                file_names[i] = file_names[i].replace('NH_AT_v2.0.2', 'NH_WP_v2.0.6').replace('WINDS', 'GATES_WINDS')
    elif region == 'SOUTHERN_HEMISPHERE':
        if subregion == 'AU':
            for i in range(len(file_names)):
                file_names[i] = file_names[i].replace('NH_AT_v2.0.2', 'SH_AU_v2.0.3').replace('WINDS', 'GATES_WINDS')
        elif subregion == 'AI':
            for i in range(len(file_names)):
                file_names[i] = file_names[i].replace('NH_AT_v2.0.2', 'SH_AI_v2.0.3').replace('WINDS', 'GATES_WINDS')
        elif subregion == 'PI':
            for i in range(len(file_names)):
                file_names[i] = file_names[i].replace('NH_AT_v2.0.2', 'SH_PI_v2.0.3').replace('WINDS', 'GATES_WINDS')
    return file_names


def get_subregion(region):
    if region =='PACIFIC':
        subregion = ['EP','WP'] # EP OR WP
    elif region =='SOUTHERN_HEMISPHERE':
        subregion = ['PI', 'AU', 'AI'] 
    elif region =='NORTH_ATLANTIC':
        subregion = ['AT'] 
    elif region =='NORTH_INDIAN':
        subregion = ['NI'] 

    return subregion

def get_trackset_files(warming):

    c_df = pd.read_csv(f'/fast/dev/utc22_storage/WARMING_LEVELS/ONECONCERN_AT_LONGER/WARMING_HOTBIAS_CORRECTION_CESM-LENS2_Selection_CONTINUOUS_for_{warming}C_SampleLevel_200climates.csv')
    c_df = c_df.loc[c_df.YEAR != 2100]
    c_df = c_df[0:100]

    file_names = c_df.PATH.unique()
    file_names = [os.path.basename(file_path) for file_path in file_names]
    file_names_array = np.array(file_names)

    print('No of SEASON, ENSEMBLE combo: ',file_names_array.shape[0])

    return file_names

def get_era5_gate_files(region):

    utcfiles = sorted(glob.glob(os.path.join(f'/slow/nico/utc22/DATA/OFFICIAL_Jan2023/{region}/ERA5/GATES_OFFSHORE', 'GATES*.snp')))

    print('ERA5: total files = ',len(utcfiles))

    return utcfiles
    


def get_gate_files(region,subregion, file_names):

    utcfiles = []

    for subreg in subregion:
        print('region ',subreg)

        file_names_r = rename_filenames(region, subreg, file_names.copy())
            
        for file in file_names_r:
            utcfiles.append(os.path.join(f'/slow/nico/utc22/DATA/OFFICIAL_Jan2023/{region}/CESM-LENS2_CLIMATOLOGY/GATES_OFFSHORE', file))

    print('CESM: total files = ',len(utcfiles))

    return utcfiles


def get_df_gates(utcfiles, gates):

    dfm = pd.DataFrame()

    for utcfile in utcfiles:

        df = pd.read_parquet(utcfile)
        df = df.loc[(df.GATE.isin(gates))]
        dfc = df.groupby(['SID', 'GATE'])['Vm_intersect'].max().reset_index()
        dfm = pd.concat([dfm, dfc])

    return dfm

def EP_at_a_gate(region= 'NORTH_ATLANTIC',
                 compare = 'ENSO',
                 gates = [26],
                 ):
    
    warming_levels = ['3.00','4.00']

    subregion = get_subregion(region)

    dfw_mas = pd.DataFrame()

    utcfiles = get_era5_gate_files(region)

    # print(utcfiles)

    utcfiles_epoch1 = [file for file in utcfiles if int(file.split('_')[10]) <= 1998]
    utcfiles_epoch2 = [file for file in utcfiles if int(file.split('_')[10]) > 1999]

    utcfiles_elnino = [file for file in utcfiles if int(file.split('_')[10]) in NINO_NH]
    utcfiles_lanina = [file for file in utcfiles if int(file.split('_')[10]) in NINA_NH]

    df_era5 = get_df_gates(utcfiles, gates)
    df_era5['model'] = 'ERA5 [1980-2020]'

    df_era5_elnino = get_df_gates(utcfiles_elnino, gates)
    df_era5_elnino['model'] = 'ERA5 El-Nino'

    df_era5_lanina = get_df_gates(utcfiles_lanina, gates)
    df_era5_lanina['model'] = 'ERA5 La-Nina'

    df_era5_amon = get_df_gates(utcfiles_epoch1, gates)
    df_era5_amon['model'] = 'ERA5 AMO[-] [1980-1998]'

    df_era5_amop = get_df_gates(utcfiles_epoch2, gates)
    df_era5_amop['model'] = 'ERA5 AMO[+] [1999-2020]'


    for lvl in warming_levels:

        print('warming level: ',lvl)

        file_names = get_trackset_files(lvl)

        utcfiles = get_gate_files(region, subregion, file_names)

        dfw = get_df_gates(utcfiles, gates)

        dfw['model'] = 'CESM_' + lvl + 'degC'

        dfw_mas = pd.concat([dfw_mas, dfw])

    
    if compare == 'ENSO':
        dfw_mas = pd.concat([df_era5, df_era5_lanina, df_era5_elnino])

    elif compare == 'ClimateChange':
        dfw_mas = pd.concat([df_era5, dfw_mas])
    
    elif compare == 'AMO':
        dfw_mas = pd.concat([df_era5, df_era5_amop, df_era5_amon,])

    
    for gate in gates:

        df = dfw_mas.loc[dfw_mas.GATE == gate]

        Plot_a_Gate(df, 25000, str(gate), what="Vm_intersect")
        # plot_reask_gate(gate)


    

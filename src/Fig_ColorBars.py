#=============================================================
#                       INFORMATION
#=============================================================
"""

  Set of ColorBars to have homegenous plots

--------------------------------------------------------------
"""
__author__      = "Nicolas Bruneau"
__copyright__   = "Copyright 2018, Reask"
__className__   = "Useful.py"

#=============================================================
#                   LOAD STANDARD MODULES
#=============================================================

import os, sys, operator, functools
import numpy as np
import matplotlib.colors   as colors
import matplotlib.pyplot   as plt
import matplotlib.colorbar as colorbar
import matplotlib.colors as mcolors
from   matplotlib.font_manager import FontProperties
font0  = FontProperties(); font = font0.copy(); font.set_weight('bold'); font.set_size('large')
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import cmasher as cmr

#=============================================================
#                      USEFUL FUNCTIONS 
#=============================================================

def hex2rgb( hexColor ):
    hexColor = hexColor.lstrip('#'); hexLen = len(hexColor)
    return tuple(int(hexColor[i:i+hexLen//3], 16)/255. for i in range(0, hexLen, hexLen//3))

def Generate_ColorBar_fromList( rgb_colors ) :
    _RGB_ = { 'red':0, 'green':1, 'blue':2, }
    n_colors = len( rgb_colors )
    spacing = np.linspace( 0., 1., n_colors )
#    spacing = [ 0. ,   0.4,  0.5,   0.6,  1.  ]
#    print spacing
    color_dict = { }
    for keyC, valC in _RGB_.items() :
        color_dict[keyC] = []
        for iC in range(n_colors):
            dum = rgb_colors[iC][valC]
            color_dict[keyC].append( ( spacing[iC], dum, dum ), )
    return color_dict

#=============================================================
#                      COLORBAR CLASS
#=============================================================

class ColorBar( object ) :

  def __init__( self, hex_colors = None, continuous_extreme=False ) :
      if hex_colors is None :
         raise ValueError( 'hex_colors_list is required' )
      if len( hex_colors ) < 4 :
         raise ValueError( 'hex_colors_list need at least 4 colors' )

      self.colors = hex_colors
      self.cmap_under = hex_colors[ 0]
      self.cmap_over  = hex_colors[-1]
      self.rgb_colors = list(map( hex2rgb, hex_colors[1:-1] ))
      if continuous_extreme : self.rgb_colors = list(map( hex2rgb, hex_colors ))
      self.n_colors   = len( self.rgb_colors )

  def STANDARD( self, nSubPoints=256 ):
      cdic = Generate_ColorBar_fromList( self.rgb_colors ) 
      cmap = colors.LinearSegmentedColormap( 'standard_{0}'.format(nSubPoints), cdic, nSubPoints )
      cmap.set_under( color = self.cmap_under )
      cmap.set_over ( color = self.cmap_over  )
      cmap.set_bad  ( color = '0.75' )
      self.cmap_standard = cmap
      return cmap

  def STANDARD_r( self, nSubPoints=256 ):
      cdic = Generate_ColorBar_fromList( self.rgb_colors[::-1] )
      cmap = colors.LinearSegmentedColormap( 'standard_r_{0}'.format(nSubPoints), cdic, nSubPoints )
      cmap.set_under( color = self.cmap_over  )
      cmap.set_over ( color = self.cmap_under )
      cmap.set_bad  ( color = '0.75' )
      self.cmap_standard_r = cmap
      return cmap

  def ANOMALY( self, nSubPoints=256 ):
      self.rgb_colors[ self.n_colors//2 ] = hex2rgb("#D3D3D3")#'#6e7177' )# '##d6d6d6' )
      #self.rgb_colors.insert( self.n_colors//2, hex2rgb('#999999' ) )
      cdic = Generate_ColorBar_fromList( self.rgb_colors )
      cmap = colors.LinearSegmentedColormap( 'anomaly_{0}'.format(nSubPoints), cdic, nSubPoints )
      cmap.set_under( color = self.cmap_under )
      cmap.set_over ( color = self.cmap_over  )
      cmap.set_bad  ( color = '0.' )
      self.cmap_anomaly = cmap
      return cmap

  def ANOMALY_r( self, nSubPoints=256 ):
      self.rgb_colors[ self.n_colors//2 ] = hex2rgb("#D3D3D3")#'#6e7177' )# '##d6d6d6' )
      #self.rgb_colors.insert( self.n_colors//2, hex2rgb('#999999' ) )
      cdic = Generate_ColorBar_fromList( self.rgb_colors[::-1] )
      cmap = colors.LinearSegmentedColormap( 'anomaly_{0}'.format(nSubPoints), cdic, nSubPoints )
      cmap.set_under( color = self.cmap_under )
      cmap.set_over ( color = self.cmap_over  )
      cmap.set_bad  ( color = '0.' )
      self.cmap_anomaly = cmap
      return cmap


  def ANOMALYW( self, nSubPoints=256 ):
      self.rgb_colors[ self.n_colors//2 ] = hex2rgb("#ffffff")#'#6e7177' )# '##d6d6d6' )
      #self.rgb_colors.insert( self.n_colors//2, hex2rgb('#999999' ) )
      cdic = Generate_ColorBar_fromList( self.rgb_colors )
      cmap = colors.LinearSegmentedColormap( 'anomaly_{0}'.format(nSubPoints), cdic, nSubPoints )
      cmap.set_under( color = self.cmap_under )
      cmap.set_over ( color = self.cmap_over  )
      cmap.set_bad  ( color = '0.75' )
      self.cmap_anomaly = cmap
      return cmap

  def ANOMALYRB( self, nSubPoints=256 ):
      rgb_colors = np.copy( self.rgb_colors )
      rgb_colors = [rgb_colors[1],hex2rgb("#ffffff"),rgb_colors[-1]]
      cdic = Generate_ColorBar_fromList( self.rgb_colors )
      cmap = colors.LinearSegmentedColormap( 'anomaly_{0}'.format(nSubPoints), cdic, nSubPoints )
      cmap.set_under( color = self.cmap_under )
      cmap.set_over ( color = self.cmap_over  )
      cmap.set_bad  ( color = '0.75' )
      self.cmap_anomaly = cmap
      return cmap


  def EACHSIDE( self, nSubPoints=256 ):
      limit = '#6e7177'; limit = '#ffffff'
      self.rgb_colors[ self.n_colors//2 ] = hex2rgb(limit )# '##d6d6d6' )
      self.rgb_colors_neg = self.rgb_colors[ 0:self.n_colors//2+1 ]
      self.rgb_colors_pos = self.rgb_colors[ self.n_colors//2:: ]

      cdic = Generate_ColorBar_fromList( self.rgb_colors_neg )
      cmap = colors.LinearSegmentedColormap( 'negative_{0}'.format(nSubPoints), cdic, nSubPoints )
      cmap.set_under( color = self.cmap_under )
      cmap.set_over ( color =  hex2rgb(limit ) )
      cmap.set_bad  ( color = '0.75' )
      self.cmap_negative = cmap

      cdic1 = Generate_ColorBar_fromList( self.rgb_colors_pos )
      cmap1 = colors.LinearSegmentedColormap( 'positive_{0}'.format(nSubPoints), cdic1, nSubPoints )
      cmap1.set_under( color = limit )
      cmap1.set_over ( color = self.cmap_over  )
      cmap1.set_bad  ( color = '0.75' )
      self.cmap_positive = cmap1

      return cmap, cmap1

  def EACHSIDE_r( self, nSubPoints=256 ):
      limit = '#6e7177'; limit = '#ffffff'
      self.rgb_colors[ self.n_colors//2 ] = hex2rgb(limit )# '##d6d6d6' )
      self.rgb_colors_neg = self.rgb_colors[ 0:self.n_colors//2+1 ][::-1]
      self.rgb_colors_pos = self.rgb_colors[ self.n_colors//2:: ][::-1]

      cdic = Generate_ColorBar_fromList( self.rgb_colors_neg )
      cmap = colors.LinearSegmentedColormap( 'negative_{0}'.format(nSubPoints), cdic, nSubPoints )
      cmap.set_over( color = self.cmap_under )
      cmap.set_under( color =  hex2rgb(limit ) )
      cmap.set_bad  ( color = '0.75' )
      self.cmap_negative = cmap

      cdic1 = Generate_ColorBar_fromList( self.rgb_colors_pos )
      cmap1 = colors.LinearSegmentedColormap( 'positive_{0}'.format(nSubPoints), cdic1, nSubPoints )
      cmap1.set_over( color = limit )
      cmap1.set_under ( color = self.cmap_over  )
      cmap1.set_bad  ( color = '0.75' )
      self.cmap_positive = cmap1

      return cmap, cmap1


  def Plot_ColorBar( self ) :
      
      fig = plt.figure(figsize=(8,3))
      ax1 = fig.add_axes([0.05, 0.7, 0.9, 0.15])
      norm = colors.Normalize( vmin=5, vmax=30 )
      cb1  = colorbar.ColorbarBase( ax1, cmap=self.cmap_standard, norm=norm, orientation='horizontal', extend='both' )
      cb1.set_label( 'Standard' )

      try :
        ax1 = fig.add_axes([0.05, 0.30, 0.9, 0.15])
        norm = colors.Normalize( vmin=-5, vmax=5 )
        cb1  = colorbar.ColorbarBase( ax1, cmap=self.cmap_anomaly, norm=norm, orientation='horizontal', extend='both' )
        cb1.set_label( 'Anomaly' )
      except : print ("no Anomaly bar")

      plt.show()


#=============================================================
#                  REASK COLORBAR CLASS
#============================================================

ReaskColors = { "DarkBlue"  : "#000025", \
                "MarineBlue": "#2f3290", \
                "DarkCyan"  : "#099fdd", \
                "Green"     : "#00b256", \
                "LimeGreen" : "#80d941", \
                "Lime"      : "#bff037", \
                "Yellow"    : "#e1ee1c", \
                "PinkRed"   : "#ff875b", \
                "FlashyRed" : "#ff2b55", \
                "Grey"      : '0.75'   , \
                "DarkGrey"  : '0.45'     }

##      Dark Blue, Marine Bl, dark cyan, Green    , Lime Gree, Lime     , Yellow   , PinkRed , FlashyRed
officialBGY = ["#000025", "#2f3290", "#099fdd", "#00b256", "#80d941", "#bff037", "#e1ee1c","#ff875b", "#ff2b55" ]
ReaskBar   = ColorBar( hex_colors = officialBGY, continuous_extreme = True  )


officialBGY = ["#2f3290", "#099fdd", "#00b256", "#80d941", "#bff037", "#e1ee1c","#ff875b", "#ff2b55" ]
ReaskBar_Catg   = ColorBar( hex_colors = officialBGY, continuous_extreme = True  )

#ReaskBarD  = ColorBar( hex_colors = BGY, continuous_extreme = False )
#BGYnoR = [ "#000025", "#2f3290", "#099fdd", "#00b256", "#80d941", "#bff037", "#e1ee1c" ]
#ReaskBar_noR  = ColorBar( hex_colors = BGYnoR, continuous_extreme = True  )
#ReaskGreen = ColorBar( hex_colors = [ "#000025", "#2f3290", "#099fdd", "#00b256" ],continuous_extreme = True )

BLUish = [ "#2f3290", "#2d5aad", "#248ac9", "#099fdd", ]
ReaskBlue = ColorBar( hex_colors = BLUish,continuous_extreme = True )


BGYBR = [ "#2f3290", "#099fdd" , "#00b256", "#bff037", "#ff2b55" ]
ReaskBarBR = ColorBar( hex_colors = BGYBR, continuous_extreme = True  )

GOOD_BAD = [ "#ff2b55", "#ff875b", "#707070", "#00b256", "#018f35" ]
GOOD_BAD = [ "#ff2b55", "#ff875b", "#707070", "#1ed655", "#05e648" ]
GOOD_BAD = [ "#ff875b", "#ff875b",  "#707070",  "#05e648" ,"#05e648" ,]
ReaskBarGB = ColorBar( hex_colors = GOOD_BAD, continuous_extreme = True  )


CONFU = [ "#2f3290", "#099fdd", "#ff875b", "#ff2b55" ]
ReaskBarCONFUSION =  ColorBar( hex_colors = CONFU, continuous_extreme = True  )


mycolours = ["#000025", "#2f3290","#099fdd", "#bff037","#e1ee1c", "#ff875b", "#ff2b55"] # yellow pink red
ReaskBarBWR=  ColorBar( hex_colors = mycolours, continuous_extreme = True  )

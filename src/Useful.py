#=============================================================
#                       INFORMATION
#=============================================================
"""

  Set of Useful functions and classes


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
import glob
import pandas as pd 
from scipy import stats
#=============================================================
#                   INTERNAL USEFUL FCTNS 
#=============================================================

def ErrorDisplay( mymessage ) :
    print("\n=================================")
    print( mymessage )
    print("=================================\n")
    sys.exit()

def get_extension( filename, nchar=3 ) :
    return filename[-nchar::]

def write_trk_file( myfile, trks ) :
    mytype = get_extension(myfile)
    if   mytype == 'hdf' : trks.to_hdf( myfile, key='trk', mode='w', complevel=7, complib='zlib' )
    elif mytype == 'csv' : trks.to_csv( myfile, index=False, float_format = '%.5f'  )
    elif mytype == 'snp' : trks.to_parquet( myfile )   ## parquet snappy
    else : ErrorDisplay( "csv or hdf or snp file accepted (given {0})".format(mytype) )

def read_trk_file( myfile ) :
    mytype = get_extension(myfile)
    if   mytype == 'hdf' : dum = pd.read_hdf( myfile, 'trk')
    elif mytype == 'csv' : dum = pd.read_csv( myfile )
    elif mytype == 'snp' : dum = pd.read_parquet( myfile )
    else : ErrorDisplay( "csv or hdf or snp file accepted (given {0})".format(mytype) )
    return dum

def multDim( datadim ) : 
    'Multiply elements of a tuple / list'
    return functools.reduce( operator.mul, datadim, 1 )

def FlattenArray( data ) :
    'Convert any array size into 2D array flattening 1, ...'
    dim = data.shape
    return np.reshape( data, [dim[0],multDim(dim[1::])] )

def FlattenArray_KeepLast( data ) :
    'Convert any array size into 2D array flattening 1, ...'
    dim = data.shape
    return np.reshape( data, [dim[0],multDim(dim[1:-1]),dim[-1]] )

def FullArray( data, dim ) :
    'Convert any flatten array back to its original size'
    return np.reshape( data, dim )

def Check( data ) :
    if len(data.shape) <= 1 : raise ValueError( 'Data needs to have at least 2 dimensions' )
    if len(data.shape) >= 3 : dataV = FlattenArray( data )
    return dataV

help_dt = np.vectorize(lambda x : x.total_seconds() )

def Create_Directory( directory ) :
    if not os.path.exists( directory ) :
       os.makedirs(directory)

def logMinMax( logFile, dum, key='Data' ) :
    logFile.write( "\n ---> {3} : {0}, {1:.2f}, {2:.2f}".format( dum.shape, np.nanmin(dum), np.nanmax(dum), key ) )

# Haversine formula to convert degree in distance in km
def haversine(Lon1,Lat1,Lon2,Lat2):
    R = 6378137.00;
    dLat = (Lat2-Lat1)*np.pi/180.;
    dLon = (Lon2-Lon1)*np.pi/180.;
    Lat1 = Lat1*np.pi/180.;
    Lat2 = Lat2*np.pi/180.;
    a = np.sin(dLat/2.) * np.sin(dLat/2.) + np.sin(dLon/2.) * np.sin(dLon/2.) * np.cos(Lat1) * np.cos(Lat2);
    c = 2 * np.arctan2(np.sqrt(a),np.sqrt(1-a));
    return  R*c/1000.

def bearing(Lon1,Lat1,Lon2,Lat2):
    dLat = (Lat2-Lat1)*np.pi/180.;
    dLon = (Lon2-Lon1)*np.pi/180.;
    Lat1 = Lat1*np.pi/180.;
    Lat2 = Lat2*np.pi/180.;
    aaaa = np.sin(dLon) * np.cos(Lat2)
    bbbb = np.cos(Lat1) * np.sin(Lat2) - np.sin(Lat1) * np.cos(Lat2) * np.cos(dLon)
    c = np.arctan2( aaaa, bbbb )
    return (c * 180. / np.pi + 360)%360

def mydistance(x):
    y = x.shift()
    return haversine(x['LON'], x['LAT'], y['LON'], y['LAT']) # .fillna(0)

def mybearing(x):
    y = x.shift()
    return bearing(x['LON'], x['LAT'], y['LON'], y['LAT'])




def pow_round(x):
    return 10**(floor(log(x,10)-log(0.5,10)))

def CeilValues( val, norder=None ) :
    if norder is None :
       norder = pow_round( np.abs(val) )
       return ceil( val  / norder ) * norder, norder
    else :
       return ceil( val  / norder ) * norder

def RoundValues( val, norder=None ) :
    if norder is None :
       norder = pow_round( np.abs(val) )
       return round( val  / norder, 1 ) * norder, norder
    else :
       return round( val  / norder, 1 ) * norder


#=============================================================
#   MANIPULATE DICTIONARY
#=============================================================

def inv_dictionary( mydict ) : 
    inv_dict = {}
    for k,v in mydict.items():
        for x in v:
            inv_dict.setdefault(x,[]).append(k)
    return inv_dict  

#=============================================================
#   ROTATE PERIODIC VARIABLE TO DIFFERENT BOUND
#=============================================================

def periodic( x, xmin=0, period=360 ):
    if not isinstance( x, np.ndarray): x=np.array(x)
    xmax = xmin+period
    dx = xmax-xmin
    #dd = stats.mode(np.diff(x))[0]
    y = x%dx
    if not isinstance( y, np.ndarray): y=np.array(y)
    ind = y <xmin; y[ind] = period+y[ind]
    ind = y>=xmax; y[ind] = y[ind]-period
    return y

def monobound( x ) :
    dum = np.diff(x)
    dx  = stats.mode(dum)[0]
    if x[ 1]-x[ 0] < 0 : x[ 0] = x[ 1]-dx
    if x[-1]-x[-2] < 0 : x[-1] = x[-2]+dx
    return x


def get_lat_label(lat) :
    if lat < 0 : lab = "{0:.1f}S".format( np.abs(lat) )
    else :       lab = "{0:.1f}N".format( np.abs(lat) )
    return lab
  

#=============================================================
#                         SCALER
#=============================================================

class SCALER1( object ) :

  def __init__( self, Data = None ) :
      '''
      Remove mean and std normalisation
      '''

      ## Checks
      if Data is None         : raise ValueError( 'Data field is required' )
      Data = np.array(Data); self.dim = Data.shape;
      Data = Check( Data )

      ## Set-Up Scaling
      self._mean_ = np.nanmean( Data, axis=0 )
      self._std_  = np.nanstd ( Data, axis=0 )
      self._std_[(self._std_==0)] = 1.

      ## WHERE SCALING IS NAN
      ind = np.isnan(self._std_)
      self._std_ [ind] = 1.
      self._mean_[ind] = 0.

  def Decode( self, data ) :
      'PCA to Estimated Data'
      dataV = Check( data )
      dum = dataV * self._std_ + self._mean_ 
      return np.reshape( dum, (data.shape[0],)+self.dim[1::] )

  def Encode( self, data ) :
      'Raw Data to Scaled Data per features (second dimension)'
      dataV = Check( data )
      dum = (dataV-self._mean_) / self._std_
      #dum[np.isnan(dum)] = -999
      return np.reshape( dum, (data.shape[0],)+self.dim[1::] )

#=============================================================
#https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
     
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError( "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError( "Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError( "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

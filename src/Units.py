#=============================================================
#                       INFORMATION
#=============================================================
"""

  Unit conversion 


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
from scipy import stats

#=============================================================
#                   INTERNAL USEFUL FCTNS 
#=============================================================

## speed
kt_to_ms  = np.vectorize( lambda x: x * 0.514444 )           ## from knot to m/s
ms_to_kt  = np.vectorize( lambda x: x * 1.94384  )           ## from m/s to knot
mph_to_ms = np.vectorize( lambda x: x * 0.44704  )           ## from mph to m/s 
ms_to_mph = np.vectorize( lambda x: x / 0.44704  )           ## from m/s to mph

# angles
deg_to_rad = np.vectorize( lambda x: x * np.pi / 180. )     ## from degrees to radian
rad_to_deg = np.vectorize( lambda x: x * 180. / np.pi )     ## from radian to degrees


#=============================================================
#                    CYCLONE DEFINITION
#=============================================================

TC_cat = pd.DataFrame( { "Cat_ID"      : [    0 ,    1 ,    2 ,    3 ,    4 ,    5  ], \
                         "Cat_Name"    : ["Cat0","Cat1","Cat2","Cat3","Cat4","Cat5" ], \
                         "Vm_Lbnd"     : [   18.,   33.,   43.,   50.,   58.,   70. ], \
                         "Vm_Rbnd"     : [   33.,   43.,   50.,   58.,   70., 1000. ], \
                         "Vm_mph_Lbnd" : [   39.,   74.,   96.,  111.,  130.,  157. ], \
                         "Vm_mph_Rbnd" : [   74.,   96.,  111.,  130.,  157., 1000. ], \
                         "Cp_hPa_Lbnd" : [  990.,  980.,  965.,  945.,  920.,  800. ], \
                         "Cp_hPa_Rbnd" : [ 1010.,  990.,  980.,  965.,  945.,  920. ], \
                     } ) 

TC_cat[ "Vm_kt_Lbnd" ] = ms_to_kt( TC_cat[ "Vm_Lbnd" ] )
TC_cat[ "Vm_kt_Rbnd" ] = ms_to_kt( TC_cat[ "Vm_Rbnd" ] )

TC_cat[ "Vm_ms_Lbnd" ] = mph_to_ms( TC_cat[ "Vm_mph_Lbnd" ] )
TC_cat[ "Vm_ms_Rbnd" ] = mph_to_ms( TC_cat[ "Vm_mph_Rbnd" ] )
 
mycat_Vm   =  TC_cat[ 'Vm_ms_Lbnd'  ] .values.tolist() + [TC_cat[ 'Vm_ms_Rbnd'  ].values[-1], ]
mycat_Cp   = (TC_cat[ 'Cp_hPa_Rbnd' ] .values.tolist() + [TC_cat[ 'Cp_hPa_Lbnd' ].values[-1], ])
mycat_Vmph =  TC_cat[ 'Vm_mph_Lbnd'  ] .values.tolist() + [TC_cat[ 'Vm_mph_Rbnd'  ].values[-1], ]

#mycat_Vmph[ mycat_Vmph==111] = 112

#print( "\n")
#
#print ( TC_cat[["Cat_Name", "Cp_hPa_Lbnd", "Cp_hPa_Rbnd", "Vm_Lbnd", "Vm_Rbnd", \
#               "Vm_mph_Lbnd", "Vm_mph_Rbnd", "Vm_ms_Lbnd", "Vm_ms_Rbnd"]] )
#print( "\n")




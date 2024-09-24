#=============================================================
#                       INFORMATION
#=============================================================
"""

  Set of statistics / score functions


--------------------------------------------------------------
"""
__author__      = "Nicolas Bruneau"
__copyright__   = "Copyright 2018, Reask"
__className__   = "Stats.py"

#=============================================================
#                   LOAD STANDARD MODULES
#=============================================================

import os, sys, operator, functools
import numpy as np

#=============================================================
#                   INTERNAL USEFUL FCTNS 
#=============================================================

def RMSe( d1, d2, axis=None ) :
    return np.sqrt( np.nanmean( (d1-d2)**2, axis=axis ) )

def BIAS( d1, d2, axis=None ) :
    return np.nanmean( d1-d2, axis=axis )

#=============================================================



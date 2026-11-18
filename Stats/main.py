#!/usr/bin/env python

import cp_functions as cp

data = cp.ReadData('/Users/coltenrodriguez/Desktop/UCSB/Projects/Santa_Barbara_PPTVAA/Grain_Size/Rodriguez_Colten/Grain_Size/PPT_VAA/AC19-01_1of1/AC19-01.xps')

summary_data = cp.Summarize(data)

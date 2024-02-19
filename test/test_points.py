"""
Example script to compute a spider segmented with dragonfly. Make sure to have one folder with all the pngs for all the
slices, containing segmented lens and retina for each eye, plus the 7 cephalothorax markers.
"""
# Sys to add the path
import sys
import os
INTERNAL_PATH = '/spider_ws' + '/ctSpyderFields/ctSpyderFields'
sys.path.append(os.path.expanduser('~') + INTERNAL_PATH)
# sys.path.insert(0, os.path.expanduser('~') + INTERNAL_PATH)

import ctSpyderFields
import numpy as np


path = '../Data/Philaeus-chrysops/'

### IF STARTING FROM PICKLE
PhilaeusChrysops = ctSpyderFields.Spider(workdir=path, voxelsize=0.003)
PhilaeusChrysops.load(filename='PhilaeusChrysops', type='pickle')
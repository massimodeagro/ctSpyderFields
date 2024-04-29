"""
Example script to compute a spider starting from saved pickle.
"""
# Sys to add the path
import sys
import os

import matplotlib.pyplot as plt

import ctSpyderFields.ctSpyderFields as ct
import numpy as np

path =  '/mnt/Archive_NTFS/Drive/Experiments/Spider_ctSpyderFields/Data/PhilaeusChrysops/'
paramspath = '/home/massimodeagro/PycharmProjects/ctSpyderFields/examples/params.yaml'

### IF STARTING FROM PICKLE
GenusSpecies = ct.Spider(workdir=path, voxelsize=0.003, paramspath=paramspath)
GenusSpecies.load(filename='PhilaeusChrysops', type='pickle')
GenusSpecies.spider_SoR = np.linalg.inv(GenusSpecies.head_SoR(plot=False))  # [4, 4] \in SE(3)

GenusSpecies.compute_eyes()
GenusSpecies.from_std_to_head()
GenusSpecies.project_retinas_full(field_mm=150)
GenusSpecies.find_all_fields_contours(stepsizes=[500, 1000, 1000, 1000], tolerances=[500,5000,5000,5000])
GenusSpecies.sphericalCoordinates_compute()
GenusSpecies.sphericalCoordinates_plotSorted()
GenusSpecies.sphericalCoordinates_plotFields()
GenusSpecies.sphericalCoordinates_plotSpans(disc='general')
GenusSpecies.sphericalCoordinates_plotSpans(disc='specific')


# GenusSpecies.plot_pyplot(elements=("FOVoutline"))
#GenusSpecies.plot_matplotlib(elements=("FOVoutline"))
"""
Example script to compute a spider starting from saved pickle.
"""
# Sys to add the path
import sys
import os
INTERNAL_PATH = os.path.expanduser('~') + '/spider_ws' + '/ctSpyderFields'
sys.path.append(INTERNAL_PATH + '/ctSpyderFields')

import matplotlib.pyplot as plt

import ctSpyderFields as ct
import numpy as np

path =  INTERNAL_PATH + '/Data/Philaeus-chrysops/'
paramspath = INTERNAL_PATH + '/examples/params.yaml'

### IF STARTING FROM PICKLE
GenusSpecies = ct.Spider(workdir=path, voxelsize=0.003, paramspath=paramspath)
GenusSpecies.load(filename='Philaeus-chrysops', type='pickle')
GenusSpecies.spider_SoR = np.linalg.inv(GenusSpecies.head_SoR(plot=False))  # [4, 4] \in SE(3)

# # Usage of se2_from_two_points
# p1 = np.array([1, 0])
# p2 = np.array([1, 1])
# test_2d_sor = GenusSpecies.se2_from_two_points(p1, p2)
# # Transform from global to new reference
# p2_local = np.dot(np.linalg.inv(test_2d_sor),(np.array([np.append(p2, 1)]).T))
# print(p2_local[0:2])

GenusSpecies.compute_eyes()
GenusSpecies.from_std_to_head()
GenusSpecies.project_retinas_full(field_mm=150)
GenusSpecies.find_all_fields_contours(stepsizes=[500, 1000, 1000, 1000], tolerances=[500,5000,5000,5000])
GenusSpecies.eyes['AME'].calculate_span2(150, 0.003)
# spans = GenusSpecies.calculate_eyes_spans(field_radius=150)

# GenusSpecies.plot_pyplot(elements=("FOVoutline"))
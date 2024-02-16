"""
Example script to compute a spider segmented with dragonfly. Make sure to have one folder with all the pngs for all the
slices, containing segmented lens and retina for each eye, plus the 7 cephalothorax markers. Always pass a param, even if empty
"""
from ctSpyderFields import ctSpyderFields
import numpy as np


path = '/path/to/images/'
paramspath = '/path/to/params/params.yaml'

labelnames = {'AME': {'Lens': 'Lens_AME', 'Retina': 'Retina_AME'},
                  'ALE': {'Lens': 'Lens_ALE', 'Retina': 'Retina_ALE'},
                  'PME': {'Lens': 'Lens_PME', 'Retina': 'Retina_PME'},
                  'PLE': {'Lens': 'Lens_PLE', 'Retina': 'Retina_PLE'},
                  'Markers': {'center': 'Marker_center',
                              'front': 'Marker_front', 'back': 'Marker_back',
                              'bottom': 'Marker_bottom', 'top': 'Marker_top',
                              'left': 'Marker_left', 'right': 'Marker_right'}}

GenusSpecies = ctSpyderFields.Spider(workdir=path, label_names=labelnames, voxelsize=0.001, paramspath=paramspath)
GenusSpecies.load_all_labels_split(style='color')
GenusSpecies.find_eyes_points(style='color')
GenusSpecies.compute_cephalothorax(style='color')
GenusSpecies.compute_eyes()
GenusSpecies.orient_to_standard()

GenusSpecies.project_retinas_full(field_mm=150)
GenusSpecies.save(filename='GenusSpecies')
GenusSpecies.save('GenusSpecies', type='h5')

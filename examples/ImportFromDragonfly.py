"""
Example script to compute a spider segmented with dragonfly. Make sure to have one folder with all the pngs for all the
slices, containing segmented lens and retina for each eye, plus the 7 cephalothorax markers.
"""
from ctSpyderFields import ctSpyderFields
import numpy as np


path = 'C:/Users/lauren.sumner-rooney/Documents/DragonflyProjects/DragonflyExport/VisualFieldFinder/Philaeus-chrysops/'
phillabelnames = {'AME': {'Lens': 'Philaeus_chrysops-Lenses-AME', 'Retina': 'Philaeus_chrysops-Retinas-AME'},
                  'ALE': {'Lens': 'Philaeus_chrysops-Lenses-ALE', 'Retina': 'Philaeus_chrysops-Retinas-ALE'},
                  'PME': {'Lens': 'Philaeus_chrysops-Lenses-PME', 'Retina': 'Philaeus_chrysops-Retinas-PME'},
                  'PLE': {'Lens': 'Philaeus_chrysops-Lenses-PLE', 'Retina': 'Philaeus_chrysops-Retinas-PLE'},
                  'Markers': {'center': 'Philaeus_chrysops-Markers-center',
                              'front': 'Philaeus_chrysops-Markers-front', 'back': 'Philaeus_chrysops-Markers-back',
                              'bottom': 'Philaeus_chrysops-Markers-bottom', 'top': 'Philaeus_chrysops-Markers-top',
                              'left': 'Philaeus_chrysops-Markers-left', 'right': 'Philaeus_chrysops-Markers-right'}}

PhilaeusChrysops = ctSpyderFields.Spider(workdir=path, dragonfly_label_names=phillabelnames, voxelsize=0.003)
PhilaeusChrysops.dragonfly_load_all_labels()

PhilaeusChrysops.compute_cephalothorax()
PhilaeusChrysops.dragonfly_find_eyes_points()
PhilaeusChrysops.compute_eyes()
PhilaeusChrysops.orient_to_standard()

PhilaeusChrysops.project_retinas(field_mm=150)

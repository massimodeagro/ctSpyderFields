from ctSpyderFields import ctSpyderFields
import numpy as np


path = 'C:/Users/lauren.sumner-rooney/Documents/DragonflyProjects/DragonflyExport/VisualFieldFinder/Philaeus-chrysops/'

PhilaeusChrysops = ctSpyderFields.Spider(workdir=path, voxelsize=0.003)

PhilaeusChrysops.load(filename='Philaeus_chrysops', type='pickle')
PhilaeusChrysops.compute_eyes()
PhilaeusChrysops.orient_to_standard()
PhilaeusChrysops.project_retinas_full(150)

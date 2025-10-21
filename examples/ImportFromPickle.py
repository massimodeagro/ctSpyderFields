
from src.ctSpyderFields import ctSpyderFields

path = 'C:/Users/lauren.sumner-rooney/Documents/DragonflyProjects/DragonflyExport/VisualFieldFinder/Philaeus-chrysops/'

PhilaeusChrysops = ctSpyderFields.Spider(workdir=path, voxelsize=0.003)

PhilaeusChrysops.load(filename='Philaeus_chrysops', type='pickle')
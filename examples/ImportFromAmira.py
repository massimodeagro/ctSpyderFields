import os

import trimesh
from tqdm import tqdm
import cv2
from ctSpyderFields import ctSpyderFields
import numpy as np



path = 'C:/Users/lauren.sumner-rooney/Documents/DragonflyProjects/DragonflyExport/VisualFieldFinder/Philaeus-chrysops/'
phillabelnames = {'AME':{'Lens':'Philaeus_chrysops-Lenses-AME', 'Retina':'Philaeus_chrysops-Retinas-AME'},
                  'ALE':{'Lens':'Philaeus_chrysops-Lenses-ALE', 'Retina':'Philaeus_chrysops-Retinas-ALE'},
                  'PME':{'Lens':'Philaeus_chrysops-Lenses-PME', 'Retina':'Philaeus_chrysops-Retinas-PME'},
                  'PLE':{'Lens':'Philaeus_chrysops-Lenses-PLE', 'Retina':'Philaeus_chrysops-Retinas-PLE'},
                  'Markers':{'center':'Philaeus_chrysops-Markers-center',
                             'front':'Philaeus_chrysops-Markers-front', 'back':'Philaeus_chrysops-Markers-back',
                             'bottom':'Philaeus_chrysops-Markers-bottom', 'top':'Philaeus_chrysops-Markers-top',
                             'left':'Philaeus_chrysops-Markers-left', 'right':'Philaeus_chrysops-Markers-right'}}

PhilaeusChrysops = ctSpyderFields.Spider(workdir=path, dragonfly_label_names=phillabelnames, voxelsize=0.003)

PhilaeusChrysops.load(filename='Philaeus_chrysops', type='pickle')

PhilaeusChrysops.dragonfly_load_all_labels()
PhilaeusChrysops.compute_cephalothorax()
PhilaeusChrysops.dragonfly_find_eyes_points()
PhilaeusChrysops.compute_eyes()
PhilaeusChrysops.orient_to_standard()
PhilaeusChrysops.project_retinas(150)

PhilaeusChrysops.save('Philaeus_chrysops')
PhilaeusChrysops.save('Philaeus_chrysops', type='h5')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1,1,1])
lens = PhilaeusChrysops.AME.StandardOrientationLensCloud.convex_hull.vertices
ax.scatter(lens[:,0],lens[:,1],lens[:,2], color='purple')
retina = PhilaeusChrysops.AME.StandardOrientationRetinaCloud.convex_hull.vertices
ax.scatter(retina[:,0],retina[:,1],retina[:,2], color='purple')
Project = np.array(PhilaeusChrysops.AME.StandardOrientationProjectedVectorsFull)[:,2]
ax.scatter(Project[:,0],Project[:,1],Project[:,2], color='indigo', alpha=0.01)

lens = PhilaeusChrysops.ALE.StandardOrientationLensCloud.convex_hull.vertices
ax.scatter(lens[:,0],lens[:,1],lens[:,2], color='green')
retina = PhilaeusChrysops.ALE.StandardOrientationRetinaCloud.convex_hull.vertices
ax.scatter(retina[:,0],retina[:,1],retina[:,2], color='green')
Project = np.array(PhilaeusChrysops.ALE.StandardOrientationProjectedVectorsFull)[:,2]
ax.scatter(Project[:,0],Project[:,1],Project[:,2], color='darkgreen', alpha=0.01)

lens = PhilaeusChrysops.PME.StandardOrientationLensCloud.convex_hull.vertices
ax.scatter(lens[:,0],lens[:,1],lens[:,2], color='goldenrod')
retina = PhilaeusChrysops.PME.StandardOrientationRetinaCloud.convex_hull.vertices
ax.scatter(retina[:,0],retina[:,1],retina[:,2], color='goldenrod')
Project = np.array(PhilaeusChrysops.PME.StandardOrientationProjectedVectorsFull)[:,2]
ax.scatter(Project[:,0],Project[:,1],Project[:,2], color='darkgoldenrod', alpha=0.01)

lens = PhilaeusChrysops.PLE.StandardOrientationLensCloud.convex_hull.vertices
ax.scatter(lens[:,0],lens[:,1],lens[:,2], color='darkred')
retina = PhilaeusChrysops.PLE.StandardOrientationRetinaCloud.convex_hull.vertices
ax.scatter(retina[:,0],retina[:,1],retina[:,2], color='darkred')
Project = np.array(PhilaeusChrysops.PLE.StandardOrientationProjectedVectorsFull)[:,2]
ax.scatter(Project[:,0],Project[:,1],Project[:,2], color='maroon', alpha=0.01)

plt.show()

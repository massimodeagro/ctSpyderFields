from ctSpyderFields import ctSpyderFields
import numpy as np


path = 'C:/Users/lauren.sumner-rooney/Documents/DragonflyProjects/DragonflyExport/VisualFieldFinder/Philaeus-chrysops/'

PhilaeusChrysops = ctSpyderFields.Spider(workdir=path, voxelsize=0.003)

PhilaeusChrysops.load(filename='Philaeus_chrysops', type='pickle')
PhilaeusChrysops.compute_eyes()
PhilaeusChrysops.orient_to_standard()
PhilaeusChrysops.project_retinas_full(150)
PhilaeusChrysops.find_all_fields_contours(stepsizes=[500, 1000, 1000, 1000], tolerances=[500,5000,5000,5000])

PhilaeusChrysops.plot(elements=['FOVoutline'], field_mm=145)

### Using PLOTLY

import plotly.graph_objects as go

u, v = np.mgrid[0:2 * np.pi:50j, 0:np.pi:50j]
x = 150/0.003 * np.cos(u) * np.sin(v)
y = 150/0.003 * np.sin(u) * np.sin(v)
z = 150/0.003 * np.cos(v)

sphere = go.Surface(x=x, y=y, z=z, opacity=0.7,colorscale=[[0, 'white'], [1,'white']],
             showscale=False)
Outline = PhilaeusChrysops.AME.FOVcontourPoints
dots1 = go.Scatter3d(x=Outline[:,0], y=Outline[:,1], z=Outline[:,2],
                     mode='markers', marker={'color': 'purple', 'size': 3})
Outline = PhilaeusChrysops.ALE.FOVcontourPoints
dots2 = go.Scatter3d(x=Outline[:,0], y=Outline[:,1], z=Outline[:,2],
                     mode='markers', marker={'color': 'green', 'size': 3})
Outline = PhilaeusChrysops.PME.FOVcontourPoints
dots3 = go.Scatter3d(x=Outline[:,0], y=Outline[:,1], z=Outline[:,2],
                     mode='markers', marker={'color': 'goldenrod', 'size': 3})
Outline = PhilaeusChrysops.PLE.FOVcontourPoints
dots4 = go.Scatter3d(x=Outline[:,0], y=Outline[:,1], z=Outline[:,2],
                     mode='markers', marker={'color': 'darkred', 'size': 3})

Outline = PhilaeusChrysops.AME.FOVcontourPoints
dots1rev = go.Scatter3d(x=np.multiply(Outline[:,0], -1), y=Outline[:,1], z=Outline[:,2],
                        mode='markers', marker={'color':'purple', 'size': 3})
Outline = PhilaeusChrysops.ALE.FOVcontourPoints
dots2rev = go.Scatter3d(x=np.multiply(Outline[:,0], -1), y=Outline[:,1], z=Outline[:,2],
                        mode='markers', marker={'color':'green', 'size': 3})
Outline = PhilaeusChrysops.PME.FOVcontourPoints
dots3rev = go.Scatter3d(x=np.multiply(Outline[:,0], -1), y=Outline[:,1], z=Outline[:,2],
                        mode='markers', marker={'color':'goldenrod', 'size': 3})
Outline = PhilaeusChrysops.PLE.FOVcontourPoints
dots4rev = go.Scatter3d(x=np.multiply(Outline[:,0], -1), y=Outline[:,1], z=Outline[:,2],
                        mode='markers', marker={'color':'darkred', 'size': 3})

fig = go.Figure(data=[sphere, dots1, dots2, dots3, dots4, dots1rev, dots2rev, dots3rev, dots4rev])
fig.show()

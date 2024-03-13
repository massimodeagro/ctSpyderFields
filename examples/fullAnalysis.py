"""
Example script to compute a spider starting from saved pickle.
"""
import matplotlib.pyplot as plt

import ctSpyderFields.ctSpyderFields as ct
import numpy as np

path = '/home/massimodeagro/CTspyderFields/ctSpyderFields/Data/'
paramspath = '/home/massimodeagro/CTspyderFields/ctSpyderFields/examples/params.yaml'

### IF STARTING FROM PICKLE
GenusSpecies = ct.Spider(workdir=path, voxelsize=0.003, paramspath=paramspath)
GenusSpecies.load(filename='PhilaeusChrysops', type='pickle')
GenusSpecies.spider_SoR = np.linalg.inv(GenusSpecies.head_SoR(plot=False))  # [4, 4] \in SE(3)

GenusSpecies.compute_eyes()
GenusSpecies.from_std_to_head()
GenusSpecies.project_retinas_full(field_mm=150)
GenusSpecies.find_all_fields_contours(stepsizes=[500, 1000, 1000, 1000], tolerances=[500,5000,5000,5000])
spans = GenusSpecies.calculate_eyes_spans(field_radius=150)

GenusSpecies.plot_pyplot(elements=("FOVoutline"))
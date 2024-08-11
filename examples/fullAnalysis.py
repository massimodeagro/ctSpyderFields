"""
Example script to compute a spider starting from saved pickle.
"""
# Sys to add the path
import ctSpyderFields.ctSpyderFields as Ct
import numpy as np

path = '/home/massimodeagro/CTspyderFields/ctSpyderFields/Data/PhilaeusChrysops/'
paramspath = '/home/massimodeagro/CTspyderFields/ctSpyderFields/examples/params.yaml'

path = '/home/mad/Archive_NTFS/Drive/Experiments/Spider_ctSpyderFields/Data/PhilaeusChrysops/'
paramspath = '/home/mad/PycharmProjects/ctSpyderFields/examples/params.yaml'

# IF STARTING FROM PICKLE
GenusSpecies = Ct.Spider(workdir=path, voxelsize=0.003, paramspath=paramspath)

GenusSpecies.load(filename='PhilaeusChrysops', type='pickle')
GenusSpecies.head_SoR(flipZ=True, plot=True)  # [4, 4] \in SE(3)
GenusSpecies.compute_eyes(focal_point_type='sphere')
GenusSpecies.from_std_to_head()
GenusSpecies.plot_matplotlib(elements=("lens", 'retina'), plot_FOV_sphere=False, field_mm=3)

GenusSpecies.project_retinas_full(field_mm=150)
GenusSpecies.find_all_fields_contours_alphashape([90, 20, 20, 20])
GenusSpecies.plot_matplotlib(elements=("FOVoutline"))
GenusSpecies.sphericalCoordinates_plotFields()

GenusSpecies.sphericalCoordinates_compute(specific_discretization=15, general_discretization=36)
GenusSpecies.sphericalCoordinates_plotFields()


GenusSpecies.binocularOverlap_compute()
GenusSpecies.multiEyeOverlap_compute()
GenusSpecies.sphericalCoordinates_plotSorted()
GenusSpecies.sphericalCoordinates_plotSpans(disc='general')
GenusSpecies.sphericalCoordinates_plotSpans(disc='specific')

GenusSpecies.sphericalCoordinates_save(filename='PhilaeusChrysops')
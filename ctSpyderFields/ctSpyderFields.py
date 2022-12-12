import cv2 #computer vision 2, reads images
import os #looks into files and folder paths
import numpy as np #deals with matrix and arrays
from mpl_toolkits import mplot3d #to plot in 3d
import matplotlib.pyplot as plt #to generally plot
from tqdm import tqdm #to show percentage bars
import pickle
import pandas as pd

import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist, euclidean
from skspatial.objects import Plane, Points
from skspatial.plotting import plot_3d

class UnrecognizedEye(Exception):
    pass

class Eye:
    def __init__(self, eye_identity: str):
        ## Standardized Data
        self.EyeIdentity = eye_identity
        if self.EyeIdentity == 'AME':
            LensColor = (np.array([250,0,250]),np.array([256,0,256]))
            RetinaColor = (np.array([110,0,110]),np.array([130,0,130]))
        elif self.EyeIdentity == 'ALE':
            LensColor = (np.array([0,250,0]),np.array([0,256,0]))
            RetinaColor = (np.array([0,110,0]),np.array([0,130,0]))
        elif self.EyeIdentity == 'PME':
            LensColor = (np.array([0,250,250]),np.array([0,256,256]))
            RetinaColor = (np.array([0,110,110]),np.array([0,130,130]))
        elif self.EyeIdentity == 'PLE':
            LensColor = (np.array([0,0,250]),np.array([0,0,256]))
            RetinaColor = (np.array([0,0,110]),np.array([0,0,130]))
        else:
            raise UnrecognizedEye('you inputted the wrong eye name, abort.')
        self.LensColor = LensColor
        self.RetinaColor = RetinaColor

        ## Slices Stacks
        self.LensMask = [] #This is created by find_lens
        self.RetinaMask = [] #This is created by find_lens

        ## Coordinates
        self.LensPoints = None
        self.RetinaPoints = None
        self.LensCloud = None
        self.RetinaCloud = None

        self.RotatedLensPoints = None
        self.RotatedRetinaPoints = None
        self.RotatedLensCloud = None
        self.RotatedRetinaCloud = None

        self.StandardOrientationLensPoints = None
        self.StandardOrientationRetinaPoints = None
        self.StandardOrientationLensCloud = None
        self.StandardOrientationRetinaCloud = None

        self.StandardOrientationProjectedVectors = []
        self.StandardOrientationProjectedVectorsFull = []


    def amira_find_lens_points(self, labels_pictures_list):
        '''
        this formula takes the list of label pictures and threshold it based on
        the given colours to find lenses
        '''
        for label in tqdm(labels_pictures_list, desc='finding '+ self.EyeIdentity+' Lens'):  # for every slice
            # find pixels with the determed color and set them as 1, all else as 0
            self.LensMask.append(cv2.inRange(label, self.LensColor[0], self.LensColor[1]))

        print('Computing coordinates...')
        self.LensMask = np.array(self.LensMask)
        self.LensPoints = np.argwhere(self.LensMask > 0)

    def amira_find_retinas_points(self, labels_pictures_list):
        '''
        this formula takes the list of label pictures and threshold it based on
        the given colours to find retinas
        '''
        for label in tqdm(labels_pictures_list, desc='finding '+ self.EyeIdentity+' Retina'):  # for every slice
            # find pixels with the determed color and set them as 1, all else as 0
            self.RetinaMask.append(cv2.inRange(label, self.RetinaColor[0], self.RetinaColor[1]))

        print('Computing coordinates...')
        self.RetinaMask = np.array(self.RetinaMask)
        self.RetinaPoints = np.argwhere(self.RetinaMask > 0)


    def amira_find_all_points(self, labels_pictures_list):
        '''
        duh
        '''
        self.amira_find_lens_points(labels_pictures_list)
        self.amira_find_retinas_points(labels_pictures_list)

    def dragonfly_find_points(self, piclist, part='lens'):
        if part == 'Lens':
            self.LensPoints = np.argwhere(np.array(piclist) > 0)
        elif part == 'Retina':
            self.RetinaPoints = np.argwhere(np.array(piclist) > 0)

    def define_lens_cloud(self):
        print('finding '+ self.EyeIdentity+' lens Hull...')
        self.LensCloud = trimesh.points.PointCloud(self.LensPoints)

    def define_retina_cloud(self):
        print('finding '+ self.EyeIdentity+' retina Hull...')
        self.RetinaCloud = trimesh.points.PointCloud(self.RetinaPoints)

    def define_all_clouds(self):
        self.define_lens_cloud()
        self.define_retina_cloud()

    def align_to_zero(self):
        rotationMatrix = self.LensCloud.convex_hull.principal_inertia_transform

        self.RotatedLensPoints = trimesh.transform_points(self.LensPoints, rotationMatrix)
        self.RotatedLensCloud = trimesh.points.PointCloud(self.RotatedLensPoints)
        self.RotatedRetinaPoints = trimesh.transform_points(self.RetinaPoints,rotationMatrix)
        self.RotatedRetinaCloud = trimesh.points.PointCloud(self.RotatedRetinaPoints)

    def find_split_plane(self):
        '''
        this function finds the plane (xy, xz, yz) that divides retina from lens,
        as well as on which side of the two is the retina and on which is the lens.
        This is needed for finding the cap of the lens.
        '''

        LensSpanX = (max(self.RotatedLensCloud.vertices[:, 0]),
                     min(self.RotatedLensCloud.vertices[:, 0]))
        LensSpanY = (max(self.RotatedLensCloud.vertices[:, 1]),
                     min(self.RotatedLensCloud.vertices[:, 1]))
        LensSpanZ = (max(self.RotatedLensCloud.vertices[:, 2]),
                     min(self.RotatedLensCloud.vertices[:, 2]))

        RetinaSpanX = (max(self.RotatedRetinaCloud.vertices[:, 0]),
                       min(self.RotatedRetinaCloud.vertices[:, 0]))
        RetinaSpanY = (max(self.RotatedRetinaCloud.vertices[:, 1]),
                       min(self.RotatedRetinaCloud.vertices[:, 1]))
        RetinaSpanZ = (max(self.RotatedRetinaCloud.vertices[:, 2]),
                       min(self.RotatedRetinaCloud.vertices[:, 2]))

        overlapX = min(LensSpanX[0], RetinaSpanX[0]) - max(LensSpanX[1], RetinaSpanX[1])
        overlapY = min(LensSpanY[0], RetinaSpanY[0]) - max(LensSpanY[1], RetinaSpanY[1])
        overlapZ = min(LensSpanZ[0], RetinaSpanZ[0]) - max(LensSpanZ[1], RetinaSpanZ[1])

        if overlapX != overlapY != overlapZ != overlapX:
            if overlapX < overlapY and overlapX < overlapZ:
                if LensSpanX[0] > RetinaSpanX[0]:
                    mask = (self.RotatedLensCloud.convex_hull.vertices[:, 0] > 0)
                else:
                    mask = (self.RotatedLensCloud.convex_hull.vertices[:, 0] < 0)
            elif overlapY < overlapX and overlapY < overlapZ:
                if LensSpanY[0] > RetinaSpanY[0]:
                    mask = (self.RotatedLensCloud.convex_hull.vertices[:, 1] > 0)
                else:
                    mask = (self.RotatedLensCloud.convex_hull.vertices[:, 1] < 0)
            elif overlapZ < overlapX and overlapZ < overlapY:
                if LensSpanZ[0] > RetinaSpanZ[0]:
                    mask = (self.RotatedLensCloud.convex_hull.vertices[:, 2] > 0)
                else:
                    mask = (self.RotatedLensCloud.convex_hull.vertices[:, 2] < 0)
            self.RotatedLensSurfacePoints = self.RotatedLensCloud.convex_hull.vertices[mask, :]
        else:
            print('cry')
            pass

    def sphere_fit(self, point_cloud):
        """
        script from https://programming-surgeon.com/en/sphere-fit-python/
        input
            point_cloud: xyz of the point clouds　numpy array
        output
            radius : radius of the sphere
            sphere_center : xyz of the sphere center
        """

        A_1 = np.zeros((3, 3))
        # A_1 : 1st item of A
        v_1 = np.array([0.0, 0.0, 0.0])
        v_2 = 0.0
        v_3 = np.array([0.0, 0.0, 0.0])
        # mean of multiplier of point vector of the point_clouds
        # v_1, v_3 : vector, v_2 : scalar

        N = len(point_cloud)
        # N : number of the points

        """Calculation of the sum(sigma)"""
        for v in tqdm(point_cloud):
            v_1 += v
            v_2 += np.dot(v, v)
            v_3 += np.dot(v, v) * v

            A_1 += np.dot(np.array([v]).T, np.array([v]))

        v_1 /= N
        v_2 /= N
        v_3 /= N
        A = 2 * (A_1 / N - np.dot(np.array([v_1]).T, np.array([v_1])))
        b = v_3 - v_2 * v_1
        sphere_center = np.dot(np.linalg.inv(A), b)
        radius = (sum(np.linalg.norm(np.array(point_cloud) - sphere_center, axis=1))
                  / len(point_cloud))

        return (radius, sphere_center)

    def find_lens_sphere(self):
        self.find_split_plane()
        self.RotatedLensSphere = self.sphere_fit(self.RotatedLensSurfacePoints)

    def rotate_back(self):
        rotationMatrix = np.linalg.inv(self.LensCloud.convex_hull.principal_inertia_transform)

        self.LensSphere = (self.RotatedLensSphere[0], trimesh.transform_points([self.RotatedLensSphere[1]], rotationMatrix)[0])

    def orientToStandard(self, rotationMatrix):
        self.StandardOrientationLensPoints = trimesh.transform_points(self.LensPoints, rotationMatrix)
        self.StandardOrientationLensCloud = trimesh.points.PointCloud(self.StandardOrientationLensPoints)
        self.StandardOrientationRetinaPoints = trimesh.transform_points(self.RetinaPoints,rotationMatrix)
        self.StandardOrientationRetinaCloud = trimesh.points.PointCloud(self.StandardOrientationRetinaPoints)
        self.StandardOrientationLensSphere = (self.LensSphere[0], trimesh.transform_points([self.LensSphere[1]], rotationMatrix)[0])

    def project_retina(self, visual_field_radius):
        for point in tqdm(self.StandardOrientationRetinaCloud.convex_hull.vertices, desc='projecting '+self.EyeIdentity+' retina'):
            sx = self.StandardOrientationLensSphere[1][0]
            sy = self.StandardOrientationLensSphere[1][1]
            sz = self.StandardOrientationLensSphere[1][2]

            rx = point[0]
            ry = point[1]
            rz = point[2]
            v = (sx-rx, sy-ry,sz-rz)
            vmag = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
            vu = (v[0]/vmag, v[1]/vmag, v[2]/vmag)

            """Ray/sphere intersection, from https://gist.github.com/thegedge/4769985"""
            vuflip = np.multiply(vu,-1) #I have no idea why, but data is wrong with the vector as calculated above, needs to be pointing in the other direction
            r=visual_field_radius
            dDotR0 = np.dot(vuflip, point)
            t = -dDotR0 - (dDotR0 * dDotR0 - np.dot(point, point) + r * r) ** 0.5

            self.StandardOrientationProjectedVectors.append([point, vuflip, (point[0] + t * vuflip[0], point[1] + t * vuflip[1], point[2] + t * vuflip[2])])

    def project_retina_full(self, visual_field_radius):
        for point in tqdm(self.StandardOrientationRetinaCloud.vertices, desc='projecting '+self.EyeIdentity+' retina'):
            sx = self.StandardOrientationLensSphere[1][0]
            sy = self.StandardOrientationLensSphere[1][1]
            sz = self.StandardOrientationLensSphere[1][2]

            rx = point[0]
            ry = point[1]
            rz = point[2]
            v = (sx-rx, sy-ry,sz-rz)
            vmag = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
            vu = (v[0]/vmag, v[1]/vmag, v[2]/vmag)

            """Ray/sphere intersection, from https://gist.github.com/thegedge/4769985"""
            vuflip = np.multiply(vu,-1) #I have no idea why, but data is wrong with the vector as calculated above, needs to be pointing in the other direction
            r=visual_field_radius
            dDotR0 = np.dot(vuflip, point)
            t = -dDotR0 - (dDotR0 * dDotR0 - np.dot(point, point) + r * r) ** 0.5

            self.StandardOrientationProjectedVectorsFull.append([point, vuflip, (point[0] + t * vuflip[0], point[1] + t * vuflip[1], point[2] + t * vuflip[2])])
        self.StandardOrientationProjectedVectorsFullDownsampled = trimesh.PointCloud(np.array(self.StandardOrientationProjectedVectorsFull)[:,2]).convex_hull.vertices


class Spider:
    def __init__(self, path, dragonfly_label_names=None, voxelsize=0.001):
        self.voxelSize = voxelsize
        self.path = path
        self.AME = Eye(eye_identity='AME')
        self.ALE = Eye(eye_identity='ALE')
        self.PME = Eye(eye_identity='PME')
        self.PLE = Eye(eye_identity='PLE')

        self.cephalothoraxPoints = {'center':[],
                                    'front':[], 'back':[],
                                    'bottom':[], 'top':[],
                                    'left':[], 'right':[]}
        self.StandardOrientationCephalothoraxPoints = {'center':[],
                                    'front':[], 'back':[],
                                    'bottom':[], 'top':[],
                                    'left':[], 'right':[]}

        self.AmiraLabelPictures = []
        self.DragonflyLabelNames = dragonfly_label_names
        self.DragonflyLabelPictures = {'AME':{'Lens':[], 'Retina':[]},
                                       'ALE':{'Lens':[], 'Retina':[]},
                                       'PME':{'Lens':[], 'Retina':[]},
                                       'PLE':{'Lens':[], 'Retina':[]},
                                       'Markers':{'center':[],
                                                  'front':[], 'back':[],
                                                  'bottom':[], 'top':[],
                                                  'left':[], 'right':[]}}

    def amira_load_labels(self):
        for file in tqdm(sorted(os.listdir(self.path)), desc='loading images'):
            self.AmiraLabelPictures.append(cv2.imread(self.path + file, 1))

    def amira_find_all_points(self):
        self.AME.amira_find_all_points(self.AmiraLabelPictures)
        self.ALE.amira_find_all_points(self.AmiraLabelPictures)
        self.PME.amira_find_all_points(self.AmiraLabelPictures)
        self.PLE.amira_find_all_points(self.AmiraLabelPictures)

    def dragonfly_load_label(self, labelname, group, object):
        allpics = os.listdir(self.path)
        imagelist = []
        for file in allpics:
            if file.startswith(labelname):
                imagelist.append(file)
        for file in tqdm(sorted(imagelist), desc='loading '+labelname):
            self.DragonflyLabelPictures[group][object].append(cv2.imread(self.path + file, 0))

    def dragonfly_load_all_labels(self):
        for eye in ['AME', 'ALE', 'PME', 'PLE']:
            for label in self.DragonflyLabelNames[eye]:
                self.dragonfly_load_label(labelname=self.DragonflyLabelNames[eye][label],group=eye,object=label)
        for marker in self.DragonflyLabelPictures['Markers']:
            self.dragonfly_load_label(labelname=self.DragonflyLabelNames['Markers'][marker], group='Markers', object=marker)

    def dragonfly_find_eyes_points(self):
        print('finding AME points...')
        self.AME.dragonfly_find_points(piclist=self.DragonflyLabelPictures['AME']['Lens'], part='Lens')
        self.AME.dragonfly_find_points(piclist=self.DragonflyLabelPictures['AME']['Retina'], part='Retina')
        print('finding ALE points...')
        self.ALE.dragonfly_find_points(piclist=self.DragonflyLabelPictures['ALE']['Lens'], part='Lens')
        self.ALE.dragonfly_find_points(piclist=self.DragonflyLabelPictures['ALE']['Retina'], part='Retina')
        print('finding PME points...')
        self.PME.dragonfly_find_points(piclist=self.DragonflyLabelPictures['PME']['Lens'], part='Lens')
        self.PME.dragonfly_find_points(piclist=self.DragonflyLabelPictures['PME']['Retina'], part='Retina')
        print('finding PLE points...')
        self.PLE.dragonfly_find_points(piclist=self.DragonflyLabelPictures['PLE']['Lens'], part='Lens')
        self.PLE.dragonfly_find_points(piclist=self.DragonflyLabelPictures['PLE']['Retina'], part='Retina')

    def compute_eye(self, eye):
        if eye == 'AME':
            self.AME.define_all_clouds()
            self.AME.align_to_zero()
            self.AME.find_lens_sphere()
            self.AME.rotate_back()
        elif eye == 'ALE':
            self.ALE.define_all_clouds()
            self.ALE.align_to_zero()
            self.ALE.find_lens_sphere()
            self.ALE.rotate_back()
        elif eye == 'PME':
            self.PME.define_all_clouds()
            self.PME.align_to_zero()
            self.PME.find_lens_sphere()
            self.PME.rotate_back()
        elif eye == 'PLE':
            self.PLE.define_all_clouds()
            self.PLE.align_to_zero()
            self.PLE.find_lens_sphere()
            self.PLE.rotate_back()

    def compute_eyes(self):
        self.compute_eye('AME')
        self.compute_eye('ALE')
        self.compute_eye('PME')
        self.compute_eye('PLE')

    def compute_cephalothorax(self):
        allpoints = []
        for point in self.cephalothoraxPoints:
            print('finding '+point+' points...')
            dots = np.argwhere(np.array(self.DragonflyLabelPictures['Markers'][point]) > 0)
            self.cephalothoraxPoints[point] = (np.mean(dots[:,0]),np.mean(dots[:,1]),np.mean(dots[:,2]))
            allpoints.append(self.cephalothoraxPoints[point])
        self.cephalothoraxCloud = trimesh.points.PointCloud(allpoints)

    def orient_to_standard(self):
        rotationMatrix = self.cephalothoraxCloud.convex_hull.principal_inertia_transform
        self.AME.orientToStandard(rotationMatrix)
        self.ALE.orientToStandard(rotationMatrix)
        self.PME.orientToStandard(rotationMatrix)
        self.PLE.orientToStandard(rotationMatrix)
        for point in self.cephalothoraxPoints:
            self.StandardOrientationCephalothoraxPoints[point] = trimesh.transform_points([self.cephalothoraxPoints[point]],rotationMatrix)[0]

    def project_retinas(self, field_mm):
        self.AME.project_retina(field_mm/self.voxelSize)
        self.ALE.project_retina(field_mm/self.voxelSize)
        self.PME.project_retina(field_mm/self.voxelSize)
        self.PLE.project_retina(field_mm/self.voxelSize)

    def project_retinas_full(self, field_mm):
        self.AME.project_retina_full(field_mm/self.voxelSize)
        self.ALE.project_retina_full(field_mm/self.voxelSize)
        self.PME.project_retina_full(field_mm/self.voxelSize)
        self.PLE.project_retina_full(field_mm/self.voxelSize)

    def save(self, filename, type='pickle'):
        '''
        :param filename: not including extension
        :param type: can be pickle, h5, csv
        :return:
        '''

        print('Saving data...')
        ## Coordinates
        data = {'AME':{'Lens': {'Original': self.AME.LensPoints,
                               'Rotated': self.AME.RotatedLensPoints,
                               'Standard': self.AME.StandardOrientationLensPoints},
                       'Retina': {'Original': self.AME.RetinaPoints,
                                 'Rotated': self.AME.RotatedRetinaPoints,
                                 'Standard': self.AME.StandardOrientationRetinaPoints},
                       'Projection': {'Surface':  self.AME.StandardOrientationProjectedVectors,
                                      'Full': self.AME.StandardOrientationProjectedVectorsFull}},
                'ALE': {'Lens': {'Original': self.ALE.LensPoints,
                                 'Rotated': self.ALE.RotatedLensPoints,
                                 'Standard': self.ALE.StandardOrientationLensPoints},
                        'Retina': {'Original': self.ALE.RetinaPoints,
                                   'Rotated': self.ALE.RotatedRetinaPoints,
                                   'Standard': self.ALE.StandardOrientationRetinaPoints},
                        'Projection': {'Surface': self.ALE.StandardOrientationProjectedVectors,
                                       'Full': self.ALE.StandardOrientationProjectedVectorsFull}},
                'PME': {'Lens': {'Original': self.PME.LensPoints,
                                 'Rotated': self.PME.RotatedLensPoints,
                                 'Standard': self.PME.StandardOrientationLensPoints},
                        'Retina': {'Original': self.PME.RetinaPoints,
                                   'Rotated': self.PME.RotatedRetinaPoints,
                                   'Standard': self.PME.StandardOrientationRetinaPoints},
                        'Projection': {'Surface': self.PME.StandardOrientationProjectedVectors,
                                       'Full': self.PME.StandardOrientationProjectedVectorsFull}},
                'PLE': {'Lens': {'Original': self.PLE.LensPoints,
                                 'Rotated': self.PLE.RotatedLensPoints,
                                 'Standard': self.PLE.StandardOrientationLensPoints},
                        'Retina': {'Original': self.PLE.RetinaPoints,
                                   'Rotated': self.PLE.RotatedRetinaPoints,
                                   'Standard': self.PLE.StandardOrientationRetinaPoints},
                        'Projection': {'Surface': self.PLE.StandardOrientationProjectedVectors,
                                       'Full': self.PLE.StandardOrientationProjectedVectorsFull}},
                'cephalothorax':{'Original': self.cephalothoraxPoints,
                                 'Rotated': self.StandardOrientationCephalothoraxPoints}}
        if type=='h5':
            tab = pd.DataFrame(data)
            tab.to_hdf(self.path+filename+'.h5', 'tab')
        elif type=='pickle':
            with open(self.path+filename+'.pickle', 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('Saved')

    def load(self, filename, type='pickle'):
        print('Loading data...')
        '''

        :param filename:
        :param type:
        :return:
        '''
        if type == 'pickle':
            data = pd.read_pickle(self.path+filename+'.pickle')
        elif type == 'h5':
            data = pd.read_hdf(self.path+filename+'.h5')

        self.AME.LensPoints = data['AME']['Lens']['Original']
        self.AME.RotatedLensPoints = data['AME']['Lens']['Rotated']
        self.AME.StandardOrientationLensPoints = data['AME']['Lens']['Standard']
        self.AME.RetinaPoints = data['AME']['Retina']['Original']
        self.AME.RotatedRetinaPoints = data['AME']['Retina']['Rotated']
        self.AME.StandardOrientationRetinaPoints = data['AME']['Retina']['Standard']
        self.AME.StandardOrientationProjectedVectors = data['AME']['Projection']['Surface']
        self.AME.StandardOrientationProjectedVectorsFull = data['AME']['Projection']['Full']  
        self.AME.LensCloud = trimesh.points.PointCloud(self.AME.LensPoints)
        self.AME.RotatedLensCloud = trimesh.points.PointCloud(self.AME.RotatedLensPoints)
        self.AME.StandardOrientationLensCloud = trimesh.points.PointCloud(self.AME.StandardOrientationLensPoints)
        self.AME.RetinaCloud = trimesh.points.PointCloud(self.AME.RetinaPoints)
        self.AME.RotatedRetinaCloud = trimesh.points.PointCloud(self.AME.RotatedRetinaPoints)
        self.AME.StandardOrientationRetinaCloud = trimesh.points.PointCloud(self.AME.StandardOrientationRetinaPoints)
        
        self.ALE.LensPoints = data['ALE']['Lens']['Original']
        self.ALE.RotatedLensPoints = data['ALE']['Lens']['Rotated']
        self.ALE.StandardOrientationLensPoints = data['ALE']['Lens']['Standard']
        self.ALE.RetinaPoints = data['ALE']['Retina']['Original']
        self.ALE.RotatedRetinaPoints = data['ALE']['Retina']['Rotated']
        self.ALE.StandardOrientationRetinaPoints = data['ALE']['Retina']['Standard']
        self.ALE.StandardOrientationProjectedVectors = data['ALE']['Projection']['Surface']
        self.ALE.StandardOrientationProjectedVectorsFull = data['ALE']['Projection']['Full']
        self.ALE.LensCloud = trimesh.points.PointCloud(self.ALE.LensPoints)
        self.ALE.RotatedLensCloud = trimesh.points.PointCloud(self.ALE.RotatedLensPoints)
        self.ALE.StandardOrientationLensCloud = trimesh.points.PointCloud(self.ALE.StandardOrientationLensPoints)
        self.ALE.RetinaCloud = trimesh.points.PointCloud(self.ALE.RetinaPoints)
        self.ALE.RotatedRetinaCloud = trimesh.points.PointCloud(self.ALE.RotatedRetinaPoints)
        self.ALE.StandardOrientationRetinaCloud = trimesh.points.PointCloud(self.ALE.StandardOrientationRetinaPoints)
        
        self.PME.LensPoints = data['PME']['Lens']['Original']
        self.PME.RotatedLensPoints = data['PME']['Lens']['Rotated']
        self.PME.StandardOrientationLensPoints = data['PME']['Lens']['Standard']
        self.PME.RetinaPoints = data['PME']['Retina']['Original']
        self.PME.RotatedRetinaPoints = data['PME']['Retina']['Rotated']
        self.PME.StandardOrientationRetinaPoints = data['PME']['Retina']['Standard']
        self.PME.StandardOrientationProjectedVectors = data['PME']['Projection']['Surface']
        self.PME.StandardOrientationProjectedVectorsFull = data['PME']['Projection']['Full']
        self.PME.LensCloud = trimesh.points.PointCloud(self.PME.LensPoints)
        self.PME.RotatedLensCloud = trimesh.points.PointCloud(self.PME.RotatedLensPoints)
        self.PME.StandardOrientationLensCloud = trimesh.points.PointCloud(self.PME.StandardOrientationLensPoints)
        self.PME.RetinaCloud = trimesh.points.PointCloud(self.PME.RetinaPoints)
        self.PME.RotatedRetinaCloud = trimesh.points.PointCloud(self.PME.RotatedRetinaPoints)
        self.PME.StandardOrientationRetinaCloud = trimesh.points.PointCloud(self.PME.StandardOrientationRetinaPoints)
        
        self.PLE.LensPoints = data['PLE']['Lens']['Original']
        self.PLE.RotatedLensPoints = data['PLE']['Lens']['Rotated']
        self.PLE.StandardOrientationLensPoints = data['PLE']['Lens']['Standard']
        self.PLE.RetinaPoints = data['PLE']['Retina']['Original']
        self.PLE.RotatedRetinaPoints = data['PLE']['Retina']['Rotated']
        self.PLE.StandardOrientationRetinaPoints = data['PLE']['Retina']['Standard']
        self.PLE.StandardOrientationProjectedVectors = data['PLE']['Projection']['Surface']
        self.PLE.StandardOrientationProjectedVectorsFull = data['PLE']['Projection']['Full']
        self.PLE.LensCloud = trimesh.points.PointCloud(self.PLE.LensPoints)
        self.PLE.RotatedLensCloud = trimesh.points.PointCloud(self.PLE.RotatedLensPoints)
        self.PLE.StandardOrientationLensCloud = trimesh.points.PointCloud(self.PLE.StandardOrientationLensPoints)
        self.PLE.RetinaCloud = trimesh.points.PointCloud(self.PLE.RetinaPoints)
        self.PLE.RotatedRetinaCloud = trimesh.points.PointCloud(self.PLE.RotatedRetinaPoints)
        self.PLE.StandardOrientationRetinaCloud = trimesh.points.PointCloud(self.PLE.StandardOrientationRetinaPoints)
        
        self.cephalothoraxPoints = data['cephalothorax']['Original']
        self.StandardOrientationCephalothoraxPoints = data['cephalothorax']['Rotated']

        print('Loaded...')
        

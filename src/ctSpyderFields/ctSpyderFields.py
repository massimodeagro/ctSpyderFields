### Image processing ###
import cv2  # computer vision 2, reads images
from tifffile import imread as tifread #reads images if tiff
import trimesh  # to do 3d geometry
import alphashape
from shapely.geometry import LineString, Point, Polygon
from shapely.affinity import scale
from scipy.spatial import distance, minkowski_distance

### Tools ###
from tqdm import tqdm  # to show percentage bars
import pickle  # to save compressed files
import pandas as pd  # to work with tables
import numpy as np  # deals with matrix and arrays
import os  # looks into files and folder paths

# Storing Hard-Coded Parameters
import yaml

### PLOTTING
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import copy

### Exceptions ###
class UnrecognizedEye(Exception):
    pass

class MissingParams(Exception):
    pass

class WrongCommand(Exception):
    pass

class InvalidDataset(Exception):
    pass

class AlphaShapeError(Exception):
    pass

__version__="1.0.0"

### Classes ###
class Eye:
    """
    This class is responsible for doing operations and containing
    all the data for each eye
    """

    def __init__(self, eye_identity: str, color_params):
        """
        :param eye_identity: one of AME, ALE, PME, PLE
        """
        # Extract Parameters from YAML file
        # Debug

        ## Standardized Data        
        # New Method | More Robust and compact
        if (eye_identity == "AME") or (eye_identity == "ALE") or (eye_identity == "PME") or (eye_identity == "PLE"):
            self.EyeIdentity = eye_identity
            if color_params is not None:
                lensColor = np.array(color_params[eye_identity]["Lens"])
                self.LensColor = (lensColor-5, lensColor+5)
                retinaColor = np.array(color_params[eye_identity]["Retina"])
                self.RetinaColor = (retinaColor-5, retinaColor+5)
            else:
                self.LensColor = None
                self.RetinaColor = None
        else:
           raise UnrecognizedEye("You inputted the wrong eye name, abort.")

        ## Slices Stacks
        self.LensMask = []  # This is created by find_lens
        self.RetinaMask = []  # This is created by find_lens

        ## Coordinates
        self.LensPoints = None
        self.RetinaPoints = None
        self.LensCloud = None
        self.RetinaCloud = None

        self.RotatedLensPoints = None
        self.RotatedRetinaPoints = None
        self.RotatedLensCloud = None
        self.RotatedRetinaCloud = None
        self.RotatedLensSphere = None

        self.StandardOrientationLensPoints = None
        self.StandardOrientationRetinaPoints = None
        self.StandardOrientationLensCloud = None
        self.StandardOrientationRetinaCloud = None

        self.StandardOrientationProjectedVectors = []
        self.StandardOrientationProjectedVectorsFull = []
        self.focalLengths = []
        self.focalLengthsFull = []
        self.FOVcontourPoints = None

        self.spherical_coordinates = {'overlaps': {},
                                      'spherical_points': {},
                                      'alphashape_points': {},
                                      'polygon': {},
                                      'azimuth_max_spans': {},
                                      'elevation_max_spans': {}}


    def find_points(self, piclist, part="Lens", style='binary'):
        if style == 'binary':
            if part == "Lens":
                self.LensPoints = np.argwhere(np.array(piclist) > 0)
            elif part == "Retina":
                self.RetinaPoints = np.argwhere(np.array(piclist) > 0)
        elif style== 'color':
            if self.LensColor is None or self.RetinaColor is None:
                raise MissingParams("no colors provided. can't use color mode")

            if part == "Lens":
                for label in tqdm(piclist, desc="finding " + self.EyeIdentity + " Lens"):  # for every slice
                    # find pixels with the determined color and set them as 1, all else as 0
                    self.LensMask.append(cv2.inRange(label, self.LensColor[0], self.LensColor[1]))
                self.LensMask = np.array(self.LensMask)
                self.LensPoints = np.argwhere(self.LensMask > 0)
            elif part == "Retina":
                for label in tqdm(piclist, desc="finding " + self.EyeIdentity + " Retina"):  # for every slice
                    # find pixels with the determined color and set them as 1, all else as 0
                    self.RetinaMask.append(cv2.inRange(label, self.RetinaColor[0], self.RetinaColor[1]))
                self.RetinaMask = np.array(self.RetinaMask)
                self.RetinaPoints = np.argwhere(self.RetinaMask > 0)

    def define_lens_cloud(self):
        self.LensCloud = trimesh.points.PointCloud(self.LensPoints)

    def define_retina_cloud(self):
        self.RetinaCloud = trimesh.points.PointCloud(self.RetinaPoints)

    def define_all_clouds(self):
        self.define_lens_cloud()
        self.define_retina_cloud()

    def align_to_zero(self):
        """
        This formula rotates both the retina points and the lens points according to the rotation-translation
        matrix found by pointcloud of lens, in order to align everything to the standard axis
        """
        # Notation: # 
        # Rotation Matrix is a 3x3 matrix (SO(3) group) that expresses a Rotation
        # Homogeneous Matrix is a 4x4 matrix (SE(3) group) that expresses a Roto-translation

        hom_matrix = self.LensCloud.convex_hull.principal_inertia_transform
        # principal_inertia_transform maps points in {Camera} frame in {Lens} frame

        self.RotatedLensPoints = trimesh.transform_points(
            self.LensPoints, hom_matrix
        )
        self.RotatedLensCloud = trimesh.points.PointCloud(self.RotatedLensPoints)
        self.RotatedRetinaPoints = trimesh.transform_points(
            self.RetinaPoints, hom_matrix
        )
        self.RotatedRetinaCloud = trimesh.points.PointCloud(self.RotatedRetinaPoints)
        ### The points expressed in SoR {Camera} are expressed now in the {Lens} SoR

    def find_split_plane(self):
        """
        this function finds the plane (xy, xz, yz) that divides retina from lens,
        as well as on which side of the two is the retina and on which is the lens.
        This is needed for finding the cap of the lens.
        """

        # The cap of the lens is the part of the lens convex_hull that is the farthest
        # respect to the Retina

        LensSpanX = (
            max(self.RotatedLensCloud.vertices[:, 0]),
            min(self.RotatedLensCloud.vertices[:, 0]),
        )
        LensSpanY = (
            max(self.RotatedLensCloud.vertices[:, 1]),
            min(self.RotatedLensCloud.vertices[:, 1]),
        )
        LensSpanZ = (
            max(self.RotatedLensCloud.vertices[:, 2]),
            min(self.RotatedLensCloud.vertices[:, 2]),
        )

        RetinaSpanX = (
            max(self.RotatedRetinaCloud.vertices[:, 0]),
            min(self.RotatedRetinaCloud.vertices[:, 0]),
        )
        RetinaSpanY = (
            max(self.RotatedRetinaCloud.vertices[:, 1]),
            min(self.RotatedRetinaCloud.vertices[:, 1]),
        )
        RetinaSpanZ = (
            max(self.RotatedRetinaCloud.vertices[:, 2]),
            min(self.RotatedRetinaCloud.vertices[:, 2]),
        )

        overlapX = min(LensSpanX[0], RetinaSpanX[0]) - max(LensSpanX[1], RetinaSpanX[1])
        overlapY = min(LensSpanY[0], RetinaSpanY[0]) - max(LensSpanY[1], RetinaSpanY[1])
        overlapZ = min(LensSpanZ[0], RetinaSpanZ[0]) - max(LensSpanZ[1], RetinaSpanZ[1])

        if overlapX != overlapY != overlapZ != overlapX:
            if overlapX < overlapY and overlapX < overlapZ:
                if LensSpanX[0] > RetinaSpanX[0]:
                    mask = self.RotatedLensCloud.convex_hull.vertices[:, 0] > 0
                    split_plane = 'x+'
                else:
                    mask = self.RotatedLensCloud.convex_hull.vertices[:, 0] < 0
                    split_plane = 'x-'
            elif overlapY < overlapX and overlapY < overlapZ:
                if LensSpanY[0] > RetinaSpanY[0]:
                    mask = self.RotatedLensCloud.convex_hull.vertices[:, 1] > 0
                    split_plane = 'y+'
                else:
                    mask = self.RotatedLensCloud.convex_hull.vertices[:, 1] < 0
                    split_plane = 'y-'
            elif overlapZ < overlapX and overlapZ < overlapY:
                if LensSpanZ[0] > RetinaSpanZ[0]:
                    mask = self.RotatedLensCloud.convex_hull.vertices[:, 2] > 0
                    split_plane = 'z+'
                else:
                    mask = self.RotatedLensCloud.convex_hull.vertices[:, 2] < 0
                    split_plane = 'z-'

            self.RotatedLensSurfacePoints = self.RotatedLensCloud.convex_hull.vertices[mask, :]
        else:
            print("cry")  # TODO: do a better error catching process
            pass
        return split_plane, [LensSpanX, LensSpanY, LensSpanZ], [RetinaSpanX, RetinaSpanY, RetinaSpanZ]

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

        for v in point_cloud:
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
        radius = sum(
            np.linalg.norm(np.array(point_cloud) - sphere_center, axis=1)
        ) / len(point_cloud)

        return (radius, sphere_center)

    def find_lens_sphere(self, focal_point_type, focal_point_position):
        """
        helper function, call the two previous formulas to do everything in one step
        """
        split_plane, lens_spans, retinas_spans = self.find_split_plane()
        if focal_point_type == 'sphere':
            self.RotatedLensSphere = self.sphere_fit(self.RotatedLensSurfacePoints)
        elif focal_point_type=='given':
            if split_plane == 'x+':
                lens_top, retina_bottom = max(lens_spans[0]), min(retinas_spans[0])
                focal_x = retina_bottom + focal_point_position * (lens_top - retina_bottom)
                focal_y = np.mean(lens_spans[1])
                focal_z = np.mean(lens_spans[2])

            elif split_plane == 'x-':
                lens_top, retina_bottom = min(lens_spans[0]), max(retinas_spans[0])
                focal_x = retina_bottom + focal_point_position * (lens_top - retina_bottom)
                focal_y = np.mean(lens_spans[1])
                focal_z = np.mean(lens_spans[2])

            elif split_plane == 'y+':
                lens_top, retina_bottom = max(lens_spans[1]), min(retinas_spans[1])
                focal_y = retina_bottom + focal_point_position * (lens_top - retina_bottom)
                focal_x = np.mean(lens_spans[0])
                focal_z = np.mean(lens_spans[2])

            elif split_plane == 'y-':
                lens_top, retina_bottom = min(lens_spans[1]), max(retinas_spans[1])
                focal_y = retina_bottom + focal_point_position * (lens_top - retina_bottom)
                focal_x = np.mean(lens_spans[0])
                focal_z = np.mean(lens_spans[2])

            elif split_plane == 'z+':
                lens_top, retina_bottom = max(lens_spans[2]), min(retinas_spans[2])
                focal_z = retina_bottom + focal_point_position * (lens_top - retina_bottom)
                focal_x = np.mean(lens_spans[0])
                focal_y = np.mean(lens_spans[1])

            elif split_plane == 'z-':
                lens_top, retina_bottom = min(lens_spans[2]), max(retinas_spans[2])
                focal_z = retina_bottom + focal_point_position * (lens_top - retina_bottom)
                focal_x = np.mean(lens_spans[0])
                focal_y = np.mean(lens_spans[1])
            radius = abs(lens_top - retina_bottom)

            self.RotatedLensSphere = radius, np.array([focal_x, focal_y, focal_z])

        elif focal_point_type == 'retina_cup':
            if split_plane == 'x+':
                lens_top = np.array([max(lens_spans[0]), np.mean(lens_spans[1]), np.mean(lens_spans[2])])
                retina_bottom = np.array([min(retinas_spans[0]), np.mean(lens_spans[1]), np.mean(lens_spans[2])])
            elif split_plane == 'x-':
                lens_top = np.array([min(lens_spans[0]), np.mean(lens_spans[1]), np.mean(lens_spans[2])])
                retina_bottom = np.array([max(retinas_spans[0]), np.mean(lens_spans[1]), np.mean(lens_spans[2])])
            elif split_plane == 'y+':
                lens_top = np.array([np.mean(lens_spans[0]), max(lens_spans[1]), np.mean(lens_spans[2])])
                retina_bottom = np.array([np.mean(lens_spans[0]), min(retinas_spans[1]), np.mean(lens_spans[2])])
            elif split_plane == 'y-':
                lens_top = np.array([np.mean(lens_spans[0]), min(lens_spans[1]), np.mean(lens_spans[2])])
                retina_bottom = np.array([np.mean(lens_spans[0]), max(retinas_spans[1]), np.mean(lens_spans[2])])
            elif split_plane == 'z+':
                lens_top = np.array([np.mean(lens_spans[0]), np.mean(lens_spans[1]), max(lens_spans[2])])
                retina_bottom = np.array([np.mean(lens_spans[0]), np.mean(lens_spans[1]), min(retinas_spans[2])])
            elif split_plane == 'z-':
                lens_top = np.array([np.mean(lens_spans[0]), np.mean(lens_spans[1]), min(lens_spans[2])])
                retina_bottom = np.array([np.mean(lens_spans[0]), np.mean(lens_spans[1]), max(retinas_spans[2])])

            alldists = []
            distspans = []
            for centerdist in tqdm(np.arange(0.1, 0.9, 0.05), f"finding best focal fit for {self.EyeIdentity}"):
                center = retina_bottom + centerdist * (lens_top - retina_bottom)
                distances = []
                for pt in self.RotatedRetinaPoints:
                    distances.append(distance.euclidean(pt, center))
                alldists.append(distances)
                distspans.append(max(distances)-min(distances))

            bestfit = np.arange(0.1, 0.9, 0.05)[np.argmin(distspans)]
            center = retina_bottom + bestfit * (lens_top - retina_bottom)
            radius = abs(lens_top - retina_bottom)
            self.RotatedLensSphere = radius, np.array(center)

    def get_lens_info(self):
        split_plane, lens_spans, retinas_spans = self.find_split_plane()
        if split_plane == 'x+':
            lens_top = [max(lens_spans[0]), np.mean(lens_spans[1]), np.mean(lens_spans[2])]
            lens_bottom = [min(lens_spans[0]), np.mean(lens_spans[1]), np.mean(lens_spans[2])]
            lens_radius = np.mean([np.mean([abs(lens_spans[1][0]), abs(lens_spans[1][1])]),
                                   np.mean([abs(lens_spans[2][0]), abs(lens_spans[2][1])])])
            lens_vector = (1, 0, 0)

        elif split_plane == 'x-':
            lens_top = [min(lens_spans[0]), np.mean(lens_spans[1]), np.mean(lens_spans[2])]
            lens_bottom = [max(lens_spans[0]), np.mean(lens_spans[1]), np.mean(lens_spans[2])]
            lens_radius = np.mean([np.mean([abs(lens_spans[1][0]), abs(lens_spans[1][1])]),
                                   np.mean([abs(lens_spans[2][0]), abs(lens_spans[2][1])])])
            lens_vector = (-1, 0, 0)

        elif split_plane == 'y+':
            lens_top = [np.mean(lens_spans[0]), max(lens_spans[1]), np.mean(lens_spans[2])]
            lens_bottom = [np.mean(lens_spans[0]), min(lens_spans[1]), np.mean(lens_spans[2])]
            lens_radius = np.mean([np.mean([abs(lens_spans[0][0]), abs(lens_spans[0][1])]),
                                   np.mean([abs(lens_spans[2][0]), abs(lens_spans[2][1])])])
            lens_vector = (0, 1, 0)

        elif split_plane == 'y-':
            lens_top = [np.mean(lens_spans[0]), min(lens_spans[1]), np.mean(lens_spans[2])]
            lens_bottom = [np.mean(lens_spans[0]), max(lens_spans[1]), np.mean(lens_spans[2])]
            lens_radius = np.mean([np.mean([abs(lens_spans[0][0]), abs(lens_spans[0][1])]),
                                   np.mean([abs(lens_spans[2][0]), abs(lens_spans[2][1])])])
            lens_vector = (0, -1, 0)

        elif split_plane == 'z+':
            lens_top = [np.mean(lens_spans[0]), np.mean(lens_spans[1]), max(lens_spans[2])]
            lens_bottom = [np.mean(lens_spans[0]), np.mean(lens_spans[1]), min(lens_spans[2])]
            lens_radius = np.mean([np.mean([abs(lens_spans[1][0]), abs(lens_spans[1][1])]),
                                   np.mean([abs(lens_spans[0][0]), abs(lens_spans[0][1])])])
            lens_vector = (0, 0, 1)

        elif split_plane == 'z-':
            lens_top = [np.mean(lens_spans[0]), np.mean(lens_spans[1]), min(lens_spans[2])]
            lens_bottom = [np.mean(lens_spans[0]), np.mean(lens_spans[1]), max(lens_spans[2])]
            lens_radius = np.mean([np.mean([abs(lens_spans[1][0]), abs(lens_spans[1][1])]),
                                   np.mean([abs(lens_spans[0][0]), abs(lens_spans[0][1])])])
            lens_vector = (0, 0, -1)

        lens_center = [np.mean(lens_spans[0]), np.mean(lens_spans[1]), np.mean(lens_spans[2])]

        self.RotatedLensMarkers = {'top': lens_top, 'bottom':lens_bottom,
                                   'length': distance.euclidean(lens_top, lens_bottom),
                                   'center': lens_center, 'radius': lens_radius,
                                   'pointing_vector': lens_vector}

    def get_retina_info(self):
        pass

    def rotate_back(self):
        """
        rotate the sphere back to the original frame of reference
        """

        #TODO if we add info exctraction, remember to rotate back here

        homMatrix = np.linalg.inv(
            self.LensCloud.convex_hull.principal_inertia_transform
        )
        self.LensMarkers =  {'top': trimesh.transform_points([self.RotatedLensMarkers['top']], homMatrix)[0],
                             'center': trimesh.transform_points([self.RotatedLensMarkers['center']], homMatrix)[0],
                             'bottom': trimesh.transform_points([self.RotatedLensMarkers['bottom']], homMatrix)[0],
                             'radius': self.RotatedLensMarkers['radius'],
                             'length': self.RotatedLensMarkers['length'],
                             'pointing_vector': None}

        v = (self.LensMarkers['top'][0] - self.LensMarkers['center'][0],
             self.LensMarkers['top'][1] - self.LensMarkers['center'][1],
             self.LensMarkers['top'][2] - self.LensMarkers['center'][2])
        vmag = np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        vu = (v[0] / vmag, v[1] / vmag, v[2] / vmag)

        self.LensMarkers['pointing_vector'] = vu
        self.LensSphere = (
            self.RotatedLensSphere[0],
            trimesh.transform_points([self.RotatedLensSphere[1]], homMatrix)[0],
        )

    def orientToStandard(self, hom_matrix):
        print(f'Reorienting {self.EyeIdentity} dots...', end='')
        # Lens
        self.StandardOrientationLensPoints = trimesh.transform_points(self.LensPoints, hom_matrix)
        self.StandardOrientationLensCloud = trimesh.points.PointCloud(self.StandardOrientationLensPoints)

        self.StandardOrientationLensMarkers = {'top': trimesh.transform_points([self.LensMarkers['top']], hom_matrix)[0],
                                               'center': trimesh.transform_points([self.LensMarkers['center']], hom_matrix)[0],
                                               'bottom': trimesh.transform_points([self.LensMarkers['bottom']], hom_matrix)[0],
                                               'radius': self.LensMarkers['radius'],
                                               'length': self.LensMarkers['length'],
                                               'pointing_vector': None}

        v = (self.StandardOrientationLensMarkers['top'][0] - self.StandardOrientationLensMarkers['center'][0],
             self.StandardOrientationLensMarkers['top'][1] - self.StandardOrientationLensMarkers['center'][1],
             self.StandardOrientationLensMarkers['top'][2] - self.StandardOrientationLensMarkers['center'][2])
        vmag = np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        vu = (v[0] / vmag, v[1] / vmag, v[2] / vmag)

        self.StandardOrientationLensMarkers['pointing_vector'] = vu

        # Retina
        self.StandardOrientationRetinaPoints = trimesh.transform_points(self.RetinaPoints, hom_matrix)
        self.StandardOrientationRetinaCloud = trimesh.points.PointCloud(self.StandardOrientationRetinaPoints)
        # Lens Sphere
        self.StandardOrientationLensSphere = (self.LensSphere[0], trimesh.transform_points([self.LensSphere[1]], hom_matrix)[0], )
        print(' Done')

    def project_retina(self, visual_field_radius):
        self.StandardOrientationProjectedVectors = []
        sx = self.StandardOrientationLensSphere[1][0]
        sy = self.StandardOrientationLensSphere[1][1]
        sz = self.StandardOrientationLensSphere[1][2]
        print(f"Projecting {self.EyeIdentity} retina...", end="")

        for point in self.StandardOrientationRetinaCloud.convex_hull.vertices:
            rx = point[0]
            ry = point[1]
            rz = point[2]
            v = (sx - rx, sy - ry, sz - rz)
            vmag = np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            vu = (v[0] / vmag, v[1] / vmag, v[2] / vmag)

            """Ray/sphere intersection, from https://gist.github.com/thegedge/4769985"""
            vuflip = np.multiply(
                vu, -1
            )  # I have no idea why, but data is wrong with the vector as calculated above, needs to be pointing in the other direction
            r = visual_field_radius
            dDotR0 = np.dot(vuflip, point)
            t = -dDotR0 - (dDotR0 * dDotR0 - np.dot(point, point) + r * r) ** 0.5

            self.StandardOrientationProjectedVectors.append(
                [
                    point,
                    vuflip,
                    (
                        point[0] + t * vuflip[0],
                        point[1] + t * vuflip[1],
                        point[2] + t * vuflip[2],
                    ),
                ]
            )
            self.focalLengths.append([point, vmag])
        print(" Done")

    def project_retina_full(self, visual_field_radius):

        print(f"Projecting {self.EyeIdentity} retina...", end="")
        self.StandardOrientationProjectedVectorsFull = []

        # s = sphere's center
        sx = self.StandardOrientationLensSphere[1][0]
        sy = self.StandardOrientationLensSphere[1][1]
        sz = self.StandardOrientationLensSphere[1][2]
        for point in self.StandardOrientationRetinaCloud.vertices:
            rx = point[0]
            ry = point[1]
            rz = point[2]
            # Compute distance between every point of retina with
            # the center of the lens sphere
            v = (rx - sx, ry - sy, rz - sz)
            v_norm = np.linalg.norm(v)
            vu = (v[0] / v_norm, v[1] / v_norm, v[2] / v_norm)

            """Ray/sphere intersection, from https://gist.github.com/thegedge/4769985"""
            dDotR0 = np.dot(vu, point)
            t = -dDotR0 - (dDotR0 * dDotR0 - np.dot(point, point) + visual_field_radius * visual_field_radius) ** 0.5

            self.StandardOrientationProjectedVectorsFull.append([point, vu, (point[0] + t * vu[0], point[1] + t * vu[1], point[2] + t * vu[2], ),])
            self.focalLengthsFull.append([point, v_norm])
        print(' Done')

    def plane_slicer(self, plane, face, stepsize, tolerance):
        points = np.array(self.StandardOrientationProjectedVectorsFull)[:, 2]
        if plane == "XY":
            perpendicular_axis = 2
            slicing_directions = [0, 1]
            span_directions = [1, 0]
        elif plane == "XZ":
            perpendicular_axis = 1
            slicing_directions = [0, 2]
            span_directions = [2, 0]
        elif plane == "YZ":
            perpendicular_axis = 0
            slicing_directions = [1, 2]
            span_directions = [2, 1]
        else:
            raise
        if face == "front":
            toselect = points[:, perpendicular_axis] > 0
            selectedpoints = points[toselect]
        elif face == "back":
            toselect = points[:, perpendicular_axis] < 0
            selectedpoints = points[toselect]
        else:
            raise

        if len(selectedpoints) > 0:
            maxs = []
            mins = []
            for slicing_dir, span_dir in zip(slicing_directions, span_directions):
                span = np.arange(
                    min(selectedpoints[:, slicing_dir]) - 1,
                    max(selectedpoints[:, slicing_dir] + 1),
                    stepsize,
                )
                for start, end in zip(span[:-1], span[1:]):
                    toslice = np.multiply(
                        selectedpoints[:, slicing_dir] > start,
                        selectedpoints[:, slicing_dir] < end,
                    )
                    slice = selectedpoints[toslice]
                    if len(slice) > 0:
                        sliceMax = slice[slice[:, span_dir] == max(slice[:, span_dir])][
                            0
                        ]
                        sliceMin = slice[slice[:, span_dir] == min(slice[:, span_dir])][
                            0
                        ]
                        if not -tolerance < sliceMax[perpendicular_axis] < tolerance:
                            maxs.append(sliceMax)
                        if not -tolerance < sliceMin[perpendicular_axis] < tolerance:
                            mins.append(sliceMin)
            maxs = np.array(maxs)
            mins = np.array(mins)

            return maxs, mins
        else:
            return [], []

    def find_field_contours(self, stepsize, tolerance):
        XYf = self.plane_slicer(
            plane="XY", face="front", stepsize=stepsize, tolerance=tolerance
        )
        XYb = self.plane_slicer(
            plane="XY", face="back", stepsize=stepsize, tolerance=tolerance
        )
        XZf = self.plane_slicer(
            plane="XZ", face="front", stepsize=stepsize, tolerance=tolerance
        )
        XZb = self.plane_slicer(
            plane="XZ", face="back", stepsize=stepsize, tolerance=tolerance
        )
        YZf = self.plane_slicer(
            plane="YZ", face="front", stepsize=stepsize, tolerance=tolerance
        )
        YZb = self.plane_slicer(
            plane="YZ", face="back", stepsize=stepsize, tolerance=tolerance
        )

        tomerge = []
        for face in [XYf, XYb, XZf, XZb, YZf, YZb]:
            if len(face) > 0:
                for side in face:
                    if len(side) > 0:
                        tomerge.append(side)
        self.FOVcontourPoints = np.unique(np.concatenate(tomerge), axis=0)

    def find_field_contours_alphashape(self, alpha, voxelsize, polygons, field_mm=150):

        try:
            points = np.array((self.spherical_coordinates['spherical_points']['azimuth'],
                               self.spherical_coordinates['spherical_points']['elevation'])).T
            alpha_shape = alphashape.alphashape(points, alpha)
            if polygons == 0:
                raise AlphaShapeError('need at least 1 polygon')
            elif polygons == 1:
                border_points = np.array(alpha_shape.exterior.coords)
                self.spherical_coordinates['alphashape_points'] = border_points
            else:
                if len(alpha_shape.geoms) != polygons:
                    raise AlphaShapeError(f'found {len(alpha_shape.geoms)} polygons. expected {polygons}')
                else:
                    self.spherical_coordinates['alphashape_points'] = []
                    for geom in alpha_shape.geoms:
                        border_points = np.array(geom.exterior.coords)
                        self.spherical_coordinates['alphashape_points'].append(border_points)

            self.spherical_coordinates['polygon'] = alpha_shape
            self.spherical_coordinates['polygons_amount'] = polygons

            field_vox = field_mm / voxelsize
            coords = []
            print(f'Calculating {self.EyeIdentity} FOV contour points')

            if self.spherical_coordinates['polygons_amount'] == 1:
                for dot in border_points:
                    spherical_point = [field_vox, dot[0], dot[1]]
                    coords.append(self.spherical2cartesian(spherical_point))
            else:
                for border_points in  self.spherical_coordinates['alphashape_points']:
                    for dot in border_points:
                        spherical_point = [field_vox, dot[0], dot[1]]
                        coords.append(self.spherical2cartesian(spherical_point))
            self.FOVcontourPoints = np.array(coords)
        except AttributeError:
            raise AlphaShapeError('could not generate a single contour. '
                                  'Try changing the alpha value for this eye or specify '
                                  'the expected number of contours')
    def cartesian2spherical(self, cartesian_point):
        """
            Input:
            Cartesian Point: [x, y, z] (list)
            Output:
            Spherical Point: [rho, theta (azimuth), phi (elevation)] (list)
        """
        rho = np.sqrt(cartesian_point[0]*cartesian_point[0] + cartesian_point[1]*cartesian_point[1] + cartesian_point[2]*cartesian_point[2])
        theta = np.arctan2(cartesian_point[1], cartesian_point[0])
        phi = np.arctan2(cartesian_point[2], np.sqrt(cartesian_point[0]*cartesian_point[0] + cartesian_point[1]*cartesian_point[1]))

        return [rho, theta, phi]
    def spherical2cartesian(self, spherical_point):
        """
            Input:
            Spherical Point: [rho, theta (azimuth), phi (elevation)] (list)
            Output:
            Cartesian Point: [x, y, z] (list)
        """
        x = spherical_point[0] * np.cos(spherical_point[1]) * np.cos(spherical_point[2])
        y = spherical_point[0] * np.cos(spherical_point[2]) * np.sin(spherical_point[1])
        z = spherical_point[0] * np.sin(spherical_point[2])

        return [x, y, z]
    
    def pairwise_angle_diff(self, angles):
        n_data = len(angles)
        pairwise_diff = []

        for i in range(n_data):
            for j in range(n_data):
                diff = angles[i] - angles[j]
                pairwise_diff.append([np.arctan2(np.sin(diff), np.cos(diff)),
                                       angles[i], angles[j]])
        return np.array(pairwise_diff)

    def calculate_spherical_coordinates(self, full=False):
        print(f"Calculating {self.EyeIdentity} spherical coordinates...")
        spherical_points = []
        if not full:
            for point in tqdm(self.FOVcontourPoints):
                spherical_points.append(self.cartesian2spherical(list(point)))
        else:
            for point in tqdm(np.array(self.StandardOrientationProjectedVectorsFull)[:, 2]):
                spherical_points.append(self.cartesian2spherical(list(point)))

        spherical_points = np.array(spherical_points)

        self.spherical_coordinates['spherical_points'] = {'azimuth': spherical_points[:,1], 'elevation': spherical_points[:,2]}

    def identify_clusters(self, values, gap_multiplier=5.0):
        # Step 1: Sort the values
        values = np.sort(values)
        # Step 2: Compute gaps
        gaps = np.diff(values)
        if len(gaps) == 0:
            return 1, [np.mean(values)]  # Handle single-value case
        # Step 3: Determine gap threshold
        median_gap = np.median(gaps)  # Typical gap size
        max_gap = np.max(gaps)
        gap_threshold = gap_multiplier * median_gap
        # If the largest gap is not significantly larger than the typical gaps, it's a single cluster
        if max_gap <= gap_threshold:
            return 1, [np.mean(values)]
        # Step 4: Find cluster boundaries (indices where the gap is significant)
        large_gap_indices = np.where(gaps > gap_threshold)[0]
        # Step 5: Create clusters
        clusters = []
        start = 0
        for idx in large_gap_indices:
            clusters.append(values[start:idx + 1])
            start = idx + 1
        clusters.append(values[start:])  # Add the last cluster
        # Step 6: Compute cluster centers
        cluster_centers = [np.mean(cluster) for cluster in clusters]
        return len(clusters), cluster_centers


    def calculate_span4(self, steps=72):
        # Sorting Points in terms of (i) azimuth and (ii) elevation
        azimuth_points = copy.deepcopy(self.spherical_coordinates['spherical_points']['azimuth'])
        azimuth_points.sort()
        az = copy.deepcopy(self.spherical_coordinates['spherical_points']['azimuth'])

        elevation_points = copy.deepcopy(self.spherical_coordinates['spherical_points']['elevation'])
        elevation_points.sort()
        el = copy.deepcopy(self.spherical_coordinates['spherical_points']['elevation'])

        self.spherical_coordinates['azimuth_max_spans'] = {}
        self.spherical_coordinates['elevation_max_spans'] = {}

        for focus in ['azimuth', 'elevation']:
            if focus == 'azimuth':
                # Azimuth range
                azimuth_range = np.linspace(np.deg2rad(-180), np.deg2rad(180), steps)
                ranges = [azimuth_range[:-1], azimuth_range[1:]]
                elevation_max_spans = {'azimuth_range': [], 'span': [], 'extremes': []}
            else:
                # Elevation range
                elevation_range = np.linspace(np.deg2rad(-180), np.deg2rad(180), steps)
                ranges = [elevation_range[:-1], elevation_range[1:]]
                azimuth_max_spans = {'elevation_range': [], 'span': [], 'extremes': []}

            for r in np.array(ranges).T:
                center_angle = np.mean(r)
                if focus == 'azimuth':
                    intersecter_line = LineString([(center_angle, -4), (center_angle, 4)])
                    elevation_max_spans['azimuth_range'].append(r)
                else:
                    intersecter_line = LineString([(-4, center_angle), (4, center_angle)])
                    azimuth_max_spans['elevation_range'].append(r)
                intersect_points = self.spherical_coordinates['polygon'].intersection(intersecter_line)
                if intersect_points.geom_type == 'LineString':
                    intersect_points = np.array(list(intersect_points.xy)).T
                elif intersect_points.geom_type == 'MultiLineString':
                    points = []
                    for i in intersect_points.geoms:
                        for n in i.coords:
                            points.append(n)
                    intersect_points = np.array(points)

                intersection_number = intersect_points.shape[0]

                if intersection_number <= 1:
                    if focus == 'azimuth':
                        elevation_max_spans['span'].append(np.nan)
                        elevation_max_spans['extremes'].append([np.nan, np.nan])
                    else:
                        azimuth_max_spans['span'].append(np.nan)
                        azimuth_max_spans['extremes'].append([np.nan, np.nan])
                else:
                    if focus == 'azimuth':
                        vals = intersect_points[:,1]
                    else:
                        vals = intersect_points[:,0]
                if intersection_number == 2:
                    pairwise_diff = []
                    for i in range(len(vals)):
                        for j in range(len(vals)):
                            diff = vals[i] - vals[j]
                            pairwise_diff.append([diff,
                                                  vals[i], vals[j]])
                    spans = np.array(pairwise_diff)
                    this_spans = spans[np.argmax(spans[:, 0])]
                elif intersection_number >= 3:
                    total_span = 0
                    for first, second in zip(np.sort(vals)[:-1], np.sort(vals)[1:]):
                        #First, check if the FOV is between these two points or not
                        center = (first + second) / 2
                        if focus == 'azimuth':
                            check = Point(center_angle, center)
                        else:
                            check = Point(center, center_angle)
                        if self.spherical_coordinates['polygon'].contains(check):
                            #If the FOV is not between these two points, then we can calculate the span
                            diff = second - first
                            total_span += diff
                    this_spans = [total_span, np.sort(vals)[0], np.sort(vals)[-1]]
                if intersection_number > 1:
                    if focus == 'azimuth':
                        elevation_max_spans['span'].append(this_spans[0])
                        elevation_max_spans['extremes'].append(this_spans[1:])
                    else:
                        azimuth_max_spans['span'].append(this_spans[0])
                        azimuth_max_spans['extremes'].append(this_spans[1:])

        self.spherical_coordinates['azimuth_max_spans'] = azimuth_max_spans
        self.spherical_coordinates['elevation_max_spans'] = elevation_max_spans

    def compute_arbitraryOverlap(self, comparator_identity, other_eye_data):
        if comparator_identity not in self.spherical_coordinates['overlaps']:
            self.spherical_coordinates['overlaps'][comparator_identity] = {}
        overlap = self.spherical_coordinates['polygon'].intersection(other_eye_data['polygon'])
        overlap_percentage = overlap.area / self.spherical_coordinates['polygon'].area
        self.spherical_coordinates['overlaps'][comparator_identity]['overlap_percentage'] = overlap_percentage
        self.spherical_coordinates['overlaps'][comparator_identity]['overlap_object'] = overlap

    def compute_fullfieldOverlap(self, comparator_identity, other_eye_data):
        if comparator_identity not in self.spherical_coordinates['overlaps']:
            self.spherical_coordinates['overlaps'][comparator_identity] = {}
        overlap = self.spherical_coordinates['polygon'].intersection(other_eye_data['polygon'])
        overlap_percentage = overlap.area / other_eye_data['polygon'].area
        self.spherical_coordinates['overlaps'][comparator_identity]['overlap_percentage'] = overlap_percentage
        self.spherical_coordinates['overlaps'][comparator_identity]['overlap_object'] = overlap

    def compute_binocularOverlap(self):
        if 'binocular' not in self.spherical_coordinates['overlaps']:
            self.spherical_coordinates['overlaps']['binocular'] = {}
        other_eye = scale(self.spherical_coordinates['polygon'], xfact=-1, yfact=1, origin=(0,0))
        overlap = self.spherical_coordinates['polygon'].intersection(other_eye)
        overlap_percentage = overlap.area / self.spherical_coordinates['polygon'].area
        self.spherical_coordinates['overlaps']['binocular']['overlap_percentage'] = overlap_percentage
        self.spherical_coordinates['overlaps']['binocular']['overlap_object'] = overlap

  
class Spider:
    """
    This class creates a spider object. this does the full computation and it is the only one you need to use
    the rest is called from here
    """

    def __init__(
        self,
        workdir,
        color_paramspath=None,
        names_paramspath=None,
        voxelsize=0.001,
        available_eyes: list = ["AME", "ALE", "PME", "PLE"],
        eyes_toplot_colors: dict = {'AME': 'purple', 'ALE': 'darkgreen', 'PME': 'darkgoldenrod', 'PLE': 'maroon'}
    ):
        """
        explain here all self

        :param workdir: directory where all the files of the focus spider exist and ONLY those
        :param dragonfly_label_names: not required. specify the filenames for dragonfly different binary images
        :param voxelsize: how many points for mm
        """
        self.voxelSize = voxelsize
        self.path = workdir
        self.available_eyes = available_eyes

        if color_paramspath is not None:
            with open(color_paramspath, 'r') as file:
                self.colors = yaml.safe_load(file)
        else:
            self.colors = None

        if names_paramspath is not None:
            with open(names_paramspath, 'r') as file:
                self.LabelNames = yaml.safe_load(file)
        else:
            self.LabelNames = None

        ## New Version with a Dictionary
        self.eyes = {}
        for eye in self.available_eyes:
            self.eyes[eye] = Eye(eye_identity=eye, color_params=self.colors)

        self.cephalothoraxMarkers = {
            "center": [],
            "front": [],
            "back": [],
            "bottom": [],
            "top": [],
            "left": [],
            "right": [],
        }
        
        self.StandardOrientationCephalothoraxPoints = {
            "center": [],
            "front": [],
            "back": [],
            "bottom": [],
            "top": [],
            "left": [],
            "right": [],
        }

        self.FullLabelPictures = []

        self.SeparateLabelPictures = {
            "AME": {"Lens": [], "Retina": []},
            "ALE": {"Lens": [], "Retina": []},
            "PME": {"Lens": [], "Retina": []},
            "PLE": {"Lens": [], "Retina": []},
            "Markers": {
                "center": [],
                "front": [],
                "back": [],
                "bottom": [],
                "top": [],
                "left": [],
                "right": [],
            },
        }

        self.cephalothoraxCloud = None

        self.spider_SoR = None
        
        self.toplot_colors = eyes_toplot_colors

    def load_label_split(self, labelname, group, object, style='binary'):
        """
        This function pull all the pngs from workdir and load them according to file names
        provided in self.DragonflyLabelNames
        :param labelname: the name of label as provided in self.DragonflyLabelNames
        :param group: can be either an eye or marker
        :param object: lens, retina or one of the 7 markers
        """
        allpics = os.listdir(self.path)
        imagelist = []
        for file in allpics:
            if file.startswith(labelname):
                imagelist.append(file)
        for file in tqdm(sorted(imagelist), desc="loading " + labelname):
            if style=='binary':
                if self.LabelNames is None:
                    raise MissingParams("no label names provided. can't use binary")
                if os.path.splitext(file)[1].lower() in [".tif", ".tiff"]: #handle compressed binary tiffs
                    image = tifread(self.path + file)
                    if image.dtype == bool:
                        image = image.astype('uint8')
                    self.SeparateLabelPictures[group][object].append(image)
                else:
                    self.SeparateLabelPictures[group][object].append(cv2.imread(self.path + file, 0))
            elif style == 'color':
                if self.colors is None:
                    raise MissingParams("no colors param provided. can't use color")
                self.SeparateLabelPictures[group][object].append(cv2.imread(self.path + file, 1))

    def load_all_labels_split(self,  style='binary'):
        """
        This function calls dragonfly_load_label for 4 + 7 times
        """
        for eye in self.available_eyes:
            for label in self.LabelNames[eye]:
                self.load_label_split(
                    labelname=self.LabelNames[eye][label],
                    group=eye,
                    object=label,
                    style=style
                )
        for marker in self.SeparateLabelPictures["Markers"]:
            self.load_label_split(
                labelname=self.LabelNames["Markers"][marker],
                group="Markers",
                object=marker,
                style=style
            )

    def find_eyes_points(self, style='binary'):
        """
        helper function to find all eyes at once. to see how points are found, look in class eyes, function dragonfly_find_points
        """
        # Find Points DragonFly for each eye
        for eye in self.available_eyes:
            print("finding " + eye + " points...")
            for blob in ["Lens", "Retina"]:
                self.eyes[eye].find_points(piclist=self.SeparateLabelPictures[eye][blob], part=blob, style=style)

    def compute_eye(self, eye, focal_point_type='sphere', focal_point_position=0.5):
        """
        helper function to do all the required computation for each eye. look into class eye for each single function

        :param eye: the name of the eye
        :param focal_point_type: can be either sphere or given
        :param focal_point_position: if focal_point_type is given, this is the position of the focal point as relative distance
        between lens and retina
        """
        if eye in self.available_eyes:
            self.eyes[eye].define_all_clouds()
            self.eyes[eye].align_to_zero()
            self.eyes[eye].find_lens_sphere(focal_point_type, focal_point_position)
            self.eyes[eye].get_lens_info()
            self.eyes[eye].rotate_back()
        else:
            raise(UnrecognizedEye("Unrecognized Eye: Computation aborted."))

    def compute_eyes(self, focal_point_type='sphere', focal_point_position=0.5):
        """
        run this! compute all eyes together
        """
        print('Computing lenses and retina geometries...', end='')
        for n, eye in enumerate(self.available_eyes):
            if focal_point_type == 'given_different':
                self.compute_eye(eye, focal_point_type, focal_point_position[n])
            else:
                self.compute_eye(eye, focal_point_type, focal_point_position)
        print(' Done')

    def find_cephalothorax_points(self, style='binary'):
        """
        this first translates the binary pictures in a set of points, and then find the center
        """
        allpoints = []
        for marker in self.cephalothoraxMarkers:
            print("finding " + marker + " points...")
            if style == 'binary':
                dots = np.argwhere(np.array(self.SeparateLabelPictures["Markers"][marker]) > 0)
            elif style == 'color':
                tmpdots = []
                for label in tqdm(self.SeparateLabelPictures["Markers"][marker], desc="finding " + marker + " points"):  # for every slice
                    # find pixels with the determined color and set them as 1, all else as 0
                    tmpdots.append(cv2.inRange(label, np.array(self.colors["Markers"][marker]["low_color"]), np.array(self.colors["Markers"][marker]["high_color"])))
                tmpdots = np.array(tmpdots)
                dots = np.argwhere(tmpdots > 0)
            self.cephalothoraxMarkers[marker] = (
                np.mean(dots[:, 0]),
                np.mean(dots[:, 1]),
                np.mean(dots[:, 2]),
            )
            allpoints.append(self.cephalothoraxMarkers[marker])
        self.cephalothoraxCloud = trimesh.points.PointCloud(allpoints)

        # Compute the SoR of the Head
        # This matrix maps: global (camera) -> local (spider)
        #self.head_SoR()   # [4, 4] \in SE(3)


    #it seems that this now needs dropping
    def orient_to_standard(self):
        hom_matrix = self.cephalothoraxCloud.convex_hull.principal_inertia_transform

        # Rotate each eye
        for eye in self.available_eyes:
            self.eyes[eye].orientToStandard(hom_matrix)

        for marker in self.cephalothoraxMarkers:
            self.StandardOrientationCephalothoraxPoints[marker] = trimesh.transform_points([self.cephalothoraxMarkers[marker]], hom_matrix)[0]
            
    def from_std_to_head(self):
        # Rotate each eye
        for eye in self.available_eyes:
            self.eyes[eye].orientToStandard(self.spider_SoR)

        for marker in self.cephalothoraxMarkers:
            self.StandardOrientationCephalothoraxPoints[marker] = trimesh.transform_points([self.cephalothoraxMarkers[marker]], self.spider_SoR)[0]

    def project_retinas(self, field_mm):
        # Project each retina
        for eye in self.available_eyes:
            self.eyes[eye].project_retina(field_mm / self.voxelSize)

    def project_retinas_full(self, field_mm):
        # Project each retina
        for eye in self.available_eyes:
            self.eyes[eye].project_retina_full(field_mm / self.voxelSize)

    def find_all_fields_contours(
        self, stepsizes=(500, 500, 500, 500), tolerances=(500, 500, 500, 500)
    ):
        """

        :param stepsizes: the size of slices in each direction in pixels. Always need to be a 4 long list, even with less eyes
        :param tolerances: the span from 0 from where to remove points found as contour, they probably are just plane edges
        :return:
        """
        print("Finding fields of view contours...", end="")
        for i in range(len(self.available_eyes)):
            self.eyes[list(self.available_eyes)[i]].find_field_contours(stepsizes[i], tolerances[i])

        print(" Done")        

    def find_all_fields_contours_alphashape(self, alphas=[90,90,90,90],
                                            overwrite=[True, True, True, True],
                                            polygons=[1, 1, 1, 1]):
        """

        :param stepsizes: the size of slices in each direction in pixels. Always need to be a 4 long list, even with less eyes
        :param tolerances: the span from 0 from where to remove points found as contour, they probably are just plane edges
        :return:
        """
        print("Finding fields of view contours. This will take time...")

        for i in range(len(self.available_eyes)):
            if overwrite[i]:
                self.eyes[list(self.available_eyes)[i]].calculate_spherical_coordinates(full=True)
            else:
                if len(self.eyes[list(self.available_eyes)[i]].spherical_coordinates['spherical_points']) > 0:
                    pass
                else:
                    self.eyes[list(self.available_eyes)[i]].calculate_spherical_coordinates(full=True)
            if overwrite[i]:
                self.eyes[list(self.available_eyes)[i]].find_field_contours_alphashape(voxelsize=self.voxelSize,
                                                                                       alpha=alphas[i],
                                                                                       polygons=polygons[i])
            else:
                if len(self.eyes[list(self.available_eyes)[i]].spherical_coordinates['alphashape_points']) > 0:
                    pass
                else:
                    self.eyes[list(self.available_eyes)[i]].find_field_contours_alphashape(voxelsize=self.voxelSize,
                                                                                           alpha=alphas[i],
                                                                                           polygons=polygons[i])

        print("Done")

    def pure_geometrical(self, u, v):
        ### This value is obtained by:    ###
        # w \in null(A)                     #
        # where A = [v.T; cross(u, v).T]    #
        # This condition means that the     #
        # vector w is orthogonal with v     #
        # and coplanar with u.              #
        #####################################
        
        w = []
        # w[0]
        w.append((u[0]*(v[1]**2 + v[2]**2) - v[0]*(u[1]*v[1] + u[2]*v[2]))/(u[2]*(v[0]**2 + v[1]**2) - v[2]*(u[0]*v[0] + u[1]*v[1])))
        # w[1]
        w.append((u[1]*(v[0]**2 + v[2]**2) - v[1]*(u[0]*v[0] + u[2]*v[2]))/(u[2]*(v[0]**2 + v[1]**2) - v[2]*(u[0]*v[0] + u[1]*v[1])))
        # w[2]
        w.append(1.0)
        w = np.array(w)
        
        # Normalizing
        w /= np.linalg.norm(w)
        return w
        
    def head_SoR(self, flipX=False, flipZ=False, plot=False):

        # Read all markers
        n_marker = len(self.cephalothoraxMarkers)
        marker_type = list(self.cephalothoraxMarkers.keys())
        marker_color = ['#323031', '#177E89', '#084C61', '#DB3A34', '#FFC857', '#FF9F1C', '#8ED081']
        
        point_dataset = []


        for i in range(n_marker):
            point_dataset.append(list(self.cephalothoraxMarkers[marker_type[i]]))
        
        point_dataset = np.array(point_dataset)            

        ## Create 3D Rectangle ##
        
        x_hand = np.array(list(self.cephalothoraxMarkers['front'])) - np.array(list(self.cephalothoraxMarkers['back']))
        width = np.linalg.norm(x_hand)
        y_hand = np.array(list(self.cephalothoraxMarkers['left'])) - np.array(list(self.cephalothoraxMarkers['right']))
        depth = np.linalg.norm(y_hand)
        z_hand = np.array(list(self.cephalothoraxMarkers['top'])) - np.array(list(self.cephalothoraxMarkers['bottom']))
        height = np.linalg.norm(z_hand)

        # # Proposal 2: Pure geometrical method
        # Axis 1: back -> front
        x_axis = x_hand
        # x_center = np.array(list(self.cephalothoraxMarkers['back'])) + 0.5*x_hand
        x_axis /= width
        if flipX:
            x_axis = -x_axis
        
        # Orthogonal Axis
        z_hand /= height
        z_axis = self.pure_geometrical(z_hand, x_axis)
        if flipZ:
            z_axis = -z_axis
        
        # Finally, find y by cross product (z cross x)
        y_axis = np.cross(z_axis, x_axis)
        
        # Compose SO(3) group
        R = np.array([x_axis, y_axis, z_axis])
        # Origin as the center marker
        origin = list(self.cephalothoraxMarkers['center'])

        if plot:
            # For each marker, plot a different color
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.view_init(elev=-160, azim=106)
            for i in range(n_marker):
                ax.scatter(self.cephalothoraxMarkers[marker_type[i]][0],
                           self.cephalothoraxMarkers[marker_type[i]][1],
                           self.cephalothoraxMarkers[marker_type[i]][2],
                           color=marker_color[i])
                ax.text(self.cephalothoraxMarkers[marker_type[i]][0],
                        self.cephalothoraxMarkers[marker_type[i]][1],
                        self.cephalothoraxMarkers[marker_type[i]][2],
                        marker_type[i])
                # Plot axes
                # (x) back -> top
                ax.plot([self.cephalothoraxMarkers['back'][0], self.cephalothoraxMarkers['front'][0]],
                        [self.cephalothoraxMarkers['back'][1], self.cephalothoraxMarkers['front'][1]],
                        [self.cephalothoraxMarkers['back'][2], self.cephalothoraxMarkers['front'][2]], 'r')
                # (z) bottom -> top
                ax.plot([self.cephalothoraxMarkers['bottom'][0], self.cephalothoraxMarkers['top'][0]],
                        [self.cephalothoraxMarkers['bottom'][1], self.cephalothoraxMarkers['top'][1]],
                        [self.cephalothoraxMarkers['bottom'][2], self.cephalothoraxMarkers['top'][2]], 'b')
                # (y) right -> left
                ax.plot([self.cephalothoraxMarkers['right'][0], self.cephalothoraxMarkers['left'][0]],
                        [self.cephalothoraxMarkers['right'][1], self.cephalothoraxMarkers['left'][1]],
                        [self.cephalothoraxMarkers['right'][2], self.cephalothoraxMarkers['left'][2]], 'g')

                ax.set_xlabel('X [pixel]')
                ax.set_ylabel('Y [pixel]')
                ax.set_zlabel('Z [n° layer]')
            # Visualizing
            for axis in R:
                ax.quiver(*origin, *axis, length=500)

            plt.show()
                
        # Composing SE(3) group
        R = R.T
        T = np.concatenate((R, np.array([origin]).T), axis=1)
        T = np.concatenate((T, np.array([[0, 0, 0, 1]])), axis=0)

        self.spider_SoR = np.linalg.inv(T)

    def save(self, filename, type="pickle"):
        """
        :param filename: not including extension
        :param type: can be pickle, h5, csv
        :return:
        """

        print("Saving data...")
        ## Coordinates
        data = {
            "AME": {
                "Lens": {
                    "Original": self.eyes["AME"].LensPoints,
                    "Rotated": self.eyes["AME"].RotatedLensPoints,
                    "Standard": self.eyes["AME"].StandardOrientationLensPoints,
                },
                "Retina": {
                    "Original": self.eyes["AME"].RetinaPoints,
                    "Rotated": self.eyes["AME"].RotatedRetinaPoints,
                    "Standard": self.eyes["AME"].StandardOrientationRetinaPoints,
                },
                "Projection": {
                    "Surface": self.eyes["AME"].StandardOrientationProjectedVectors,
                    "Full": self.eyes["AME"].StandardOrientationProjectedVectorsFull,
                },
            },
            "ALE": {
                "Lens": {
                    "Original": self.eyes["ALE"].LensPoints,
                    "Rotated":  self.eyes["ALE"].RotatedLensPoints,
                    "Standard":  self.eyes["ALE"].StandardOrientationLensPoints,
                },
                "Retina": {
                    "Original":  self.eyes["ALE"].RetinaPoints,
                    "Rotated":  self.eyes["ALE"].RotatedRetinaPoints,
                    "Standard":  self.eyes["ALE"].StandardOrientationRetinaPoints,
                },
                "Projection": {
                    "Surface":  self.eyes["ALE"].StandardOrientationProjectedVectors,
                    "Full":  self.eyes["ALE"].StandardOrientationProjectedVectorsFull,
                },
            },
            "PME": {
                "Lens": {
                    "Original": self.eyes["PME"].LensPoints,
                    "Rotated": self.eyes["PME"].RotatedLensPoints,
                    "Standard": self.eyes["PME"].StandardOrientationLensPoints,
                },
                "Retina": {
                    "Original": self.eyes["PME"].RetinaPoints,
                    "Rotated": self.eyes["PME"].RotatedRetinaPoints,
                    "Standard": self.eyes["PME"].StandardOrientationRetinaPoints,
                },
                "Projection": {
                    "Surface": self.eyes["PME"].StandardOrientationProjectedVectors,
                    "Full": self.eyes["PME"].StandardOrientationProjectedVectorsFull,
                },
            },
            "PLE": {
                "Lens": {
                    "Original": self.eyes["PLE"].LensPoints,
                    "Rotated": self.eyes["PLE"].RotatedLensPoints,
                    "Standard": self.eyes["PLE"].StandardOrientationLensPoints,
                },
                "Retina": {
                    "Original": self.eyes["PLE"].RetinaPoints,
                    "Rotated": self.eyes["PLE"].RotatedRetinaPoints,
                    "Standard": self.eyes["PLE"].StandardOrientationRetinaPoints,
                },
                "Projection": {
                    "Surface": self.eyes["PLE"].StandardOrientationProjectedVectors,
                    "Full": self.eyes["PLE"].StandardOrientationProjectedVectorsFull,
                },
            },
            "cephalothorax": {
                "Original": self.cephalothoraxMarkers,
                "Rotated": self.StandardOrientationCephalothoraxPoints,
            },
            "SOR": self.spider_SoR
        }
        
        # Save into a file
        if type == "h5":
            tab = pd.DataFrame(data)
            tab.to_hdf(self.path + filename + ".h5", "tab")
        elif type == "pickle":
            with open(self.path + filename + ".pickle", "wb") as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            raise(InvalidDataset("Invalid extension. Saving process aborted."))

        print("Saved")

    def load(self, filename, type="pickle"):
        print("Loading data...", end="")
        """

        :param filename:
        :param type:
        :return:
        """
        if type == "pickle":
            data = pd.read_pickle(self.path + filename + ".pickle")
        elif type == "h5":
            data = pd.read_hdf(self.path + filename + ".h5")
        else:
            raise InvalidDataset("Invalid extension " + type +  "of the file " + filename + " .")

        for eye in self.available_eyes:
            self.eyes[eye].LensPoints = data[eye]["Lens"]["Original"]
            self.eyes[eye].RotatedLensPoints = data[eye]["Lens"]["Rotated"]
            self.eyes[eye].StandardOrientationLensPoints = data[eye]["Lens"]["Standard"]
            self.eyes[eye].RetinaPoints = data[eye]["Retina"]["Original"]
            self.eyes[eye].RotatedRetinaPoints = data[eye]["Retina"]["Rotated"]
            self.eyes[eye].StandardOrientationRetinaPoints = data[eye]["Retina"]["Standard"]
            self.eyes[eye].StandardOrientationProjectedVectors = data[eye]["Projection"]["Surface"]
            self.eyes[eye].StandardOrientationProjectedVectorsFull = data[eye]["Projection"]["Full"]
            self.eyes[eye].LensCloud = trimesh.points.PointCloud(self.eyes[eye].LensPoints)
            self.eyes[eye].RotatedLensCloud = trimesh.points.PointCloud(self.eyes[eye].RotatedLensPoints)
            self.eyes[eye].StandardOrientationLensCloud = trimesh.points.PointCloud(self.eyes[eye].StandardOrientationLensPoints)
            self.eyes[eye].RetinaCloud = trimesh.points.PointCloud(self.eyes[eye].RetinaPoints)
            self.eyes[eye].RotatedRetinaCloud = trimesh.points.PointCloud(self.eyes[eye].RotatedRetinaPoints)
            self.eyes[eye].StandardOrientationRetinaCloud = trimesh.points.PointCloud(self.eyes[eye].StandardOrientationRetinaPoints)

        self.cephalothoraxMarkers = data["cephalothorax"]["Original"]

        # Switch left & right (human error)
        left_value = self.cephalothoraxMarkers['left']
        self.cephalothoraxMarkers['left'] = self.cephalothoraxMarkers['right']
        self.cephalothoraxMarkers['right'] = left_value

        points = []
        for marker in self.cephalothoraxMarkers:
            points.append(self.cephalothoraxMarkers[marker])
        self.cephalothoraxCloud = trimesh.points.PointCloud(points)
        self.StandardOrientationCephalothoraxPoints = data["cephalothorax"]["Rotated"]

        if 'SOR' in data:
            self.spider_SoR = data['SOR']

        print(" Done")

    def sphericalCoordinates_compute(self, eyes=("AME", "ALE", "PME", "PLE"), full=False):

        # Extract Information
        for eye in eyes:
            if eye in self.available_eyes:
                self.eyes[eye].calculate_spherical_coordinates(full=full)

    def sphericalCoordinates_calculateSpan(self, eyes=("AME", "ALE", "PME", "PLE"),
                                           spans=72):
        for eye in eyes:
            if eye in self.available_eyes:
                self.eyes[eye].calculate_span4(spans)

    def binocularOverlap_compute(self, eyes=("AME", "ALE", "PME", "PLE")):
        for eye in eyes:
            if eye in self.available_eyes:
                self.eyes[eye].compute_binocularOverlap()

    def multiEyeOverlap_compute(self, eyes=("AME", "ALE", "PME", "PLE")):
        for focus_eye in eyes:
            for compare_eye in eyes:
                if focus_eye != compare_eye:
                    if focus_eye in self.available_eyes and compare_eye in self.available_eyes:
                        compare_data = self.eyes[compare_eye].spherical_coordinates
                        self.eyes[focus_eye].compute_arbitraryOverlap(compare_eye, compare_data)

    def fullSphereOverlap_compute(self, eyes=("AME", "ALE", "PME", "PLE")):
        for eye in eyes:
            if eye in self.available_eyes:
                compare_data = {'polygon': Polygon([(-np.pi, np.pi/2), (np.pi, np.pi/2), (np.pi, -np.pi/2), (-np.pi, -np.pi/2)])}
                self.eyes[eye].compute_fullfieldOverlap('FullField', compare_data)

    def overlapsReport_save(self, filename, eyes=("AME", "ALE", "PME", "PLE")):

        checks = ['AME', 'ALE', 'PME', 'PLE', 'binocular', 'FullField']
        all_overlaps = {}
        for eye in eyes:
            if eye in self.available_eyes:
                overlaps = []
                for check in checks:
                    if check in self.eyes[eye].spherical_coordinates['overlaps']:
                        val = self.eyes[eye].spherical_coordinates['overlaps'][check]['overlap_percentage']
                    else:
                        val = np.nan
                    overlaps.append(val)
                all_overlaps[eye] = overlaps
        all_overlaps = pd.DataFrame(all_overlaps, index=checks)
        all_overlaps = all_overlaps.T
        all_overlaps.to_csv(self.path + filename + '_overlapsReport.csv')

    def sphericalCoordinates_save(self, filename, eyes=("AME", "ALE", "PME", "PLE")):
        eyeSummary_general_azimuth = {}
        eyeSummary_general_elevation = {}
        for eye in eyes:
            if eye in self.available_eyes:
                spherical_points = pd.DataFrame(self.eyes[eye].spherical_coordinates['spherical_points'])
                spherical_points.to_csv(self.path+filename+'_'+eye+'_spherical_points.csv')
                spherical_points_outline = self.eyes[eye].spherical_coordinates['alphashape_points']:               
                spherical_points_outline.to_csv(self.path+filename+'_'+eye+'_spherical_points_outline.csv')
                
                data = self.eyes[eye].spherical_coordinates['azimuth_max_spans']
                if 'elevation_bin_min' not in list(eyeSummary_general_azimuth.keys()):
                    eyeSummary_general_azimuth['elevation_bin_min'] = np.array(data['elevation_range'])[:, 0]
                    eyeSummary_general_azimuth['elevation_bin_max'] = np.array(data['elevation_range'])[:, 1]
                eyeSummary_general_azimuth[eye+'_span'] = np.array(data['span'])
                eyeSummary_general_azimuth[eye+'_angle_span_max'] = np.array(data['extremes'])[:, 0]
                eyeSummary_general_azimuth[eye+'_angle_span_min'] = np.array(data['extremes'])[:, 1]


                data = self.eyes[eye].spherical_coordinates['elevation_max_spans']
                if 'azimuth_bin_min' not in list(eyeSummary_general_elevation.keys()):
                    eyeSummary_general_elevation['azimuth_bin_min'] = np.array(data['azimuth_range'])[:, 0]
                    eyeSummary_general_elevation['azimuth_bin_max'] = np.array(data['azimuth_range'])[:, 1]
                eyeSummary_general_elevation[eye+'_span'] = np.array(data['span'])
                eyeSummary_general_elevation[eye+'_angle_span_max'] = np.array(data['extremes'])[:, 0]
                eyeSummary_general_elevation[eye+'_angle_span_min'] = np.array(data['extremes'])[:, 1]

        eyeSummary_general_azimuth = pd.DataFrame(eyeSummary_general_azimuth)
        eyeSummary_general_azimuth.to_csv(
            self.path + filename + '_AzimuthSpans.csv')
        eyeSummary_general_elevation = pd.DataFrame(eyeSummary_general_elevation)
        eyeSummary_general_elevation.to_csv(
            self.path + filename + '_ElevationSpans.csv')


    def sphericalCoordinates_plotSorted(self, eyes=("AME", "ALE", "PME", "PLE")):

        fig, axs = plt.subplots(1,2)
        for eye in eyes:
            # # Azimuth and Elevation single plots
            axs[0].plot(range(len(self.eyes[eye].spherical_coordinates['spherical_points']['azimuth'])),
                       np.sort(self.eyes[eye].spherical_coordinates['spherical_points']['azimuth']),
                        linewidth=2, label=eye, color=self.toplot_colors[eye])
            axs[1].plot(range(len(self.eyes[eye].spherical_coordinates['spherical_points']['elevation'])),
                        np.sort(self.eyes[eye].spherical_coordinates['spherical_points']['elevation']),
                        linewidth=2, label=eye, color=self.toplot_colors[eye])
        # # Single Plot
        axs[0].grid()
        axs[0].set_xlabel('N° of Points (sorted in terms of azimuth)')
        axs[0].set_ylabel('Azimuth [rad]')
        axs[0].set_title('Azimuth')
        axs[0].legend()

        axs[1].grid()
        axs[1].set_xlabel('N° of Points (sorted in terms of elevation)')
        axs[1].set_ylabel('Elevation [rad]')
        axs[1].set_title('Elevation')
        axs[1].legend()
        fig.show()

    def sphericalCoordinates_plotFields(self, eyes=("AME", "ALE", "PME", "PLE"), binocular=True, ret=False):
        fig, ax = plt.subplots()
        from shapely.affinity import scale
        for eye in eyes:
            if self.eyes[eye].spherical_coordinates['polygons_amount'] == 1:
                ax.fill(self.eyes[eye].spherical_coordinates['polygon'].exterior.xy[0],
                        self.eyes[eye].spherical_coordinates['polygon'].exterior.xy[1],
                        label=eye, color=self.toplot_colors[eye], alpha=0.5)
            else:
                for geom in self.eyes[eye].spherical_coordinates['polygon'].geoms:
                    ax.fill(geom.exterior.xy[0],
                            geom.exterior.xy[1],
                            label=eye, color=self.toplot_colors[eye], alpha=0.5)
            if binocular:
                if self.eyes[eye].spherical_coordinates['polygons_amount'] == 1:
                    reverse = scale(self.eyes[eye].spherical_coordinates['polygon'], xfact=-1, yfact=1, origin=(0,0))
                    ax.fill(reverse.exterior.xy[0],
                            reverse.exterior.xy[1],
                            label=eye, color=self.toplot_colors[eye], alpha=0.5)
                else:
                    for geom in self.eyes[eye].spherical_coordinates['polygon'].geoms:
                        reverse = scale(geom, xfact=-1, yfact=1, origin=(0,0))
                        ax.fill(reverse.exterior.xy[0],
                                reverse.exterior.xy[1],
                                label=eye, color=self.toplot_colors[eye], alpha=0.5)

        # # Azimuth vs Elevation plots
        ax.grid()
        ax.set_xlabel('Azimuth [rad]')
        ax.set_ylabel('Elevation [rad]')
        ax.set_title("Elevation vs Azimuth")
        ax.set_xlim(-np.pi, np.pi)
        ax.legend()
        ax.set_aspect('equal')
        fig.show()

    def sphericalCoordinates_plotSpans(self, eyes=("AME", "ALE", "PME", "PLE"), ret=False):
        fig, axs = plt.subplots(1,2)
        for eye in eyes:
            az = self.eyes[eye].spherical_coordinates['azimuth_max_spans']
            el = self.eyes[eye].spherical_coordinates['elevation_max_spans']

            aznames = np.rad2deg(np.array(az['elevation_range']).flatten())
            azvals = np.rad2deg(np.repeat(az['span'], 2))
            elnames = np.rad2deg(np.array(el['azimuth_range']).flatten())
            elvals = np.rad2deg(np.repeat(el['span'], 2))

            # Span
            axs[1].plot(azvals,
                        aznames, label='Span of ' + eye, color=self.toplot_colors[eye])
            axs[0].plot(elnames,
                        elvals, label='Span of ' + eye, color=self.toplot_colors[eye])

        # Span
        axs[0].grid()
        axs[0].set_xlabel('Azimuth Range')
        axs[0].set_ylabel('Elevation')
        axs[0].set_title("FOV elevation span per azimuth window")
        axs[0].set_xlim(-180, 180)
        axs[0].set_ylim(0, 360)
        axs[0].legend()

        axs[1].grid()
        axs[1].set_xlabel('Azimuth')
        axs[1].set_ylabel('Elevation Range')
        axs[1].set_title("FOV azimuth span per elevation window")
        axs[1].set_xlim(0, 360)
        axs[1].set_ylim(-90, 90)
        axs[1].legend()
        fig.show()


    def plot_matplotlib(
        self,
        eyes=("AME", "ALE", "PME", "PLE"),
        elements=("lens", "retina", "projection", "projection_full", "FOVoutline"),
        plot_FOV_sphere=True,
        field_mm=150,
        alpha=1,
    ):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_box_aspect([1, 1, 1])
        for eye in eyes:
            if eye in self.available_eyes:
                if "lens" in elements:
                    lens = self.eyes[eye].StandardOrientationLensCloud.convex_hull.vertices
                    ax.scatter(lens[:, 0], lens[:, 1], lens[:, 2], color=self.toplot_colors[eye])
                if "retina" in elements:
                    retina = self.eyes[eye].StandardOrientationRetinaCloud.convex_hull.vertices
                    ax.scatter(retina[:, 0], retina[:, 1], retina[:, 2], color=self.toplot_colors[eye])
                if "projection" in elements:
                    Project = np.array(self.eyes[eye].StandardOrientationProjectedVectors)[:, 2]
                    ax.scatter(
                        Project[:, 0],
                        Project[:, 1],
                        Project[:, 2],
                        color=self.toplot_colors[eye],
                        alpha=alpha,
                    )
                if "projection_full" in elements:
                    Project = np.array(self.eyes[eye].StandardOrientationProjectedVectorsFull)[
                        :, 2
                    ]
                    ax.scatter(
                        Project[:, 0],
                        Project[:, 1],
                        Project[:, 2],
                        color=self.toplot_colors[eye],
                        alpha=alpha,
                    )
                if "FOVoutline" in elements:
                    Outline = self.eyes[eye].FOVcontourPoints
                    ax.scatter(
                        Outline[:, 0],
                        Outline[:, 1],
                        Outline[:, 2],
                        color=self.toplot_colors[eye],
                        alpha=alpha,
                    )

        if plot_FOV_sphere:
            u, v = np.mgrid[0 : 2 * np.pi : 50j, 0 : np.pi : 50j]
            x = int(field_mm / self.voxelSize) * np.cos(u) * np.sin(v)
            y = int(field_mm / self.voxelSize) * np.sin(u) * np.sin(v)
            z = int(field_mm / self.voxelSize) * np.cos(v)

            ax.plot_wireframe(x, y, z, linewidth=0.50, color="darkgrey")

        # Compose SO(3) group
        R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # Origin as the center marker
        origin = [0, 0, 0]
        
        # Visualizing
        for axis in R:
            ax.quiver(*origin, *axis, length=int((field_mm / self.voxelSize)/2))
        ax.text(int(field_mm / self.voxelSize), 0, 0, "x")
        ax.text(0, int(field_mm / self.voxelSize), 0, "y")
        ax.text(0, 0, int(field_mm / self.voxelSize), "z")

        plt.show()

    def plot_pyplot(
        self,
        eyes=("AME", "ALE", "PME", "PLE"),
        elements=("lens", "retina", "projection", "projection_full", "FOVoutline", "planes"),
        plot_FOV_sphere=True,
        field_mm=150,
        alpha=1
    ):
        toplot = []
        if plot_FOV_sphere:
            u, v = np.mgrid[0:2 * np.pi:50j, 0:np.pi:50j]
            x = field_mm / self.voxelSize * np.cos(u) * np.sin(v)
            y = field_mm / self.voxelSize * np.sin(u) * np.sin(v)
            z = field_mm / self.voxelSize * np.cos(v)

            sphere = go.Surface(x=x, y=y, z=z, opacity=0.7, colorscale=[[0, 'white'], [1, 'white']],
                                showscale=False)
            toplot.append(sphere)

        for eye in eyes:
            if "FOVoutline" in elements:
                # Compact Form
                Outline = self.eyes[eye].FOVcontourPoints
                dots = go.Scatter3d(x=Outline[:, 0], y=Outline[:, 1], z=Outline[:, 2],
                                     mode='markers', marker={'color': self.toplot_colors[eye], 'size': 2})
                toplot.append(dots)
        fig = go.Figure(data=toplot)
        fig.show()

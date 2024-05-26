### Image processing ###
import cv2  # computer vision 2, reads images
import trimesh  # to do 3d geometry

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
from matplotlib import cm
import plotly.graph_objects as go

import copy

### Exceptions ###
class UnrecognizedEye(Exception):
    pass

class WrongCommand(Exception):
    pass

class InvalidDataset(Exception):
    pass

### Classes ###
class Eye:
    """
    This class is responsible for doing operations and containing
    all the data for each eye
    """

    def __init__(self, eye_identity: str, params):
        """
        :param eye_identity: one of AME, ALE, PME, PLE
        """
        # Extract Parameters from YAML file
        # Debug

        ## Standardized Data        
        # New Method | More Robust and compact
        if (eye_identity == "AME") or (eye_identity == "ALE") or (eye_identity == "PME") or (eye_identity == "PLE"):
            self.EyeIdentity = eye_identity
            LensColor = (np.array(params[eye_identity]["Lens"]["low_color"]), np.array(params[eye_identity]["Lens"]["high_color"]))
            RetinaColor = (np.array(params[eye_identity]["Retina"]["low_color"]), np.array(params[eye_identity]["Retina"]["high_color"]))
        else:
           raise UnrecognizedEye("You inputted the wrong eye name, abort.")

        self.LensColor = LensColor
        self.RetinaColor = RetinaColor

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

        self.spherical_coordinates = {}

    '''
    def amira_find_lens_points(self, labels_pictures_list):
        """
        this formula takes the list of label pictures and threshold it based on
        the given colours to find lenses
        """
        for label in tqdm(
            labels_pictures_list, desc="finding " + self.EyeIdentity + " Lens"
        ):  # for every slice
            # find pixels with the determed color and set them as 1, all else as 0
            self.LensMask.append(
                cv2.inRange(label, self.LensColor[0], self.LensColor[1])
            )

        print("Computing coordinates...")
        self.LensMask = np.array(self.LensMask)
        self.LensPoints = np.argwhere(self.LensMask > 0)

    def amira_find_retinas_points(self, labels_pictures_list):
        """
        this formula takes the list of label pictures and threshold it based on
        the given colours to find retinas
        """
        for label in tqdm(
            labels_pictures_list, desc="finding " + self.EyeIdentity + " Retina"
        ):  # for every slice
            # find pixels with the determed color and set them as 1, all else as 0
            self.RetinaMask.append(
                cv2.inRange(label, self.RetinaColor[0], self.RetinaColor[1])
            )

        print("Computing coordinates...")
        self.RetinaMask = np.array(self.RetinaMask)
        self.RetinaPoints = np.argwhere(self.RetinaMask > 0)

    def amira_find_all_points(self, labels_pictures_list):
        """
        duh
        """
        self.amira_find_lens_points(labels_pictures_list)
        self.amira_find_retinas_points(labels_pictures_list)
    '''

    def find_points(self, piclist, part="Lens", style='binary'):
        if style == 'binary':
            if part == "Lens":
                self.LensPoints = np.argwhere(np.array(piclist) > 0)
            elif part == "Retina":
                self.RetinaPoints = np.argwhere(np.array(piclist) > 0)
        elif style== 'color':
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
                else:
                    mask = self.RotatedLensCloud.convex_hull.vertices[:, 0] < 0
            elif overlapY < overlapX and overlapY < overlapZ:
                if LensSpanY[0] > RetinaSpanY[0]:
                    mask = self.RotatedLensCloud.convex_hull.vertices[:, 1] > 0
                else:
                    mask = self.RotatedLensCloud.convex_hull.vertices[:, 1] < 0
            elif overlapZ < overlapX and overlapZ < overlapY:
                if LensSpanZ[0] > RetinaSpanZ[0]:
                    mask = self.RotatedLensCloud.convex_hull.vertices[:, 2] > 0
                else:
                    mask = self.RotatedLensCloud.convex_hull.vertices[:, 2] < 0
            
            self.RotatedLensSurfacePoints = self.RotatedLensCloud.convex_hull.vertices[mask, :]
        else:
            print("cry")  # TODO: do a better error catching process
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

    def find_lens_sphere(self):
        """
        helper function, call the two previous formulas to do everything in one step
        """
        self.find_split_plane()
        self.RotatedLensSphere = self.sphere_fit(self.RotatedLensSurfacePoints)

    def rotate_back(self):
        """
        rotate the sphere back to the original frame of reference
        """

        homMatrix = np.linalg.inv(
            self.LensCloud.convex_hull.principal_inertia_transform
        )

        self.LensSphere = (
            self.RotatedLensSphere[0],
            trimesh.transform_points([self.RotatedLensSphere[1]], homMatrix)[0],
        )

    def orientToStandard(self, hom_matrix):
        print(f'Reorienting {self.EyeIdentity} dots...', end='')
        # Lens
        self.StandardOrientationLensPoints = trimesh.transform_points(self.LensPoints, hom_matrix)
        self.StandardOrientationLensCloud = trimesh.points.PointCloud(self.StandardOrientationLensPoints)
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
    
    def calculate_span(self, visual_field_radius, voxel_size): #TODO obsolete
        outlinePoints = self.FOVcontourPoints

        transversePlanePoints = np.delete(outlinePoints, 2, 1)
        transversePlaneDistances = np.sqrt(transversePlanePoints[:, 0] * transversePlanePoints[:, 0] +
                                           transversePlanePoints[:, 1] * transversePlanePoints[:, 1])
        transverseArgsPointsOnPlane = np.where(
            np.logical_and(transversePlaneDistances >= (visual_field_radius / voxel_size) * 0.95,
                           transversePlaneDistances <= (visual_field_radius / voxel_size) * 1))
        transverseAngles = np.rad2deg(np.arctan(transversePlanePoints[:, 1] / transversePlanePoints[:, 0]))
        transverseAnglesOnPlane = transverseAngles[transverseArgsPointsOnPlane]
        if len(transverseAnglesOnPlane) > 0:
            transverseSpan = [np.round(np.min(transverseAnglesOnPlane), 3),
                              np.round(np.max(transverseAnglesOnPlane), 3)]
            transverseArgSpan = [np.argmin(transverseAnglesOnPlane), np.argmax(transverseAnglesOnPlane)]
        else:
            transverseSpan = [None, None]

        coronalPlanePoints = np.delete(outlinePoints, 0, 1)
        coronalPlaneDistances = np.sqrt(coronalPlanePoints[:, 0] * coronalPlanePoints[:, 0] +
                                        coronalPlanePoints[:, 1] * coronalPlanePoints[:, 1])
        coronalArgsPointsOnPlane = np.where(
            np.logical_and(coronalPlaneDistances >= (visual_field_radius / voxel_size) * 0.95,
                           coronalPlaneDistances <= (visual_field_radius / voxel_size) * 1.2))
        coronalAngles = np.rad2deg(np.arctan(coronalPlanePoints[:, 1] / coronalPlanePoints[:, 0]))
        coronalAnglesOnPlane = coronalAngles[coronalArgsPointsOnPlane]
        if len(coronalAnglesOnPlane) > 0:
            coronalSpan = [np.round(np.min(coronalAnglesOnPlane), 3),
                           np.round(np.max(coronalAnglesOnPlane), 3)]
            coronalArgSpan = [np.argmin(coronalAnglesOnPlane), np.argmax(coronalAnglesOnPlane)]
        else:
            coronalSpan = [None, None]

        sagittalPlanePoints = np.delete(outlinePoints, 1, 1)
        sagittalPlaneDistances = np.sqrt(sagittalPlanePoints[:, 0] * sagittalPlanePoints[:, 0] +
                                         sagittalPlanePoints[:, 1] * sagittalPlanePoints[:, 1])
        sagittalArgsPointsOnPlane = np.where(
            np.logical_and(sagittalPlaneDistances >= (visual_field_radius / voxel_size) * 0.95,
                           sagittalPlaneDistances <= (visual_field_radius / voxel_size) * 1.2))
        sagittalAngles = np.rad2deg(np.arctan(sagittalPlanePoints[:, 1] / sagittalPlanePoints[:, 0]))
        sagittalAnglesOnPlane = sagittalAngles[sagittalArgsPointsOnPlane]
        if len(sagittalAnglesOnPlane) > 0:
            sagittalSpan = [np.round(np.min(sagittalAnglesOnPlane), 3),
                            np.round(np.max(sagittalAnglesOnPlane), 3)]
            sagittalArgSpan = [np.argmin(sagittalAnglesOnPlane), np.argmax(sagittalAnglesOnPlane)]
        else:
            sagittalSpan = [None, None]

        return {'Transverse': transverseSpan, 'Coronal': coronalSpan, 'Sagittal': sagittalSpan}
    
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
    
    def pairwise_angle_diff(self, angles):
        n_data = len(angles)
        pairwise_diff = []

        for i in range(n_data):
            for j in range(n_data):
                diff = angles[i] - angles[j]
                pairwise_diff.append([np.arctan2(np.sin(diff), np.cos(diff)),
                                       angles[i], angles[j]])
        return np.array(pairwise_diff)

    def calculate_span2(self, specific_discretization=15, general_discretization=72):
        """
            (i)     visual_field_radius: Radius of the Sphere;
            (ii)    voxel_size: how many points in mm.  
        """
        # From Cartesian to Spherical
        spherical_points = []
        for point in self.FOVcontourPoints:
            spherical_points.append(self.cartesian2spherical(list(point)))

        spherical_points = np.array(spherical_points)

        # Sorting Points in terms of (i) azimuth and (ii) elevation
        azimuth_points = copy.deepcopy(spherical_points[:, 1])
        azimuth_points.sort()

        elevation_points = copy.deepcopy(spherical_points[:, 2])
        elevation_points.sort()

        ### SPECIFIC DISCRETIZATION ###

        # Azimuth range
        azimuth_range = np.linspace(azimuth_points[0], azimuth_points[-1], specific_discretization)
        azimuth_ranges = [azimuth_range[:-1],azimuth_range[1:]]
        elevation_max_spans = {'azimuth_range': [], 'span': [], 'extremes': []}

        for r in np.array(azimuth_ranges).T:
            # Points in the current range
            current_points = np.argwhere((spherical_points[:, 1] < r[1]) & (spherical_points[:, 1] >= r[0])).flatten()

            elevation_max_spans['azimuth_range'].append(r)

            if len(current_points) < 2:
                elevation_max_spans['span'].append(np.nan)
                elevation_max_spans['extremes'].append(np.nan, np.nan)

            else:
                spans = self.pairwise_angle_diff(spherical_points[current_points, 2])
                this_spans = spans[np.argmax(spans[:,0])]
                elevation_max_spans['span'].append(this_spans[0])
                elevation_max_spans['extremes'].append(this_spans[1:])

        # Elevation range
        elevation_range = np.linspace(elevation_points[0], elevation_points[-1], specific_discretization)
        elevation_ranges = [elevation_range[:-1], elevation_range[1:]]
        azimuth_max_spans = {'elevation_range': [], 'span': [], 'extremes': []}

        for r in np.array(elevation_ranges).T:
            # Points in the current range
            current_points = np.argwhere((spherical_points[:, 2] < r[1]) & (spherical_points[:, 2] >= r[0])).flatten()

            azimuth_max_spans['elevation_range'].append(r)
            if len(current_points) < 2:
                azimuth_max_spans['span'].append(np.nan)
                azimuth_max_spans['extremes'].append([np.nan, np.nan])

            else:
                spans = self.pairwise_angle_diff(spherical_points[current_points, 1])
                this_spans = spans[np.argmax(spans[:, 0])]
                azimuth_max_spans['span'].append(this_spans[0])
                azimuth_max_spans['extremes'].append(this_spans[1:])

        self.spherical_coordinates['spherical_points'] = {'azimuth': spherical_points[:,1], 'elevation': spherical_points[:,2]}
        self.spherical_coordinates['azimuth_max_spans'] = {}
        self.spherical_coordinates['elevation_max_spans'] = {}
        self.spherical_coordinates['azimuth_max_spans']['specific_discretization'] = azimuth_max_spans
        self.spherical_coordinates['elevation_max_spans']['specific_discretization'] = elevation_max_spans

        ### General DISCRETIZATION ###

        # Azimuth range
        azimuth_range = np.linspace(np.deg2rad(-180), np.deg2rad(180), general_discretization)
        azimuth_ranges = [azimuth_range[:-1], azimuth_range[1:]]
        elevation_max_spans = {'azimuth_range': [], 'span': [], 'extremes': []}

        for r in np.array(azimuth_ranges).T:
            # Points in the current range
            current_points = np.argwhere((spherical_points[:, 1] < r[1]) & (spherical_points[:, 1] >= r[0])).flatten()

            elevation_max_spans['azimuth_range'].append(r)
            if len(current_points) < 2:
                elevation_max_spans['span'].append(np.nan)
                elevation_max_spans['extremes'].append([np.nan, np.nan])

            else:
                spans = self.pairwise_angle_diff(spherical_points[current_points, 2])
                this_spans = spans[np.argmax(spans[:, 0])]
                elevation_max_spans['span'].append(this_spans[0])
                elevation_max_spans['extremes'].append(this_spans[1:])

        # Elevation range
        elevation_range = np.linspace(np.deg2rad(-180), np.deg2rad(180), general_discretization)
        elevation_ranges = [elevation_range[:-1], elevation_range[1:]]
        azimuth_max_spans = {'elevation_range': [], 'span': [], 'extremes': []}

        for r in np.array(elevation_ranges).T:
            # Points in the current range
            current_points = np.argwhere((spherical_points[:, 2] < r[1]) & (spherical_points[:, 2] >= r[0])).flatten()

            azimuth_max_spans['elevation_range'].append(r)
            if len(current_points) < 2:
                azimuth_max_spans['span'].append(np.nan)
                azimuth_max_spans['extremes'].append([np.nan, np.nan])

            else:
                spans = self.pairwise_angle_diff(spherical_points[current_points, 1])
                this_spans = spans[np.argmax(spans[:, 0])]
                azimuth_max_spans['span'].append(this_spans[0])
                azimuth_max_spans['extremes'].append(this_spans[1:])

        self.spherical_coordinates['azimuth_max_spans']['general_discretization'] = azimuth_max_spans
        self.spherical_coordinates['elevation_max_spans']['general_discretization'] = elevation_max_spans

  
class Spider:
    """
    This class creates a spider object. this does the full computation and it is the only one you need to use
    the rest is called from here
    """

    def __init__(
        self,
        workdir,
        paramspath,
        label_names=None,
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

        with open(paramspath, 'r') as file:
            self.colors = yaml.safe_load(file)

        ## New Version with a Dictionary
        self.eyes = {}
        for eye in self.available_eyes:
            self.eyes[eye] = Eye(eye_identity=eye, params=self.colors)

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
        self.LabelNames = label_names

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

        self.toplot_colors = eyes_toplot_colors

    '''
    
    LEGACY CODE. KEEP FOR NOW BUT TO TRASH
    
    def amira_load_full_labels(self):
        for file in tqdm(sorted(os.listdir(self.path)), desc="loading images"):
            self.AmiraLabelPictures.append(cv2.imread(self.path + file, 1))

    def amira_find_all_points(self):
        # Find Lens and Retina Points for each eye
        for eye in self.available_eyes:
            self.eyes[eye].amira_find_all_points(self.AmiraLabelPictures)
    '''
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
                self.SeparateLabelPictures[group][object].append(cv2.imread(self.path + file, 0))
            elif style == 'color':
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

    def compute_eye(self, eye):
        """
        helper function to do all the required computation for each eye. look into class eye for each single function
        :param eye: eye identity. can be AME, ALE, PME, PLE
        """
        if eye in self.available_eyes:
            self.eyes[eye].define_all_clouds()
            self.eyes[eye].align_to_zero()
            self.eyes[eye].find_lens_sphere()
            self.eyes[eye].rotate_back()
        else:
            raise(UnrecognizedEye("Unrecognized Eye: Computation aborted."))

    def compute_eyes(self):
        """
        run this! compute all eyes together
        """
        print('Computing lenses and retina geometries...', end='')
        for eye in self.available_eyes:
            self.compute_eye(eye)
        print(' Done')

    def compute_cephalothorax(self, style='binary'):
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
        self.spider_SoR = np.linalg.inv(self.head_SoR(plot=False))    # [4, 4] \in SE(3)


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
        
    def head_SoR(self, plot=False):
        ## Plot points ##
        # Create fig obj
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.view_init(elev=-160, azim=106)
        
        # Read all markers
        n_marker = len(self.cephalothoraxMarkers)
        marker_type = list(self.cephalothoraxMarkers.keys())
        marker_color = ['#323031', '#177E89', '#084C61', '#DB3A34', '#FFC857', '#FF9F1C', '#8ED081']
        
        point_dataset = []
        
        # For each marker, plot a different color
        for i in range(n_marker):
            if plot:
                ax.scatter(self.cephalothoraxMarkers[marker_type[i]][0], 
                        self.cephalothoraxMarkers[marker_type[i]][1], 
                        self.cephalothoraxMarkers[marker_type[i]][2],
                        color = marker_color[i])
                ax.text(self.cephalothoraxMarkers[marker_type[i]][0], 
                        self.cephalothoraxMarkers[marker_type[i]][1], 
                        self.cephalothoraxMarkers[marker_type[i]][2],
                        marker_type[i])
            point_dataset.append(list(self.cephalothoraxMarkers[marker_type[i]]))
        
        point_dataset = np.array(point_dataset)            
        
        if plot:
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
        
        ## Create 3D Rectangle ##
        
        x_hand = np.array(list(self.cephalothoraxMarkers['front'])) - np.array(list(self.cephalothoraxMarkers['back']))
        width = np.linalg.norm(x_hand)
        y_hand = np.array(list(self.cephalothoraxMarkers['left'])) - np.array(list(self.cephalothoraxMarkers['right']))
        depth = np.linalg.norm(y_hand)
        z_hand = np.array(list(self.cephalothoraxMarkers['top'])) - np.array(list(self.cephalothoraxMarkers['bottom']))
        height = np.linalg.norm(z_hand)

        if plot:
            ax.set_xlabel('X [pixel]')
            ax.set_ylabel('Y [pixel]')
            ax.set_zlabel('Z [n° layer]')
        
        # # Proposal 2: Pure geometrical method
        # Axis 1: back -> front
        x_axis = x_hand
        # x_center = np.array(list(self.cephalothoraxMarkers['back'])) + 0.5*x_hand
        x_axis /= width
        
        # Orthogonal Axis
        z_hand /= height
        z_axis = - self.pure_geometrical(z_hand, x_axis)        
        
        # Finally, find y by cross product (z cross x)
        y_axis =  np.cross(z_axis, x_axis)
        
        # Compose SO(3) group
        R = np.array([x_axis, y_axis, z_axis])
        # Origin as the center marker
        origin = list(self.cephalothoraxMarkers['center'])
        
        if plot:
            # Visualizing
            for axis in R:
                ax.quiver(*origin, *axis, length=500)
                
        # Composing SE(3) group
        R = R.T
        T = np.concatenate((R, np.array([origin]).T), axis=1)
        T = np.concatenate((T, np.array([[0, 0, 0, 1]])), axis=0)
            
        if plot:    
            plt.show()

        return T

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

    def calculate_eyes_spans(
            self,
            field_radius,
            eyes=("AME", "ALE", "PME", "PLE")
    ): #TODO obsolete
        spans = dict.fromkeys(eyes, None)

        for eye in eyes:
            spans[eye] = self.eyes[eye].calculate_span(visual_field_radius=field_radius,
                                                       voxel_size=self.voxelSize)
        return spans

    #TODO I am not sure this works still. Ask Daniele
    def sphericalCoordinates_compute(self, eyes=("AME", "ALE", "PME", "PLE"), specific_discretization=15, general_discretization=72):

        # Extract Information
        for eye in eyes:
            self.eyes[eye].calculate_span2(specific_discretization, general_discretization)

    def sphericalCoordinates_save(self, eyes=("AME", "ALE", "PME", "PLE"), raw=False, span=False, overlap=False):
        pass
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
    def sphericalCoordinates_plotFields(self, eyes=("AME", "ALE", "PME", "PLE"), ret=False):
        fig, ax = plt.subplots()
        for eye in eyes:
            ax.plot(self.eyes[eye].spherical_coordinates['spherical_points']['azimuth'],
                    self.eyes[eye].spherical_coordinates['spherical_points']['elevation'],
                    'o', label=eye, markersize=3, color=self.toplot_colors[eye])

        # # Azimuth vs Elevation plots
        ax.grid()
        ax.set_xlabel('Azimuth [rad]')
        ax.set_ylabel('Elevation [rad]')
        ax.set_title("Elevation vs Azimuth")
        ax.set_xlim(-np.pi, np.pi)
        ax.legend()
        fig.show()

    def sphericalCoordinates_plotSpans(self, eyes=("AME", "ALE", "PME", "PLE"), disc='general', ret=False):
        fig, axs = plt.subplots(1,2)
        for eye in eyes:
            if disc=='general':
                az = self.eyes[eye].spherical_coordinates['azimuth_max_spans']['general_discretization']
                el = self.eyes[eye].spherical_coordinates['elevation_max_spans']['general_discretization']
            elif disc=='specific':
                az = self.eyes[eye].spherical_coordinates['azimuth_max_spans']['specific_discretization']
                el = self.eyes[eye].spherical_coordinates['elevation_max_spans']['specific_discretization']
            else:
                raise ValueError("undefined disdcretization. please check")

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
        axs[0].set_ylim(0, 180)
        axs[0].legend()

        axs[1].grid()
        axs[1].set_xlabel('Azimuth')
        axs[1].set_ylabel('Elevation Range')
        axs[1].set_title("FOV azimuth span per elevation window")
        axs[1].set_xlim(0, 180)
        axs[1].set_ylim(-180, 180)
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
            ax.quiver(*origin, *axis, length=20000)
        ax.text(20000, 0, 0, "x")
        ax.text(0, 20000, 0, "y")
        ax.text(0, 0, 20000, "z")

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
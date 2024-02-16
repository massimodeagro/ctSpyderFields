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
        matrix found by pointcloud of retina, in order to align everything to the standard axis
        """
        # Notation: # 
        # Rotation Matrix is a 3x3 matrix (SO(3) group) that expresses a Rotation
        # Homogeneous Matrix is a 4x4 matrix (SE(3) group) that expresses a Roto-translation

        hom_matrix = self.LensCloud.convex_hull.principal_inertia_transform

        self.RotatedLensPoints = trimesh.transform_points(
            self.LensPoints, hom_matrix
        )
        self.RotatedLensCloud = trimesh.points.PointCloud(self.RotatedLensPoints)
        self.RotatedRetinaPoints = trimesh.transform_points(
            self.RetinaPoints, hom_matrix
        )
        self.RotatedRetinaCloud = trimesh.points.PointCloud(self.RotatedRetinaPoints)

    def find_split_plane(self):
        """
        this function finds the plane (xy, xz, yz) that divides retina from lens,
        as well as on which side of the two is the retina and on which is the lens.
        This is needed for finding the cap of the lens.
        """

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
            self.RotatedLensSurfacePoints = self.RotatedLensCloud.convex_hull.vertices[
                mask, :
            ]
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

    def orientToStandard(self, rotationMatrix):
        print(f'Reorienting {self.EyeIdentity} dots...', end='')
        self.StandardOrientationLensPoints = trimesh.transform_points(
            self.LensPoints, rotationMatrix
        )
        self.StandardOrientationLensCloud = trimesh.points.PointCloud(
            self.StandardOrientationLensPoints
        )
        self.StandardOrientationRetinaPoints = trimesh.transform_points(
            self.RetinaPoints, rotationMatrix
        )
        self.StandardOrientationRetinaCloud = trimesh.points.PointCloud(
            self.StandardOrientationRetinaPoints
        )
        self.StandardOrientationLensSphere = (
            self.LensSphere[0],
            trimesh.transform_points([self.LensSphere[1]], rotationMatrix)[0],
        )
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
        sx = self.StandardOrientationLensSphere[1][0]
        sy = self.StandardOrientationLensSphere[1][1]
        sz = self.StandardOrientationLensSphere[1][2]
        for point in self.StandardOrientationRetinaCloud.vertices:
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

            self.StandardOrientationProjectedVectorsFull.append(
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
            self.focalLengthsFull.append([point, vmag])
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
                    tmpdots.append(cv2.inRange(label, self.colors[marker]["low_color"], self.colors[marker]["high_color"]))
                tmpdots = np.array(tmpdots)
                dots = np.argwhere(tmpdots > 0)
            self.cephalothoraxMarkers[marker] = (
                np.mean(dots[:, 0]),
                np.mean(dots[:, 1]),
                np.mean(dots[:, 2]),
            )
            allpoints.append(self.cephalothoraxMarkers[marker])
        self.cephalothoraxCloud = trimesh.points.PointCloud(allpoints)

    def orient_to_standard(self):
        rotationMatrix = self.cephalothoraxCloud.convex_hull.principal_inertia_transform

        # Rotate each eye
        for eye in self.available_eyes:
            self.eyes[eye].orientToStandard(rotationMatrix)

        for marker in self.cephalothoraxMarkers:
            self.StandardOrientationCephalothoraxPoints[
                marker
            ] = trimesh.transform_points(
                [self.cephalothoraxMarkers[marker]], rotationMatrix
            )[
                0
            ]

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
        points = []
        for marker in self.cephalothoraxMarkers:
            points.append(self.cephalothoraxMarkers[marker])
        self.cephalothoraxCloud = trimesh.points.PointCloud(points)
        self.StandardOrientationCephalothoraxPoints = data["cephalothorax"]["Rotated"]

        print(" Done")

    def plot(
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
        if "AME" in eyes and "AME" in self.available_eyes:
            if "lens" in elements:
                lens = self.eyes["AME"].StandardOrientationLensCloud.convex_hull.vertices
                ax.scatter(lens[:, 0], lens[:, 1], lens[:, 2], color="purple")
            if "retina" in elements:
                retina = self.eyes["AME"].StandardOrientationRetinaCloud.convex_hull.vertices
                ax.scatter(retina[:, 0], retina[:, 1], retina[:, 2], color="purple")
            if "projection" in elements:
                Project = np.array(self.eyes["AME"].StandardOrientationProjectedVectors)[:, 2]
                ax.scatter(
                    Project[:, 0],
                    Project[:, 1],
                    Project[:, 2],
                    color="indigo",
                    alpha=alpha,
                )
            if "projection_full" in elements:
                Project = np.array(self.eyes["AME"].StandardOrientationProjectedVectorsFull)[
                    :, 2
                ]
                ax.scatter(
                    Project[:, 0],
                    Project[:, 1],
                    Project[:, 2],
                    color="indigo",
                    alpha=alpha,
                )
            if "FOVoutline" in elements:
                Outline = self.eyes["AME"].FOVcontourPoints
                ax.scatter(
                    Outline[:, 0],
                    Outline[:, 1],
                    Outline[:, 2],
                    color="indigo",
                    alpha=alpha,
                )
        if "ALE" in eyes and "ALE" in self.available_eyes:
            if "lens" in elements:
                lens = self.eyes["ALE"].StandardOrientationLensCloud.convex_hull.vertices
                ax.scatter(lens[:, 0], lens[:, 1], lens[:, 2], color="green")
            if "retina" in elements:
                retina = self.eyes["ALE"].StandardOrientationRetinaCloud.convex_hull.vertices
                ax.scatter(retina[:, 0], retina[:, 1], retina[:, 2], color="green")
            if "projection" in elements:
                Project = np.array(self.eyes["ALE"].StandardOrientationProjectedVectors)[:, 2]
                ax.scatter(
                    Project[:, 0],
                    Project[:, 1],
                    Project[:, 2],
                    color="darkgreen",
                    alpha=alpha,
                )
            if "projection_full" in elements:
                Project = np.array(self.eyes["ALE"].StandardOrientationProjectedVectorsFull)[
                    :, 2
                ]
                ax.scatter(
                    Project[:, 0],
                    Project[:, 1],
                    Project[:, 2],
                    color="darkgreen",
                    alpha=alpha,
                )
            if "FOVoutline" in elements:
                Outline = self.eyes["ALE"].FOVcontourPoints
                ax.scatter(
                    Outline[:, 0],
                    Outline[:, 1],
                    Outline[:, 2],
                    color="darkgreen",
                    alpha=alpha,
                )
        if "PME" in eyes and "PME" in self.available_eyes:
            if "lens" in elements:
                lens = self.eyes["PME"].StandardOrientationLensCloud.convex_hull.vertices
                ax.scatter(lens[:, 0], lens[:, 1], lens[:, 2], color="goldenrod")
            if "retina" in elements:
                retina = self.eyes["PME"].StandardOrientationRetinaCloud.convex_hull.vertices
                ax.scatter(retina[:, 0], retina[:, 1], retina[:, 2], color="goldenrod")
            if "projection" in elements:
                Project = np.array(self.eyes["PME"].StandardOrientationProjectedVectors)[:, 2]
                ax.scatter(
                    Project[:, 0],
                    Project[:, 1],
                    Project[:, 2],
                    color="darkgoldenrod",
                    alpha=alpha,
                )
            if "projection_full" in elements:
                Project = np.array(self.eyes["PME"].StandardOrientationProjectedVectorsFull)[
                    :, 2
                ]
                ax.scatter(
                    Project[:, 0],
                    Project[:, 1],
                    Project[:, 2],
                    color="darkgoldenrod",
                    alpha=alpha,
                )
            if "FOVoutline" in elements:
                Outline = self.eyes["PME"].FOVcontourPoints
                ax.scatter(
                    Outline[:, 0],
                    Outline[:, 1],
                    Outline[:, 2],
                    color="darkgoldenrod",
                    alpha=alpha,
                )
        if "PLE" in eyes and "PLE" in self.available_eyes:
            if "lens" in elements:
                lens = self.eyes["PLE"].StandardOrientationLensCloud.convex_hull.vertices
                ax.scatter(lens[:, 0], lens[:, 1], lens[:, 2], color="darkred")
            if "retina" in elements:
                retina = self.eyes["PLE"].StandardOrientationRetinaCloud.convex_hull.vertices
                ax.scatter(retina[:, 0], retina[:, 1], retina[:, 2], color="darkred")
            if "projection" in elements:
                Project = np.array(self.eyes["PLE"].StandardOrientationProjectedVectors)[:, 2]
                ax.scatter(
                    Project[:, 0],
                    Project[:, 1],
                    Project[:, 2],
                    color="maroon",
                    alpha=alpha,
                )
            if "projection_full" in elements:
                Project = np.array(self.eyes["PLE"].StandardOrientationProjectedVectorsFull)[
                    :, 2
                ]
                ax.scatter(
                    Project[:, 0],
                    Project[:, 1],
                    Project[:, 2],
                    color="maroon",
                    alpha=alpha,
                )
            if "FOVoutline" in elements:
                Outline = self.eyes["PLE"].FOVcontourPoints
                ax.scatter(
                    Outline[:, 0],
                    Outline[:, 1],
                    Outline[:, 2],
                    color="maroon",
                    alpha=alpha,
                )
        if plot_FOV_sphere:
            u, v = np.mgrid[0 : 2 * np.pi : 50j, 0 : np.pi : 50j]
            x = int(field_mm / self.voxelSize) * np.cos(u) * np.sin(v)
            y = int(field_mm / self.voxelSize) * np.sin(u) * np.sin(v)
            z = int(field_mm / self.voxelSize) * np.cos(v)

            ax.plot_wireframe(x, y, z, linewidth=0.50, color="black")

        plt.show()

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

    def __init__(self, eye_identity: str):
        """
        :param eye_identity: one of AME, ALE, PME, PLE
        """
        # Extract Parameters from YAML file
        # Debug
        with open('../ctSpyderFields/params.yaml', 'r') as file:
            params = yaml.safe_load(file)

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

    def dragonfly_find_points(self, piclist, part="Lens"):
        if part == "Lens":
            self.LensPoints = np.argwhere(np.array(piclist) > 0)
        elif part == "Retina":
            self.RetinaPoints = np.argwhere(np.array(piclist) > 0)

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
        rotation_matrix = self.LensCloud.convex_hull.principal_inertia_transform

        self.RotatedLensPoints = trimesh.transform_points(
            self.LensPoints, rotation_matrix
        )
        self.RotatedLensCloud = trimesh.points.PointCloud(self.RotatedLensPoints)
        self.RotatedRetinaPoints = trimesh.transform_points(
            self.RetinaPoints, rotation_matrix
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

        rotationMatrix = np.linalg.inv(
            self.LensCloud.convex_hull.principal_inertia_transform
        )

        self.LensSphere = (
            self.RotatedLensSphere[0],
            trimesh.transform_points([self.RotatedLensSphere[1]], rotationMatrix)[0],
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
        dragonfly_label_names=None,
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

        # TODO: implement this in all the other formulas
        # self.eyes = {}
        # for eye in self.available_eyes:
        #     self.eyes[eye] = Eye(eye_identity=eye)

        ## OLD version
        if "AME" in self.available_eyes:
            self.AME = Eye(eye_identity="AME")
        if "ALE" in self.available_eyes:
            self.ALE = Eye(eye_identity="ALE")
        if "PME" in self.available_eyes:
            self.PME = Eye(eye_identity="PME")
        if "PLE" in self.available_eyes:
            self.PLE = Eye(eye_identity="PLE")

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

        self.AmiraLabelPictures = []
        self.DragonflyLabelNames = dragonfly_label_names
        self.DragonflyLabelPictures = {
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

    def amira_load_labels(self):
        for file in tqdm(sorted(os.listdir(self.path)), desc="loading images"):
            self.AmiraLabelPictures.append(cv2.imread(self.path + file, 1))

    def amira_find_all_points(self):
        self.AME.amira_find_all_points(self.AmiraLabelPictures)
        self.ALE.amira_find_all_points(self.AmiraLabelPictures)
        self.PME.amira_find_all_points(self.AmiraLabelPictures)
        self.PLE.amira_find_all_points(self.AmiraLabelPictures)

    def dragonfly_load_label(self, labelname, group, object):
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
            self.DragonflyLabelPictures[group][object].append(
                cv2.imread(self.path + file, 0)
            )

    def dragonfly_load_all_labels(self):
        """
        This function calls dragonfly_load_label for 4 + 7 times
        """
        for eye in self.available_eyes:
            for label in self.DragonflyLabelNames[eye]:
                self.dragonfly_load_label(
                    labelname=self.DragonflyLabelNames[eye][label],
                    group=eye,
                    object=label,
                )
        for marker in self.DragonflyLabelPictures["Markers"]:
            self.dragonfly_load_label(
                labelname=self.DragonflyLabelNames["Markers"][marker],
                group="Markers",
                object=marker,
            )

    def dragonfly_find_eyes_points(self):
        """
        helper function to find all eyes at once. to see how points are found, look in class eyes, function dragonfly_find_points
        """
        if "AME" in self.available_eyes:
            print("finding AME points...")
            self.AME.dragonfly_find_points(
                piclist=self.DragonflyLabelPictures["AME"]["Lens"], part="Lens"
            )
            self.AME.dragonfly_find_points(
                piclist=self.DragonflyLabelPictures["AME"]["Retina"], part="Retina"
            )
        if "ALE" in self.available_eyes:
            print("finding ALE points...")
            self.ALE.dragonfly_find_points(
                piclist=self.DragonflyLabelPictures["ALE"]["Lens"], part="Lens"
            )
            self.ALE.dragonfly_find_points(
                piclist=self.DragonflyLabelPictures["ALE"]["Retina"], part="Retina"
            )
        if "PME" in self.available_eyes:
            print("finding PME points...")
            self.PME.dragonfly_find_points(
                piclist=self.DragonflyLabelPictures["PME"]["Lens"], part="Lens"
            )
            self.PME.dragonfly_find_points(
                piclist=self.DragonflyLabelPictures["PME"]["Retina"], part="Retina"
            )
        if "PLE" in self.available_eyes:
            print("finding PLE points...")
            self.PLE.dragonfly_find_points(
                piclist=self.DragonflyLabelPictures["PLE"]["Lens"], part="Lens"
            )
            self.PLE.dragonfly_find_points(
                piclist=self.DragonflyLabelPictures["PLE"]["Retina"], part="Retina"
            )

    def compute_eye(self, eye):
        """
        helper function to do all the required computation for each eye. look into class eye for each single function
        :param eye: eye identity. can be AME, ALE, PME, PLE
        """
        if eye == "AME":
            self.AME.define_all_clouds()
            self.AME.align_to_zero()
            self.AME.find_lens_sphere()
            self.AME.rotate_back()
        elif eye == "ALE":
            self.ALE.define_all_clouds()
            self.ALE.align_to_zero()
            self.ALE.find_lens_sphere()
            self.ALE.rotate_back()
        elif eye == "PME":
            self.PME.define_all_clouds()
            self.PME.align_to_zero()
            self.PME.find_lens_sphere()
            self.PME.rotate_back()
        elif eye == "PLE":
            self.PLE.define_all_clouds()
            self.PLE.align_to_zero()
            self.PLE.find_lens_sphere()
            self.PLE.rotate_back()

    def compute_eyes(self):
        """
        run this! compute all eyes together
        """
        print('Computing lenses and retina geometries...', end='')
        for eye in self.available_eyes:
            #self.eyes[eye].define_all_clouds()
            #self.eyes[eye].align_to_zero()
            #self.eyes[eye].find_lens_sphere()
            #self.eyes[eye].rotate_back()
            self.compute_eye(eye)
        print(' Done')


    def compute_cephalothorax(self):
        """
        this first translates the binary pictures in a set of points, and then find the center
        """
        allpoints = []
        for marker in self.cephalothoraxMarkers:
            print("finding " + marker + " points...")
            dots = np.argwhere(
                np.array(self.DragonflyLabelPictures["Markers"][marker]) > 0
            )
            self.cephalothoraxMarkers[marker] = (
                np.mean(dots[:, 0]),
                np.mean(dots[:, 1]),
                np.mean(dots[:, 2]),
            )
            allpoints.append(self.cephalothoraxMarkers[marker])
        self.cephalothoraxCloud = trimesh.points.PointCloud(allpoints)

    def orient_to_standard(self):
        rotationMatrix = self.cephalothoraxCloud.convex_hull.principal_inertia_transform
        self.AME.orientToStandard(rotationMatrix)
        self.ALE.orientToStandard(rotationMatrix)
        self.PME.orientToStandard(rotationMatrix)
        self.PLE.orientToStandard(rotationMatrix)
        for marker in self.cephalothoraxMarkers:
            self.StandardOrientationCephalothoraxPoints[
                marker
            ] = trimesh.transform_points(
                [self.cephalothoraxMarkers[marker]], rotationMatrix
            )[
                0
            ]

    def project_retinas(self, field_mm):
        self.AME.project_retina(field_mm / self.voxelSize)
        self.ALE.project_retina(field_mm / self.voxelSize)
        self.PME.project_retina(field_mm / self.voxelSize)
        self.PLE.project_retina(field_mm / self.voxelSize)

    def project_retinas_full(self, field_mm):
        self.AME.project_retina_full(field_mm / self.voxelSize)
        self.ALE.project_retina_full(field_mm / self.voxelSize)
        self.PME.project_retina_full(field_mm / self.voxelSize)
        self.PLE.project_retina_full(field_mm / self.voxelSize)

    def find_all_fields_contours(
        self, stepsizes=(500, 500, 500, 500), tolerances=(500, 500, 500, 500)
    ):
        """

        :param stepsizes: the size of slices in each direction in pixels. Always need to be a 4 long list, even with less eyes
        :param tolerances: the span from 0 from where to remove points found as contour, they probably are just plane edges
        :return:
        """
        print("Finding fields of view contours...", end="")
        if "AME" in self.available_eyes:
            self.AME.find_field_contours(stepsizes[0], tolerances[0])
        if "ALE" in self.available_eyes:
            self.ALE.find_field_contours(stepsizes[1], tolerances[1])
        if "PME" in self.available_eyes:
            self.PME.find_field_contours(stepsizes[2], tolerances[2])
        if "PLE" in self.available_eyes:
            self.PLE.find_field_contours(stepsizes[3], tolerances[3])
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
                    "Original": self.AME.LensPoints,
                    "Rotated": self.AME.RotatedLensPoints,
                    "Standard": self.AME.StandardOrientationLensPoints,
                },
                "Retina": {
                    "Original": self.AME.RetinaPoints,
                    "Rotated": self.AME.RotatedRetinaPoints,
                    "Standard": self.AME.StandardOrientationRetinaPoints,
                },
                "Projection": {
                    "Surface": self.AME.StandardOrientationProjectedVectors,
                    "Full": self.AME.StandardOrientationProjectedVectorsFull,
                },
            },
            "ALE": {
                "Lens": {
                    "Original": self.ALE.LensPoints,
                    "Rotated": self.ALE.RotatedLensPoints,
                    "Standard": self.ALE.StandardOrientationLensPoints,
                },
                "Retina": {
                    "Original": self.ALE.RetinaPoints,
                    "Rotated": self.ALE.RotatedRetinaPoints,
                    "Standard": self.ALE.StandardOrientationRetinaPoints,
                },
                "Projection": {
                    "Surface": self.ALE.StandardOrientationProjectedVectors,
                    "Full": self.ALE.StandardOrientationProjectedVectorsFull,
                },
            },
            "PME": {
                "Lens": {
                    "Original": self.PME.LensPoints,
                    "Rotated": self.PME.RotatedLensPoints,
                    "Standard": self.PME.StandardOrientationLensPoints,
                },
                "Retina": {
                    "Original": self.PME.RetinaPoints,
                    "Rotated": self.PME.RotatedRetinaPoints,
                    "Standard": self.PME.StandardOrientationRetinaPoints,
                },
                "Projection": {
                    "Surface": self.PME.StandardOrientationProjectedVectors,
                    "Full": self.PME.StandardOrientationProjectedVectorsFull,
                },
            },
            "PLE": {
                "Lens": {
                    "Original": self.PLE.LensPoints,
                    "Rotated": self.PLE.RotatedLensPoints,
                    "Standard": self.PLE.StandardOrientationLensPoints,
                },
                "Retina": {
                    "Original": self.PLE.RetinaPoints,
                    "Rotated": self.PLE.RotatedRetinaPoints,
                    "Standard": self.PLE.StandardOrientationRetinaPoints,
                },
                "Projection": {
                    "Surface": self.PLE.StandardOrientationProjectedVectors,
                    "Full": self.PLE.StandardOrientationProjectedVectorsFull,
                },
            },
            "cephalothorax": {
                "Original": self.cephalothoraxMarkers,
                "Rotated": self.StandardOrientationCephalothoraxPoints,
            },
        }
        if type == "h5":
            tab = pd.DataFrame(data)
            tab.to_hdf(self.path + filename + ".h5", "tab")
        elif type == "pickle":
            with open(self.path + filename + ".pickle", "wb") as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

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

        self.AME.LensPoints = data["AME"]["Lens"]["Original"]
        self.AME.RotatedLensPoints = data["AME"]["Lens"]["Rotated"]
        self.AME.StandardOrientationLensPoints = data["AME"]["Lens"]["Standard"]
        self.AME.RetinaPoints = data["AME"]["Retina"]["Original"]
        self.AME.RotatedRetinaPoints = data["AME"]["Retina"]["Rotated"]
        self.AME.StandardOrientationRetinaPoints = data["AME"]["Retina"]["Standard"]
        self.AME.StandardOrientationProjectedVectors = data["AME"]["Projection"][
            "Surface"
        ]
        self.AME.StandardOrientationProjectedVectorsFull = data["AME"]["Projection"][
            "Full"
        ]
        self.AME.LensCloud = trimesh.points.PointCloud(self.AME.LensPoints)
        self.AME.RotatedLensCloud = trimesh.points.PointCloud(
            self.AME.RotatedLensPoints
        )
        self.AME.StandardOrientationLensCloud = trimesh.points.PointCloud(
            self.AME.StandardOrientationLensPoints
        )
        self.AME.RetinaCloud = trimesh.points.PointCloud(self.AME.RetinaPoints)
        self.AME.RotatedRetinaCloud = trimesh.points.PointCloud(
            self.AME.RotatedRetinaPoints
        )
        self.AME.StandardOrientationRetinaCloud = trimesh.points.PointCloud(
            self.AME.StandardOrientationRetinaPoints
        )

        self.ALE.LensPoints = data["ALE"]["Lens"]["Original"]
        self.ALE.RotatedLensPoints = data["ALE"]["Lens"]["Rotated"]
        self.ALE.StandardOrientationLensPoints = data["ALE"]["Lens"]["Standard"]
        self.ALE.RetinaPoints = data["ALE"]["Retina"]["Original"]
        self.ALE.RotatedRetinaPoints = data["ALE"]["Retina"]["Rotated"]
        self.ALE.StandardOrientationRetinaPoints = data["ALE"]["Retina"]["Standard"]
        self.ALE.StandardOrientationProjectedVectors = data["ALE"]["Projection"][
            "Surface"
        ]
        self.ALE.StandardOrientationProjectedVectorsFull = data["ALE"]["Projection"][
            "Full"
        ]
        self.ALE.LensCloud = trimesh.points.PointCloud(self.ALE.LensPoints)
        self.ALE.RotatedLensCloud = trimesh.points.PointCloud(
            self.ALE.RotatedLensPoints
        )
        self.ALE.StandardOrientationLensCloud = trimesh.points.PointCloud(
            self.ALE.StandardOrientationLensPoints
        )
        self.ALE.RetinaCloud = trimesh.points.PointCloud(self.ALE.RetinaPoints)
        self.ALE.RotatedRetinaCloud = trimesh.points.PointCloud(
            self.ALE.RotatedRetinaPoints
        )
        self.ALE.StandardOrientationRetinaCloud = trimesh.points.PointCloud(
            self.ALE.StandardOrientationRetinaPoints
        )

        self.PME.LensPoints = data["PME"]["Lens"]["Original"]
        self.PME.RotatedLensPoints = data["PME"]["Lens"]["Rotated"]
        self.PME.StandardOrientationLensPoints = data["PME"]["Lens"]["Standard"]
        self.PME.RetinaPoints = data["PME"]["Retina"]["Original"]
        self.PME.RotatedRetinaPoints = data["PME"]["Retina"]["Rotated"]
        self.PME.StandardOrientationRetinaPoints = data["PME"]["Retina"]["Standard"]
        self.PME.StandardOrientationProjectedVectors = data["PME"]["Projection"][
            "Surface"
        ]
        self.PME.StandardOrientationProjectedVectorsFull = data["PME"]["Projection"][
            "Full"
        ]
        self.PME.LensCloud = trimesh.points.PointCloud(self.PME.LensPoints)
        self.PME.RotatedLensCloud = trimesh.points.PointCloud(
            self.PME.RotatedLensPoints
        )
        self.PME.StandardOrientationLensCloud = trimesh.points.PointCloud(
            self.PME.StandardOrientationLensPoints
        )
        self.PME.RetinaCloud = trimesh.points.PointCloud(self.PME.RetinaPoints)
        self.PME.RotatedRetinaCloud = trimesh.points.PointCloud(
            self.PME.RotatedRetinaPoints
        )
        self.PME.StandardOrientationRetinaCloud = trimesh.points.PointCloud(
            self.PME.StandardOrientationRetinaPoints
        )

        self.PLE.LensPoints = data["PLE"]["Lens"]["Original"]
        self.PLE.RotatedLensPoints = data["PLE"]["Lens"]["Rotated"]
        self.PLE.StandardOrientationLensPoints = data["PLE"]["Lens"]["Standard"]
        self.PLE.RetinaPoints = data["PLE"]["Retina"]["Original"]
        self.PLE.RotatedRetinaPoints = data["PLE"]["Retina"]["Rotated"]
        self.PLE.StandardOrientationRetinaPoints = data["PLE"]["Retina"]["Standard"]
        self.PLE.StandardOrientationProjectedVectors = data["PLE"]["Projection"][
            "Surface"
        ]
        self.PLE.StandardOrientationProjectedVectorsFull = data["PLE"]["Projection"][
            "Full"
        ]
        self.PLE.LensCloud = trimesh.points.PointCloud(self.PLE.LensPoints)
        self.PLE.RotatedLensCloud = trimesh.points.PointCloud(
            self.PLE.RotatedLensPoints
        )
        self.PLE.StandardOrientationLensCloud = trimesh.points.PointCloud(
            self.PLE.StandardOrientationLensPoints
        )
        self.PLE.RetinaCloud = trimesh.points.PointCloud(self.PLE.RetinaPoints)
        self.PLE.RotatedRetinaCloud = trimesh.points.PointCloud(
            self.PLE.RotatedRetinaPoints
        )
        self.PLE.StandardOrientationRetinaCloud = trimesh.points.PointCloud(
            self.PLE.StandardOrientationRetinaPoints
        )

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
                lens = self.AME.StandardOrientationLensCloud.convex_hull.vertices
                ax.scatter(lens[:, 0], lens[:, 1], lens[:, 2], color="purple")
            if "retina" in elements:
                retina = self.AME.StandardOrientationRetinaCloud.convex_hull.vertices
                ax.scatter(retina[:, 0], retina[:, 1], retina[:, 2], color="purple")
            if "projection" in elements:
                Project = np.array(self.AME.StandardOrientationProjectedVectors)[:, 2]
                ax.scatter(
                    Project[:, 0],
                    Project[:, 1],
                    Project[:, 2],
                    color="indigo",
                    alpha=alpha,
                )
            if "projection_full" in elements:
                Project = np.array(self.AME.StandardOrientationProjectedVectorsFull)[
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
                Outline = self.AME.FOVcontourPoints
                ax.scatter(
                    Outline[:, 0],
                    Outline[:, 1],
                    Outline[:, 2],
                    color="indigo",
                    alpha=alpha,
                )
        if "ALE" in eyes and "ALE" in self.available_eyes:
            if "lens" in elements:
                lens = self.ALE.StandardOrientationLensCloud.convex_hull.vertices
                ax.scatter(lens[:, 0], lens[:, 1], lens[:, 2], color="green")
            if "retina" in elements:
                retina = self.ALE.StandardOrientationRetinaCloud.convex_hull.vertices
                ax.scatter(retina[:, 0], retina[:, 1], retina[:, 2], color="green")
            if "projection" in elements:
                Project = np.array(self.ALE.StandardOrientationProjectedVectors)[:, 2]
                ax.scatter(
                    Project[:, 0],
                    Project[:, 1],
                    Project[:, 2],
                    color="darkgreen",
                    alpha=alpha,
                )
            if "projection_full" in elements:
                Project = np.array(self.ALE.StandardOrientationProjectedVectorsFull)[
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
                Outline = self.ALE.FOVcontourPoints
                ax.scatter(
                    Outline[:, 0],
                    Outline[:, 1],
                    Outline[:, 2],
                    color="darkgreen",
                    alpha=alpha,
                )
        if "PME" in eyes and "PME" in self.available_eyes:
            if "lens" in elements:
                lens = self.PME.StandardOrientationLensCloud.convex_hull.vertices
                ax.scatter(lens[:, 0], lens[:, 1], lens[:, 2], color="goldenrod")
            if "retina" in elements:
                retina = self.PME.StandardOrientationRetinaCloud.convex_hull.vertices
                ax.scatter(retina[:, 0], retina[:, 1], retina[:, 2], color="goldenrod")
            if "projection" in elements:
                Project = np.array(self.PME.StandardOrientationProjectedVectors)[:, 2]
                ax.scatter(
                    Project[:, 0],
                    Project[:, 1],
                    Project[:, 2],
                    color="darkgoldenrod",
                    alpha=alpha,
                )
            if "projection_full" in elements:
                Project = np.array(self.PME.StandardOrientationProjectedVectorsFull)[
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
                Outline = self.PME.FOVcontourPoints
                ax.scatter(
                    Outline[:, 0],
                    Outline[:, 1],
                    Outline[:, 2],
                    color="darkgoldenrod",
                    alpha=alpha,
                )
        if "PLE" in eyes and "PLE" in self.available_eyes:
            if "lens" in elements:
                lens = self.PLE.StandardOrientationLensCloud.convex_hull.vertices
                ax.scatter(lens[:, 0], lens[:, 1], lens[:, 2], color="darkred")
            if "retina" in elements:
                retina = self.PLE.StandardOrientationRetinaCloud.convex_hull.vertices
                ax.scatter(retina[:, 0], retina[:, 1], retina[:, 2], color="darkred")
            if "projection" in elements:
                Project = np.array(self.PLE.StandardOrientationProjectedVectors)[:, 2]
                ax.scatter(
                    Project[:, 0],
                    Project[:, 1],
                    Project[:, 2],
                    color="maroon",
                    alpha=alpha,
                )
            if "projection_full" in elements:
                Project = np.array(self.PLE.StandardOrientationProjectedVectorsFull)[
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
                Outline = self.PLE.FOVcontourPoints
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

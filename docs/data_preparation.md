# Practical Guide for Segmentation and Export
The ctSpyderFields software uses manually segmented ROI of lenses and retinas of spiders as produced in a 3D segmentation tool to calculate the visual fields.

The ROIs must be extracted from the software in question to be fed to the python package. This is done by extracting the ROIs in tiff stacks.

Here we provide the guide to do so with 2 commons pieces of software, AMIRA (closed) and DragonFly (open), but you can use any software you like. 
The important part is that ROIs can be extracted in sliced images. These images can either be binary maps (so 1 stack for each ROI, with white pixels
being part of the ROI, black being outside), or colored images (so 1 stack only, with each ROI and the background being assigned a specific color).

You will have to match the type of image stack in the python object. See  [usage.md](https://github.com/massimodeagro/ctSpyderFields/blob/main/docs/usage.md).

If you don't want to segment all eyes, but only some, you can do so. remember again to specify it in your python object. See  [usage.md](https://github.com/massimodeagro/ctSpyderFields/blob/main/docs/usage.md).

Other than the eyes, you will **HAVE** to segment a set of 7 body markers: center, front, back, left, right, top, bottom. These can be any size, even 
a single voxel. The software will use their center to reorient the spider in a known reference frame. During segmentation, you can use real 
anatomical markers or arbitrary locations. The package will reorient the full reference frame aligning the X axis to center-front, 
the Y axis to center-left and the Z axis to center-top.

## Dragonfly
1)	Create a separate ROI for each individual lens, retina, and cephalothorax marker. A new ROI can be created using the “segment” window on the left-hand side as seen in the figure below:

![segment window](images/segment_window.png)

Under “Basic”, click “New….”, select the appropriate name and colour (either using the defaults or user specified colours stored under “params” in your ctSpyderFields folder) and click OK

 ![new region](images/new_region.png)

2)	Segment each individual lens, retina, and cephalothorax marker as a separate ROI using the “ROI painter” tool, also located on the left-hand side window:
   
![roi painter](images/roi_painter.png)
 
![roi painted](images/roi_painted.png)

3)	After segmenting, right click on the ROI object (all ROIs are located in the right-hand side window under “Data Properties and Settings”) and go to “Export -> ROI as binary” and store as an image stack in your desired output folder. 
DragonFly can only use binary mode, not color. Make sure to use the appropriate name for each component e.g., “Lens_AME”, “Retina_AME”, etc. The files should end up all in the same folders, to be provided 
to the package through the `workdir` argument (see [usage.md](https://github.com/massimodeagro/ctSpyderFields/blob/main/docs/usage.md)). The image stacks names 
should be provided in the `name_params.yaml` (see [usage.md](https://github.com/massimodeagro/ctSpyderFields/blob/main/docs/usage.md)).


## Amira

1)	Go to the “Segmentation” window and create a separate label file for each individual lens, retina, and cephalothorax marker:

![amira segmentation](images/amira_segmentation.png)

2) If using “binary” option in ctSpyderFields for loading the elements, set the exterior to black and the material (“Inside”) to white, otherwise use the colours specified under `color_params.yaml` in your ctSpyderFields folder
3) Segment using the desired tool:
   
![amira segmented](images/amira_segmented.png)

The various segmentation tools are visible under “Selection” on the right side:

![tools selection](images/segmentation_tools_section.png)

4)	After segmentation, export all the labels to the desired folder. To export a label, right click on the label, select “Export data as” and choose “2D TIF” as the data type.
If you are using binary, you need one stack per ROI, as in Dragonfly. Remember to give names to then write in the `name_params.yaml`. if you are using color, you will have
a single stack. still fill your `name_params.yaml`, just repeat the same filename for every element, and provide your `color_params.yaml`.


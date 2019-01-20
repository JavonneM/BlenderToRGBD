# BlenderToRGBD

This script uses *Blender* to generate RGB-D data in the format of the **Kinect V2** from a blender file using the animation system built in.
This script iterates through each of the frames and renders the RGB frame, extracts the depth image and the ground truth.

This extracts the RGBD information from a camera in the blender scene while iterating through each key frame in the Blender animation. The script is used
as follows:
```
blender *BlenderFile* -P generateRGBD.py -- *Outputfolder* -r
```
```
blender ~/TestTranslation.blend -P generateRGBD.py -- ~/ThisIsATest/ -r
```
**NOTE, PLEASE ENSURE THAT THE OUTPUTFOLDER IS EMPTY**

## Generated file format

This script generates the following files in the *OutputFolder*:
* rgb.txt
* depth.txt
* groundtruth.txt

With this the following folders are generated:
* rgb
* depth

The folders *rgb* and *depth* contain RGB files and Depth images generated from the script.

## RGB and Depth file format

The rgb.txt and depth.txt files contain the frame number and location of each rgb and depth file that has been generated
during the render process. The files have the following format:
*frame* *location*

Please note that the *location* variable is relative to the *OutputFolder* for example a rgb image is shown from frame 9
'9 rgb/Image_0009.png'

The depth files as stored as uint16 values and are scaled by a value of 5. In order to compute the actualy distance 
simply compute the actual value in the units set in the *Blender* file as follows

*distance = pixelValue/5* in units specified by the blender file


## Groundtruth File

The groundtruth file is generated as 'groundtruth.txt'. This file has the groundtruth in the following format
*frame* *x* *y* *z* *qx* *qy* *qz* *qw*

The groundtruth uses the x, y and z for the translation and qx qy qz and qw for the rotation. The rotation is represented
as a quaternions.

## Using the generated data
To use the data in your own custom scripts please load the *rgb.txt* and *depth.txt* files to locate the generated files
RGB and depth images.


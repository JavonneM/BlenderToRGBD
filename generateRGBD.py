"""
MIT License

Copyright (c) 2017 Javonne Jason Martin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import bpy
import numpy as np
import mathutils
from mathutils import Matrix
import os
import sys
from skimage import io
import skimage

ImageWidth = 640
ImageHeight = 480
ox = ImageWidth/2
oy = ImageHeight/2
fx = 588
fy = 588
K = np.array([[fx, 0, ox], [0, fy, oy], [0, 0, 1]], dtype=np.float32)
Kinv = np.linalg.inv(K)
CameraFOV = 57
MaxDepth = 1500#cmeters ?
DepthScale = 5

BlenderFile = ""
BasePath = './GeneratedImages/'
RGBPath = 'rgb/'
DepthPath = 'depth/'
EXRDepthPath = 'EXRdepth/'

GroundTruth = 'groundtruth.txt'
RGB = 'rgb.txt'
Depth = 'depth.txt'

RGBFileNameFormat = 'Image_'
EXT = '.png'


def set_base_path(base_path):
    global BasePath
    BasePath = base_path
    if not os.path.exists(base_path):
        os.makedirs(base_path + RGBPath)
        os.makedirs(base_path + DepthPath)
        os.makedirs(base_path + EXRDepthPath)


def set_camera_fov(fov):
    global CameraFOV
    CameraFOV = fov


def set_scale(scale):
    global DepthScale
    DepthScale = scale


def set_image_width_and_height(width, height):
    global ImageWidth
    global ImageHeight
    global ox
    global oy
    global K
    global Kinv
    ImageWidth = width
    ImageHeight = height
    ox = ImageWidth/2
    oy = ImageHeight/2
    K = np.array([[fx, 0, ox], [0, fy, oy], [0, 0, 1]], dtype=np.float32)
    Kinv = np.linalg.inv(K)


def set_depth_threshold(threshold):
    global MaxDepth
    MaxDepth = threshold


def set_camera_properties(camera_name="Camera", inv_camera=True):
    camera = bpy.data.cameras[camera_name]
    camera.type = 'PERSP'
    camera.lens_unit = 'FOV'
    camera.angle = np.radians(CameraFOV)
    camera.sensor_fit = 'AUTO'
    camera.sensor_width = 32.0
    # clip start/end in unit defined by Scene Should be meters
    print(camera.clip_start, type(camera.clip_start))
    print(camera.clip_end, type(camera.clip_end))
    camera.clip_start = 0.4  # meters
    camera.clip_end = 5  # meters
    # bpy.data.objects[cameraName].rotation_mode = 'QUATERNION'
    if inv_camera:
        print("InvertCamera is True", inv_camera)
        bpy.data.objects[camera_name].scale = 1, -1, -1


def set_renderer_properties(scene):
    print(scene.render.engine, type(scene.render.engine))
    scene.render.engine = 'CYCLES'
    scene.render.resolution_x = ImageWidth
    scene.render.resolution_y = ImageHeight
    scene.render.resolution_percentage = 100


def set_scene_properties(scene_name='Scene'):
    print(bpy.data.scenes[scene_name], type(bpy.data.scenes[scene_name]))
    scene = bpy.data.scenes[scene_name]
    set_renderer_properties(scene)
    scene.unit_settings.system = 'METRIC'
    scene.unit_settings.system_rotation = 'RADIANS'


def get_key_frame(scene_name='Scene'):
    scene = bpy.data.scenes[scene_name]
    return scene.frame_current


def get_camera(camera_name='Camera'):
    return bpy.data.cameras[camera_name]


def get_camera_as_object(camera_name='Camera'):
    return bpy.data.objects[camera_name]


def get_camera_transformation(camera_name='Camera'):
    camera = get_camera_as_object(camera_name)
    return camera.matrix_world


def get_scene_start_frame(scene='Scene'):
    return bpy.data.scenes[scene].frame_start


def get_scene_end_frame(scene='Scene'):
    return bpy.data.scenes[scene].frame_end


def update_scene(scene='Scene'):
    bpy.data.scenes[scene].update()


def increment_key_frame(scene_name='Scene'):
    scene = bpy.data.scenes[scene_name]
    scene.frame_set(scene.frame_current + 1)


def reset_key_frame(scene_name='Scene'):
    scene = bpy.data.scenes[scene_name]
    scene.frame_set(0)


def build_nodes():
    print("Building Nodes")
    # switch on nodes
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    # clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)

    # create input render layer node
    rl = tree.nodes.new('CompositorNodeRLayers')
    rl.location = 185,285

    # create output node
    v = tree.nodes.new('CompositorNodeViewer')
    v.location = 750,80
    v.use_alpha = False

    multiplier = tree.nodes.new('CompositorNodeMapValue')
    multiplier.location = 450, 80
    multiplier.size[0] = 1000

    output_node = tree.nodes.new('CompositorNodeOutputFile')
    output_node.location = 750, 285
    output_node.base_path = BasePath + RGBPath
    output_node.file_slots[0].path = RGBFileNameFormat
    output_node.file_slots[0].use_node_format = True
    output_node.format.color_mode = 'RGBA'
    output_node.format.color_depth = '8'
    output_node.format.file_format = 'PNG'

    outputNodeZbuffer = tree.nodes.new('CompositorNodeOutputFile')
    outputNodeZbuffer.location = 750,185
    outputNodeZbuffer.base_path = BasePath + EXRDepthPath
    outputNodeZbuffer.file_slots[0].path = RGBFileNameFormat
    outputNodeZbuffer.file_slots[0].use_node_format = False
    outputNodeZbuffer.file_slots[0].format.file_format = "OPEN_EXR"
    outputNodeZbuffer.file_slots[0].format.color_mode = 'RGB'
    print(outputNodeZbuffer.format.color_depth)
    outputNodeZbuffer.file_slots[0].format.color_depth = '32'
    # Links
    links.new(rl.outputs[2], multiplier.inputs[0])
    links.new(multiplier.outputs[0], v.inputs[0])  # link Image output to Viewer input
    links.new(rl.outputs[0], output_node.inputs[0])  # link Image output to Viewer input
    links.new(rl.outputs[2], outputNodeZbuffer.inputs[0])


def get_depth_map():
    # get viewer pixels
    pixels = bpy.data.images['Viewer Node'].pixels
    # zMap reports depth in mm from camera
    # copy buffer to numpy array for faster manipulation

    depthMap = np.zeros((ImageHeight, ImageWidth), dtype=np.uint16)
    zBuffer = np.array(pixels[:])
    print(zBuffer.shape)
    for c in range(0, ImageWidth):
        for r in range(0, ImageHeight):
            # Blender stores the image in column major format, numpy uses row major
            index = r * ImageWidth * 4 + c * 4
            red = zBuffer[index]
            green = zBuffer[index + 1]
            blue = zBuffer[index + 2]
            alpha = zBuffer[index + 3]
            if (red != green or green != blue):
                print("Failed", alpha, red, green, blue)
                return;
            depth = red
            if(depth > MaxDepth):
                depth = 0
            # New fix for the warped image
            # Z map reports the length along the ray
            # I am interested in the Z component of the vector that represents the point
            v = np.array([c, r, 1], dtype=np.float32)
            n = Kinv.dot(v)
            n = n / np.linalg.norm(n);
            n = n * depth
            # Dirty fix, Some reason the image is flipped on the x axis 'ImageHeight - 1 - r' to correct
            # depthMap[ImageHeight - 1 - r, c] = np.uint16(depth * DepthScale)
            depthMap[ImageHeight - 1 - r, c] = np.uint16(n[2] * DepthScale)

    return depthMap


def build_translation_and_rotation(x, y, z, rx, ry, rz):
    rotation = mathutils.Vector()
    rotation.x = np.radians(rx)
    rotation.y = np.radians(ry)
    rotation.z = np.radians(rz)
    translation = mathutils.Vector()
    translation.x = x
    translation.y = y
    translation.z = z
    return translation, rotation


def set_camera_location_and_rotation(translation, rotation, camera_name='Camera'):
    camera = bpy.data.objects[camera_name]
    camera.location = translation
    camera.rotation_quaternion = rotation


def save_data(depth_maps, all_translations=None, all_rotations=None, timestamps=None):
    print("Starting Save")
    if not len(depth_maps) == len(all_translations) or not len(depth_maps) == len(all_rotations):
        print('ERROR')
    print("Opening files")
    print(len(all_translations))
    print(len(all_rotations))
    print(len(timestamps))
    if not os.path.isdir(BasePath + DepthPath):
        os.mkdir(BasePath + DepthPath)
    ground_truth = open(BasePath + GroundTruth, 'w')
    depth = open(BasePath + Depth, 'w')
    rgb = open(BasePath + RGB, 'w')
    print("Saving Files")
    if len(depth_maps) == 0:
        for i in range(len(timestamps)):
            depthName = str(timestamps[i]) + '.png'
            depth.write(str(timestamps[i]) + ' ' + DepthPath + depthName + '\n')
            num = str(timestamps[i])
            if timestamps[i] < 1000:
                num = '0'+num
            if timestamps[i] < 100:
                num = '0'+num
            if timestamps[i] < 10:
                num = '0'+num
            rgb.write(str(timestamps[i]) + ' ' + RGBPath + RGBFileNameFormat + num + EXT + '\n')
            ground_truth.write(str(timestamps[i]) + ' ' + "%.4f" % all_translations[i].x + ' ' + "%.4f" % all_translations[i].y + ' ' + "%.4f" % all_translations[i].z + ' ' +
                            "%.4f" % all_rotations[i].x + ' ' + "%.4f" % all_rotations[i].y + ' ' + "%.4f" % all_rotations[i].z + ' ' + "%.4f" % all_rotations[i].w + '\n')

    for i in range(len(depth_maps)):
        # print("Iteration ", i)
        depthName = str(timestamps[i]) + '.png'
        depth.write(str(timestamps[i]) + ' ' + DepthPath + depthName + '\n')
        depthMap = depth_maps[i]
        # plugin broken for some reason
        io.imsave(BasePath + DepthPath + depthName, skimage.img_as_uint(depthMap), plugin='freeimage')
        # io.imsave(BasePath + DepthPath + depthName, skimage.img_as_uint(depthMap))
        num = str(timestamps[i])
        if(timestamps[i] < 1000):
            num = '0'+num
        if(timestamps[i] < 100):
            num = '0'+num
        if(timestamps[i] < 10):
            num = '0'+num
        rgb.write(str(timestamps[i]) + ' ' + RGBPath + RGBFileNameFormat + num + EXT + '\n')
        rotation_quaternion = all_rotations[i]
        ground_truth.write(str(timestamps[i]) + ' ' + "%.4f" % all_translations[i].x + ' ' + "%.4f" % all_translations[i].y + ' ' + "%.4f" % all_translations[i].z + ' ' +
                            "%.4f" % all_rotations[i].x + ' ' + "%.4f" % all_rotations[i].y + ' ' + "%.4f" % all_rotations[i].z + ' ' + "%.4f" % all_rotations[i].w + '\n')
    print("Closing")
    depth.close()
    ground_truth.close()
    rgb.close()
    print("Ending Save")


def animation(render=False):
    translation = mathutils.Vector()
    rotation = mathutils.Matrix.Rotation(0, 3, 'X')
    allTranslations = []
    allRotations = []
    allDepthMaps = []
    timestamps = []
    sceneStart = get_scene_start_frame()
    sceneEnd = get_scene_end_frame()
    print(sceneStart, sceneEnd)
    for i in range(sceneStart, sceneEnd):
        # UpdateScene()
        # GetRotation and translation
        T = get_camera_transformation()
        translation = T.to_translation()
        rotation = T.to_3x3()
        print('Index', i, 'test',  bpy.data.scenes['Scene'].frame_current)
        print(rotation)
        # Render
        if render:
            bpy.ops.render.render(animation=False)
            depthMap = get_depth_map()
            # add data to list
            allTranslations.append(mathutils.Vector(translation))
            allRotations.append(rotation.to_quaternion())
            allDepthMaps.append(depthMap)
            timestamps.append(get_key_frame())
        else:

            allTranslations.append(mathutils.Vector(translation))
            allRotations.append(rotation.to_quaternion())
            timestamps.append(get_key_frame())
        # updateKeyFrame
        increment_key_frame()
        # return allDepthMaps, allTranslations, allRotations, timestamps
        # update Rotation/Translation
    reset_key_frame()
    return allDepthMaps, allTranslations, allRotations, timestamps


def get_calibration_matrix_K_from_blender(camd):
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if camd.sensor_fit == 'VERTICAL':
        # the sensor height is fixed (sensor fit is horizontal),
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else: # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal),
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px*scale / 2
    v_0 = resolution_y_in_px*scale / 2
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((alpha_u, skew,    u_0),
        (0 ,  alpha_v, v_0),
        (0 ,    0,      1)))
    return K


def inspect_depth(depth_maps):
    image = depth_maps[0]
    print(image.shape)
    f = open("DepthImage.txt", 'w')
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            f.write(str(image[i][j]) + " ")
        f.write("\n")


def test(directory):
    set_base_path(directory)
    set_camera_fov(57)
    set_scale(1)
    set_image_width_and_height(160, 120)
    set_depth_threshold(10 * 1000) # 10 meters -> converted to mm
    build_nodes()
    set_camera_properties(inv_camera=False)
    print(ImageWidth, ImageHeight)
    set_scene_properties()

    depth_maps, all_translations, all_rotations, timestamps = animation(render=True)
    save_data(depth_maps, all_translations, all_rotations, timestamps)
    # InspectDepth(depthMaps)
    k = get_calibration_matrix_K_from_blender(bpy.data.objects['Camera'].data)
    print(k)


def main():
    argv = sys.argv
    argv = argv[argv.index("--") :]
    if len(argv) < 2:
        print("\t OutputDir Name")
        print("\t -r to render")
        exit(0);
    print(argv)
    directory = argv[1]
    render = False
    test_mode = False
    if len(argv) > 2:
        if argv[2] == '-r':
            render=True
        if argv[2] == '-t':
            test_mode = True

    if test_mode:
        test(directory)
        return
    # Check if file exisits
    if not os.path.isdir(directory):
        print("Can't find directory", directory)
        return
    set_base_path(directory)
    set_camera_fov(57)
    set_scale(5)
    set_image_width_and_height(640, 480)
    set_depth_threshold(5 * 1000) # 10 meters -> converted to mm
    build_nodes()
    set_camera_properties(inv_camera=False)
    set_scene_properties()
    # rotation = mathutils.Matrix.Rotation(-(3 * np.pi)/4, 3, 'X')
    # depthMaps, allTranslations, allRotations, timestamps = MoveAroundPoint(pathStart, center, \
    # distance, 100, initRotation=rotation, render=True)
    # rotation = mathutils.Matrix.Rotation(0, 3, 'X')
    # depthMaps, allTranslations, allRotations, timestamps = MoveAlongLine(pathStart, pathEnd, 200,\
    #  initRotation=rotation, render=True)

    depth_maps, all_translations, all_rotations, timestamps = animation(render=render)
    print("Ready")
    print("direc ", directory)
    save_data(depth_maps, all_translations, all_rotations, timestamps)
    print(get_calibration_matrix_K_from_blender(bpy.data.objects['Camera'].data))
    print('Data Set Generated')


if __name__ == '__main__':
    main()

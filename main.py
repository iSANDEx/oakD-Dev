# USAGE
# python main.py --demo color_camera
# python main.py --demo mono_cameras
# python main.py --demo object_detection

# import the necessary packages
from dai_tools.color_camera_preview import color_camera, create_color_camera_pipeline
from dai_tools.left_right_mono_camera_preview import create_mono_camera_pipeline, mono_cameras_preview
from dai_tools.object_detection_mobilenet import create_detection_pipeline, object_detection_mobilenet
import argparse

# define the argparser and parse the command line arguments
parser = argparse.ArgumentParser(description='OpenCV AI Kit Examples')
parser.add_argument(
    '-d', '--demo', type=str, default='color_camera',
    help='Run color camera or mono cameras or object detection exampes',
)
args = parser.parse_args()

# if demo is color_camera then call create_color_camera_pipeline()
# then pass the pipeline to color_camera method for rgb preview
if args.demo == 'color_camera':
    pipeline = create_color_camera_pipeline()
    color_camera(pipeline=pipeline)

# if demo is mono_cameras then call create_mono_camera_pipeline()
# pass the pipeline to mono_cameras_preview for displaying left &
# right grayscale camera feed
elif args.demo == 'mono_cameras':
    pipeline = create_mono_camera_pipeline()
    mono_cameras_preview(pipeline=pipeline)

# if demo is object_detection then call create_detection_pipeline()
# then pass the pipeline to object_detection_mobilenet to run object
# detection on OAK
elif args.demo == 'object_detection':
    pipeline = create_detection_pipeline()
    object_detection_mobilenet(pipeline=pipeline)
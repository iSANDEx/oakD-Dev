# import the necessary packages
from dai_tools import config
import depthai as dai
import cv2

def create_color_camera_pipeline():
    # create pipeline
    pipeline = dai.Pipeline()

    # define source and output camera node
    camRgb = pipeline.create(dai.node.ColorCamera)

    # XLinkOut node for displaying frames
    xoutRgb = pipeline.create(dai.node.XLinkOut)

    # set stream name as rgb
    xoutRgb.setStreamName('rgb')

    # set camera properties like the preview window, interleaved
    camRgb.setPreviewSize(config.COLOR_CAMERA_PREVIEW_SIZE)
    camRgb.setInterleaved(config.CAMERA_INTERLEAVED)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

    # link the camera preview to XLinkOut node input
    camRgb.preview.link(xoutRgb.input)

    # return pipeline to the calling function
    return pipeline


def color_camera(pipeline):
    # connect to device and start pipeline
    with dai.Device(pipeline) as device:
        print('Connected cameras: ', device.getConnectedCameras())
        # print out usb speed like low/high
        print('Usb speed: ', device.getUsbSpeed().name)

        # output queue will be used to get the rgb
        # frames from the output defined above
        qRgb = device.getOutputQueue(
            name='rgb',
            maxSize=config.COLOR_CAMERA_QUEUE_SIZE,
            blocking=config.QUEUE_BLOCKING,
        )

        while True:
            # blocking call, will wait until a new data has arrived
            inRgb = qRgb.get()

            # convert the rgb frame data to OpenCV format and
            # display 'bgr' (opencv format) frame
            cv2.imshow('rgb', inRgb.getCvFrame())

            # break out from the while loop if `q` key is pressed
            if cv2.waitKey(1) == ord('q'):
                break
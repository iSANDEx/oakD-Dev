# import the necessary packages
from dai_tools import config
import depthai as dai
import cv2

def create_mono_camera_pipeline():
    # create pipeline
    pipeline = dai.Pipeline()

    # define sources and outputs: creating left and right camera nodes
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)

    # XLinkOut nodes for displaying frames from left and right camera
    xoutLeft = pipeline.create(dai.node.XLinkOut)
    xoutRight = pipeline.create(dai.node.XLinkOut)

    # set XLinkOut stream name as left and right for later using in
    # OutputQueue
    xoutLeft.setStreamName('left')
    xoutRight.setStreamName('right')

    # set mono camera properties like which camera socket to use,
    # camera resolution
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoLeft.setResolution(
        dai.MonoCameraProperties.SensorResolution.THE_720_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    monoRight.setResolution(
        dai.MonoCameraProperties.SensorResolution.THE_720_P,
    )

    # link the left and right camera output to XLinkOut node input
    monoRight.out.link(xoutRight.input)
    monoLeft.out.link(xoutLeft.input)

    # return pipeline to the calling function
    return pipeline


def mono_cameras_preview(pipeline):
    # connect to device and start pipeline
    with dai.Device(pipeline) as device:
        # output queues will be used to get the grayscale
        # frames from the outputs defined above
        qLeft = device.getOutputQueue(
            name='left',
            maxSize=config.COLOR_CAMERA_QUEUE_SIZE,
            blocking=config.QUEUE_BLOCKING,
        )
        qRight = device.getOutputQueue(
            name='right',
            maxSize=config.COLOR_CAMERA_QUEUE_SIZE,
            blocking=config.QUEUE_BLOCKING,
        )

        while True:
            # instead of get (blocking), we use tryGet (non-blocking)
            # which will return the available data or None otherwise
            inLeft = qLeft.tryGet()
            inRight = qRight.tryGet()

            # check if data is available from left camera node
            if inLeft is not None:
                # convert the left camera frame data to OpenCV format and
                # display grayscale (opencv format) frame
                cv2.imshow('left', inLeft.getCvFrame())

            # check if data is available from right camera node
            if inRight is not None:
                # convert the right camera frame data to OpenCV format and
                # display grayscale (opencv format) frame
                cv2.imshow('right', inRight.getCvFrame())

            # break out from the while loop if `q` key is pressed
            if cv2.waitKey(1) == ord('q'):
                break
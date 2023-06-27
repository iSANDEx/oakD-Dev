# import the necessary packages
from dai_tools import config
from dai_tools.utils import print_neural_network_layer_names
from dai_tools.utils import displayFrame
import depthai as dai
import time
import cv2

def create_detection_pipeline():
    # create pipeline
    pipeline = dai.Pipeline()

    # define camera node
    camRgb = pipeline.create(dai.node.ColorCamera)

    # define the MobileNetDetectionNetwork node
    nn = pipeline.create(dai.node.MobileNetDetectionNetwork)

    # define three XLinkOut nodes for RGB frames, Neural network detections
    # and Neural network metadata for sending to host
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    nnOut = pipeline.create(dai.node.XLinkOut)
    nnNetworkOut = pipeline.create(dai.node.XLinkOut)

    # set the XLinkOut node names
    xoutRgb.setStreamName('rgb')
    nnOut.setStreamName('nn')
    nnNetworkOut.setStreamName('nnNetwork')

    # set camera properties like the preview window, interleaved and
    # camera FPS
    camRgb.setPreviewSize(config.COLOR_CAMERA_PREVIEW_SIZE)
    camRgb.setInterleaved(config.CAMERA_INTERLEAVED)
    camRgb.setFps(config.CAMERA_FPS)

    # define neural network hyperparameters like confidence threshold,
    # number of inference threads. The NN will make predictions
    # based on the source frames
    nn.setConfidenceThreshold(config.NN_THRESHOLD)
    nn.setNumInferenceThreads(config.INFERENCE_THREADS)

    # set mobilenet detection model blob path
    nn.setBlobPath(config.MOBILENET_DETECTION_MODEL_PATH)
    nn.input.setBlocking(False)

    # link the camera preview to XLinkOut node input
    camRgb.preview.link(xoutRgb.input)

    # camera frames linked to NN input node
    camRgb.preview.link(nn.input)

    # NN out (image detections) linked to XLinkOut node
    nn.out.link(nnOut.input)

    # NN unparsed inference results  (metadata) linked to XLinkOut node
    nn.outNetwork.link(nnNetworkOut.input)

    # return pipeline to the calling function
    return pipeline


def object_detection_mobilenet(pipeline):
    # connect to device and start pipeline
    with dai.Device(pipeline) as device:
        # output queues will be used to get the rgb frames
        # and nn data from the outputs defined above
        qRgb = device.getOutputQueue(
            name='rgb',
            maxSize=config.COLOR_CAMERA_QUEUE_SIZE,
            blocking=config.QUEUE_BLOCKING,
        )
        qDet = device.getOutputQueue(
            name='nn',
            maxSize=config.COLOR_CAMERA_QUEUE_SIZE,
            blocking=config.QUEUE_BLOCKING,
        )
        qNN = device.getOutputQueue(
            name='nnNetwork',
            maxSize=config.COLOR_CAMERA_QUEUE_SIZE,
            blocking=config.QUEUE_BLOCKING,
        )

        # initialize frame, detections list, and startTime for
        # computing FPS
        frame = None
        detections = []
        startTime = time.monotonic()
        counter = 0

        # color pattern for displaying FPS
        color2 = config.TEXT_COLOR2  

        # boolean variable for printing NN layer names on console
        printOutputLayersOnce = config.PRINT_NEURAL_NETWORK_METADATA

        while True:
            # instead of get (blocking), we use tryGet (non-blocking)
            # which will return the available data or None otherwise
            # grab the camera frames, image detections, and NN 
            # metadata
            inRgb = qRgb.tryGet()
            inDet = qDet.tryGet()
            inNN = qNN.tryGet()

            # check if we frame is available from the camera
            if inRgb is not None:
                # convert the camera frame to OpenCV format
                frame = inRgb.getCvFrame()

                # annotate the frame with FPS information
                cv2.putText(
                    frame, 'NN fps: {:.2f}'.
                    format(counter / (time.monotonic() - startTime)),
                    (2, frame.shape[0] - 4),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.4, color2,
                )

            # check if detections are available
            if inDet is not None:
                # fetch detections & increment the counter for FPS computation
                detections = inDet.detections
                counter += 1

            # check if the flag is set and NN metadata is available
            if printOutputLayersOnce and inNN is not None:
                # call the `neural network layer names method and pass
                # inNN queue object which would help extract layer names
                print_neural_network_layer_names(inNN)
                printOutputLayersOnce = False

            # if the frame is available, draw bounding boxes on it
            # and show the frame
            if frame is not None:
                displayFrame('object_detection', frame, detections)

            # break out from the while loop if `q` key is pressed
            if cv2.waitKey(1) == ord('q'):
                break
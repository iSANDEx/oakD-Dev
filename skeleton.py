"""
Sample template to create a pipeline. Based on the pyimagesearch tutorial:
https://pyimagesearch.com/2022/12/19/oak-d-understanding-and-running-neural-network-inference-with-depthai-api/?_ga=2.66028718.1206299505.1687552548-1862017530.1687552548

"""

# Import the necessary packages:
import depthai as dai
import cv2
# Additional modules can be imported here:
# ...
from dai_tools.utils import print_neural_network_layer_names, displayFrame
import time

from dai_tools import config

def setup_pipeline():
    # 1) Create a pipeline object which hosts the nodes and communications links between them:
    pipeline= dai.Pipeline()
    # Additional options and constraints to the pipeline object
    # e.g. limit to a specific OpenVINO version:
    pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2022_1)

    # 2) Create nodes, configure and link them together, e.g.:
    # Mono Cameras source nodes:
    monoCamLeftNode = pipeline.create(dai.node.MonoCamera)
    monoCamRightNode = pipeline.create(dai.node.MonoCamera)
    # RGB Camera source node:
    rgbCamNode = pipeline.create(dai.node.ColorCamera)
    # Detection node (MobileNetDetection):
    nn = pipeline.create(dai.node.MobileNetDetectionNetwork)

    # XLinkOut nodes to display the frames from the cameras:
    xOutLeft = pipeline.create(dai.node.XLinkOut)
    xOutRight = pipeline.create(dai.node.XLinkOut)
    xOutRGB = pipeline.create(dai.node.XLinkOut)

    # Neural network detections and Neural network metadata for sending to host
    nnOut = pipeline.create(dai.node.XLinkOut)
    nnNetworkOut = pipeline.create(dai.node.XLinkOut)

    # To ease the identification in the Host, set names:
    xOutLeft.setStreamName('Left')
    xOutRight.setStreamName('Right')
    xOutRGB.setStreamName('RGB')
    nnOut.setStreamName('nn')
    nnNetworkOut.setStreamName('nnNet')

    # Of course just assigning names won't tell which socket is which
    # It needs to get explicitely stated:
    monoCamRightNode.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    monoCamLeftNode.setBoardSocket(dai.CameraBoardSocket.LEFT)
    # (RIGHT and LEFT are just int constants)
    # For the Colour Camera, don't need the following line, 
    # because there is only one camera in the OAK-D/Lite
    # rgbCamNode.setBoardSocket(dai.CameraBoardSocket.RGB)

    # It is also possible to set some camera parameters,
    # such as resolution, image orientation, etc. 
    # through the MonoCameraProperties structure (see https://docs.luxonis.com/projects/api/en/latest/references/cpp/?highlight=MonoCameraProperties#_CPPv4N3dai20MonoCameraPropertiesE)
    # e.g.:
    monoCamLeftNode.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    monoCamRightNode.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    # If needed, other settings can be configured here...
    # e.g. for the colour camera, we can set the following:
    rgbCamNode.setPreviewSize(config.COLOR_CAMERA_PREVIEW_SIZE)
    rgbCamNode.setInterleaved(config.CAMERA_INTERLEAVED)
    rgbCamNode.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    
    # define neural network hyperparameters like confidence threshold,
    # number of inference threads. The NN will make predictions
    # based on the source frames
    nn.setConfidenceThreshold(config.NN_THRESHOLD)
    nn.setNumInferenceThreads(config.INFERENCE_THREADS)
    # set mobilenet detection model blob path
    nn.setBlobPath(config.MOBILENET_DETECTION_MODEL_PATH)
    nn.input.setBlocking(False)

    # Finally, link the camera nodes to the corresponding output nodes:
    monoCamLeftNode.out.link(xOutLeft.input)
    monoCamRightNode.out.link(xOutRight.input)
    # It looks like the RGB camera doesn't have the option 'OUT', instead 'PREVIEW' should be used:
    rgbCamNode.preview.link(xOutRGB.input)
    # camera frames linked to NN input node
    rgbCamNode.preview.link(nn.input)
    
    # NN out (image detections) linked to XLinkOut node
    nn.out.link(nnOut.input)

    # NN unparsed inference results  (metadata) linked to XLinkOut node
    nn.outNetwork.link(nnNetworkOut.input)

    return pipeline

def upload_pipeline(pipeline):
    # Once the pipeline is created, configured and linked
    # we upload it to the device:
    with dai.Device(pipeline=pipeline) as device:
        # Everything written here, will be performed in the device (i.e. firmware upload)
        # Just for educational purposes, write out some settings:
        print('MxID: ', device.getDeviceInfo().getMxId())
        print('USB Speed: ', device.getUsbSpeed())
        print('Connected Cameras: ', device.getConnectedCameras())
        # To retrieve everything from the device we set up output 
        # queues for each node. By reading the configuration parameters,
        # we make it flexible for the host specs:
        qLeftCam = device.getOutputQueue(
            name='Left', # This is the name set through setStreamName
            maxSize = config.COLOR_CAMERA_QUEUE_SIZE,
            blocking = config.QUEUE_BLOCKING,
        )
        qRightCam = device.getOutputQueue(
            name = 'Right',
            maxSize = config.COLOR_CAMERA_QUEUE_SIZE,
            blocking = config.QUEUE_BLOCKING,
        )
        qRGBCam = device.getOutputQueue(
            name = 'RGB',
            maxSize = config.COLOR_CAMERA_QUEUE_SIZE,
            blocking = config.QUEUE_BLOCKING,
        )
        # output queues are also used to get the rgb frames
        # and nn data from the outputs defined above
        qDet = device.getOutputQueue(
            name='nn',
            maxSize=config.COLOR_CAMERA_QUEUE_SIZE,
            blocking=config.QUEUE_BLOCKING,
        )
        qNN = device.getOutputQueue(
            name='nnNet',
            maxSize=config.COLOR_CAMERA_QUEUE_SIZE,
            blocking=config.QUEUE_BLOCKING,
        )

        # initialize frame, detections list, and startTime for
        # computing FPS
        clrFrame = None
        detections = []
        startTime = time.monotonic()
        counter = 0

        # color pattern for displaying FPS
        color2 = config.TEXT_COLOR2  
        # boolean variable for printing NN layer names on console
        printOutputLayersOnce = config.PRINT_NEURAL_NETWORK_METADATA

        # And finally the streaming block:
        while True:
            # We can use get (blocking the exec) or tryGet (non-blocking) 
            # to retrieve each frame.:
            LeftFrames = qLeftCam.tryGet()
            RightFrames = qRightCam.tryGet()
            RGBFrames = qRGBCam.tryGet()
            inDet = qDet.tryGet()
            inNN = qNN.tryGet()


            # Additionally, tryGet returns None if no data available from the queue:
            if LeftFrames is not None:
                lFrame = LeftFrames.getCvFrame()
                displayFrame('Left', lFrame, [])
                # cv2.imshow('Left', LeftFrames.getCvFrame())
            if RightFrames is not None:
                rFrame = RightFrames.getCvFrame()
                displayFrame('Right', rFrame, [])
                # cv2.imshow('Right', RightFrames.getCvFrame())

            if RGBFrames is not None:
                clrFrame = RGBFrames.getCvFrame()
                # cv2.imshow('RGB', clrFrame)
                # annotate the frame with FPS information
                cv2.putText(
                    clrFrame, 'NN fps: {:.2f}'.
                    format(counter / (time.monotonic() - startTime)),
                    (2, clrFrame.shape[0] - 4),
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
            
            #  If the frame from Colour camera is available, display it:
            if clrFrame is not None:
                displayFrame('RGB Detection', clrFrame, detections=detections)
            # break out from the while loop if `q` key is pressed
            if cv2.waitKey(1) == ord('q'):
                break

pipeline = setup_pipeline()
upload_pipeline(pipeline=pipeline)


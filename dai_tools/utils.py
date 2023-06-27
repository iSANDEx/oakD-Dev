# import the necessary packages
from dai_tools import config
import numpy as np
import cv2

# color pattern for annotating frame with object category, bounding box,
# detection confidence
color = config.TEXT_COLOR

# MobilenetSSD label list
labelMap = config.CLASS_LABELS

# nn data (bounding box locations) are in <0..1>
# range - they need to be normalized with frame width/height
def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


# displayFrame method iterates over the detections of a frame,
# denormalizes the bounding box coordinates and annotates the frame with
# class label, detection confidence, bounding box
def displayFrame(name, frame, detections):
    for detection in detections:
        bbox = frameNorm(
            frame, (
                detection.xmin, detection.ymin,
                detection.xmax, detection.ymax,
            ),
        )
        cv2.putText(
            frame, labelMap[detection.label], (
                bbox[0] + 10,
                bbox[1] + 20,
            ),
            cv2.FONT_HERSHEY_TRIPLEX, 0.5, color,
        )
        cv2.putText(
            frame, f'{int(detection.confidence * 100)}%',
            (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX,
            0.5, color,
        )
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
            color, 2)

    # show the frame
    cv2.imshow(name, frame)

# method that prints the detection network output layer name
def print_neural_network_layer_names(inNN):
    toPrint = 'Output layer names:'
    for ten in inNN.getAllLayerNames():
        toPrint = f'{toPrint} {ten},'
        print(toPrint)
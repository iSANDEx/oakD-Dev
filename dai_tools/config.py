# set the color camera preview size and interleaved
COLOR_CAMERA_PREVIEW_SIZE = 300, 300
CAMERA_INTERLEAVED = False
CAMERA_FPS = 40
# 
# queue parameters for rgb and mono camera frames at host side
COLOR_CAMERA_QUEUE_SIZE = 4
QUEUE_BLOCKING = False

# object detection class labels
CLASS_LABELS = ["background", "aeroplane", "bicycle", "bird", "boat",
                "bottle", "bus", "car", "cat", "chair", "cow",
                "diningtable", "dog", "horse", "motorbike", "person",
                "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
MOBILENET_DETECTION_MODEL_PATH = 'models/mobilenet-ssd_openvino_2021.' \
                                 '4_6shave.blob'

# neural network hyperparameters
NN_THRESHOLD = 0.5
INFERENCE_THREADS = 2
PRINT_NEURAL_NETWORK_METADATA = True

# frame text color pattern
TEXT_COLOR = (0, 255, 0)
TEXT_COLOR2 = (255, 255, 255)
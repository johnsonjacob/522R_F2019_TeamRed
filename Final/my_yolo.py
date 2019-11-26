# USAGE
# sudo MXNET_CUDNN_AUTOTUNE_DEFAULT=0 python3 yolo.py
# OPTIONAL PARAMETERS
# -c/--confidence (.0-1.0) (detected objects with a confidence higher than this will be used)

# import the necessary packages
# from car_control import steer, drive
from matplotlib import pyplot as plts
from gluoncv import model_zoo, utils
# import pyrealsense2 as rs
from PIL import Image
from signal import signal, SIGINT
from sys import exit
import numpy as np
import mxnet as mx
import argparse
import imutils
import serial
import time
import cv2
import os
import gc

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

"""Transforms for YOLO series."""


def transform_test(imgs, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    if isinstance(imgs, mx.nd.NDArray):
        imgs = [imgs]
    for im in imgs:
        assert isinstance(im, mx.nd.NDArray), "Expect NDArray, got {}".format(type(im))

    tensors = []
    origs = []
    for img in imgs:
        orig_img = img.asnumpy().astype('uint8')
        img = mx.nd.image.to_tensor(img)

        img = mx.nd.image.normalize(img, mean=mean, std=std)

        tensors.append(img.expand_dims(0))
        origs.append(orig_img)
    if len(tensors) == 1:
        return tensors[0], origs[0]
    return tensors, origs


def load_test(filenames, short=416):
    if not isinstance(filenames, list):
        filenames = [filenames]
    imgs = [letterbox_image(f, short) for f in filenames]
    return transform_test(imgs)


# this function is from yolo3.utils.letterbox_image
def letterbox_image(image, size=416):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size

    scale = min(size / iw, size / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', (size, size), (128, 128, 128))
    new_image.paste(image, ((size - nw) // 2, (size - nh) // 2))
    return mx.nd.array(np.array(new_image))


# Function to correctly exit program
def handler(signal_received, frame):
    vs.release()
    cv2.destroyAllWindows()
    print('CTRL-C detected. Exiting gracefully')
    exit(0)


# initialize the video stream, pointer to output video file, and
# frame dimensions
def run_yolo():
    vs = cv2.VideoCapture("/dev/video2", cv2.CAP_V4L)  # ls -ltr /dev/video*
    (W, H) = (None, None)

    # Implement YOLOv3MXNet
    net = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)

    # Set device to GPU
    device = mx.gpu()

    net.collect_params().reset_ctx(device)

    signal(SIGINT, handler)
    print('Running. Press CTRL-C to exit')

    current_bb = [0, 0, 0, 0]

    (grabbed, frame) = vs.read()

    if W is None or H is None:
        (H, W) = frame.shape[:2]

    if grabbed:
        annotated_img = sign_detect(frame, H, W, net, device)

    cv2.imshow('Annotated Image', annotated_img)
    vs.release()
    cv2.destroyAllWindows()


# loop over frames from the video file stream
def sign_detect(frame, H, W, net, device, current_bb):

    # read the next frame from the file
    # (grabbed, frame) = vs.read()
    #
    # # if the frame was not grabbed, then we have reached the end
    # # of the stream
    # if not grabbed:
    #     break

    # if the frame dimensions are empty, grab them
    # if W is None or H is None:
    #     (H, W) = frame.shape[:2]

    # from gluoncv import data
    yolo_image = Image.fromarray(frame, 'RGB')
    x, img = load_test(yolo_image, short=416)

    class_IDs, scores, bounding_boxs = net(x.copyto(device))

    # print(class_IDs)
    # print(scores)
    # print(bounding_boxs)

    # Convert to numpy arrays, then to lists
    class_IDs = class_IDs.asnumpy().tolist()
    scores = scores.asnumpy().tolist()
    bounding_boxs = bounding_boxs.asnumpy()

    # iterate through detected objects
    for i in range(len(class_IDs[0])):
        if ((scores[0][i])[0]) > args["confidence"]:
            current_class_id = net.classes[int((class_IDs[0][i])[0])]
            current_score = (scores[0][i])[0]
            current_bb = bounding_boxs[0][i - 1]

    gc.collect()

    # print("Class ID: ", current_class_id)
    # print("Score: ", current_score)
    # print("Bounding Box Coordinates: ", current_bb, "\n")

    red = (0, 0, 255)
    green = (0, 255, 0)
    top_left = (current_bb[0], current_bb[1])
    bottom_right = (current_bb[2], current_bb[3])
    cv2.circle(img, top_left, 5, red, -1)
    cv2.circle(img, bottom_right, 5, green, -1)

    return img

    # cv2.imshow("Camera Feed", img)
    # key = cv2.waitKey(1) & 0xFF

    # if key == ord("q"):
    #     break


run_yolo()

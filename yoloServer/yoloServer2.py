
#setup what I need for darknet
import threading
#import darknet as dn
import json
import cv2
import numpy as np
from gluoncv import model_zoo, utils
import mxnet as mx
from PIL import Image

#setup darknet
# dn.set_gpu(0)
# net = dn.load_net("cfg/yolov3-tiny.cfg".encode('utf-8'), "cfg/yolov3-tiny.weights".encode('utf-8'), 0)
# meta = dn.load_meta("cfg/coco.data".encode('utf-8'))
#setup mxnet
net = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)
device = mx.cpu()
net.collect_params().reset_ctx(device)

def check_green(objects):
    image = cv2.imread("temp.jpg")
    tl_ims = []
    print(image)
    for o in objects:
        if o[0] == 'traffic light':
            o = o[2]
            tl_ims.append(image[int(o[0]):int(o[2]),int(o[1]):int(o[3]),:])
    print(objects)
    cv2.imshow('temp', cv2.rectangle(image,(int(o[0]),int(o[1])),(int(o[2]),int(o[3])),(0,255,0),3))
    cv2.waitKey()

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


# Returns masked out images of the traffic lights. We just need to threshold the colors
# next.
def color_detect(image, bounding_box):
    # small_image = cv2.resize(image, (1008, 756))
    all_lights = []
    for i in range(len(bounding_box)):
        bb = bounding_box[i][2].astype("int")
        small_image = image[bb[1]:bb[3],bb[0]:bb[2]]
        threshold = 100

        gray = cv2.cvtColor(small_image, cv2.COLOR_RGB2GRAY)
        kernel = np.ones((9, 9), np.uint8)
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        ret, thresh = cv2.threshold(tophat, threshold, 255, cv2.THRESH_BINARY)

        #masked = np.zeros_like(small_image)
        #for i in range(3):
        #    masked[:, :, i] = np.multiply(thresh, small_image[:, :, i])
        masked = cv2.bitwise_and(small_image, small_image, mask = thresh)
        #print(small_image)
        #print(thresh)

        #masked = cv2.cvtColor(masked, cv2.COLOR_GRAY2RGB)
        all_lights.append(masked)

        # cv2.imshow("masked", image)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        #
        # cv2.imshow("masked", masked)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        lower_third = masked[int(2*masked.shape[0]/3):]
        upper_third = masked[:int(masked.shape[0]/3)]
        middle_third = masked[int(masked.shape[0]/3):int(2*masked.shape[0]/3)]
        # cv2.imshow("masked", lower_third)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        # cv2.imshow("masked", upper_third)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        def get_count(third):
            count = 0
            for h in third:
                for w in h:
                    if sum(w) > 100:
                        count += 1
            #print(count)
            return count

        upper = get_count(upper_third)
        middle = get_count(middle_third)
        lower = get_count(lower_third)

        print (upper, middle, lower)

        if upper > lower + lower / 10:
            return "red"
        elif upper < lower + lower / 10 and upper > lower - lower / 10:
            return "no light"
        else:
            return "green"
        #return (upper,middle,lower)
    return "no light"

def check_image_mx(image):
    global device
    #print(image)
    #yolo_image = Image.frombytes("RGB", (1280,720), image)
    yolo_image = Image.open(io.BytesIO(image))
    # yolo_image = image
    x, img = load_test(yolo_image, short=416)
    class_IDs, scores, bounding_boxs = net(x.copyto(device))
    detections = []
    class_IDs = class_IDs.asnumpy()
    scores = scores.asnumpy()
    #print(scores)
    bounding_boxs = bounding_boxs.asnumpy()
    # print(class_IDs.shape)
    class_IDs = class_IDs.astype(int)
    #print(net.classes)
    for i in range(len(class_IDs[0])):
        # print(class_IDs[i])
        if "traffic light" == net.classes[class_IDs[0,i,0]]:
           print(scores[0,i,0])
           if scores[0,i,0] > .3:
               detections.append(["traffic light",scores[0,i,0],bounding_boxs[0,i,0:4]])
               current_bb = bounding_boxs[0,i,0:4]
               cv2.rectangle(img, (current_bb[0], current_bb[1]), (current_bb[2], current_bb[3]), (0,0,0), 10)
               break
    #print(class_IDs, scores, bounding_boxs)

    return detections, img


def check_image(image):
    r = dn.detect(net, meta, image)
    r_out = []
    for obj in r:
        print(obj[0], end = '  \t')
        print(obj[1], end = '  \t')
        print(obj[2])
        coord = [0,0,0,0]
        coord[0] = int(obj[2][0] - obj[2][2]/2)
        coord[1] = int(obj[2][1] - obj[2][3]/2)
        coord[2] = int(obj[2][0] + obj[2][2]/2)
        coord[3] = int(obj[2][1] + obj[2][3]/2)
        obj = (str(obj[0], 'utf-8'), obj[1], coord)
        r_out.append(obj)
    #print(json.dumps(r_out))
    check_green(r_out)
    return json.dumps(r_out)


# This shows how I've gotten it working, but  I haven't worked on setting up the
# server because it was easier to debug this way.

# for color in ["red", "green", "yellow"]:
#     img = Image.open(color+'.JPG')
#     bb, small_img = check_image_mx(img)
#     small_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
#     cv2.imshow("Annotated", small_img)
#     cv2.waitKey()
#     cv2.destroyAllWindows()
#     color_detect(small_img, bb)


# setup the server
from flask import Flask
from flask import request
import io
from PIL import Image


app = Flask(__name__)

@app.route("/", methods = ['POST'])
def look_at_image():
    print("recieve")
    data = request.get_data()
    bb, small_img = check_image_mx(data)
    small_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
    output = color_detect(small_img, bb)
    print("processed")
    return json.dumps(output)
    #with open("temp.jpg", 'wb') as f:
    #    f.write(data)
    #return check_image('temp.jpg')

if __name__ == "__main__":
    #t1 = threading.Thread(target=interface.start_api_server, name = "t1")
    #t2 = threading.Thread(target=app.run(debug = True), name = "t2")

    #t1.start()
    #t2.start()
    #init_yolo()

    app.run(host='0.0.0.0')
    #check_image("data/stoplight.png")

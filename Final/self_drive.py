from intersections import Intersections
from image_process import Image_Process
from steering_pid import SteeringPid
from car_control import Car_Control
from waypoint import Waypoint
#from server import Server
import signal
from PIL import Image
import cv2
import sys
import gc
import numpy as np
from flask import Flask, request, Response, render_template
import threading
#import globals
import pyrealsense2 as rs
#from gluoncv import model_zoo, utils
#import mxnet as mx
from matplotlib import pyplot as plts
import requests
import json
import time


#import darknet as dn

global run_yolo_bool
global annotated_boxes
global stop_at_light

global frame
global image
global api_cmd
global run
global dst
global white
global yellow
global feed
global api_speed
global depth_colormap


class StopLightDetector:
    # construct the argument parse and parse the arguments
    #ap = argparse.ArgumentParser()
    #ap.add_argument("-c", "--confidence", type=float, default=0.5,
    #                help="minimum probability to filter weak detections")
    #args = vars(ap.parse_args())

    """Transforms for YOLO series."""
    def __init__(self):
        
        #dn.set_gpu(0)
        #self.net = dn.load_net("cfg/yolov3-tiny.cfg".encode('utf-8'), "cfg/yolov3-tiny.weights".encode('utf-8'), 0)
        #self.meta = dn.load_meta("cfg/coco.data".encode('utf-8'))
        self.current_frame = None

  

    def setup_frame(self):
        global frame
        frame1 = frame[:int(frame.shape[0]/2)]
        #yolo_image = Image.fromarray(frame1, 'RGB')

        #x, img = self.load_test(yolo_image, short=416)

        #class_IDs, scores, bounding_boxs = net(x.copyto(device))
        return frame1

    # initialize the video stream, pointer to output video file, and
    # frame dimensions
    def run_yolo(self):
        global frame
        global run_yolo_bool
        global annotated_boxes
        global stop_at_light

        addr = 'http://192.168.1.28:5000/'
        #addr = 'http://192.168.1.26:5000/'
        # prepare headers for http request
        content_type = 'image/jpeg'
        headers = {'content-type': content_type}
        
        while True:

            #current_bb = [0, 0, 0, 0]

            try:
                #print("while")
                run_yolo_bool = False
                frame1 = self.setup_frame()
                #print(frame1.shape)
                #cv2.imwrite('yolo_temp.jpg', frame1)
                #detections = self.get_labels("yolo_temp.jpg")
                _, img_encoded = cv2.imencode('.jpg', frame1)
                # send http request with image and receive response

                response = requests.post(addr, data=img_encoded.tostring(), headers=headers)
                print(json.loads(response.text))
                detections = json.loads(response.text)
                self.stop_light_analyze(detections)
                if detections == "red":
                    stop_at_light = True
                    print("Stopping at light")
                elif stop_at_light and detections == "no light" and count < 15:
                    count = count + 1
                    stop_at_light = True
                    print("Lost Light")
                else:
                    count = 0
                    stop_at_light = False
                #time.sleep(.25)
            except:
                print("error")


    def is_green(self, img, light):
        small = img[int(light[1]):int(light[3]),int(light[0]):int(light[2])]
        #print(small.shape)
        return np.mean(small[int(small.shape[0]/3):,:,1]) > 60

    def stop_light_analyze(self, detections):
        global stop_at_light
        global frame

        img = frame
        lights = []
        for d in detections:
            if d[0] == "traffic light":
                lights.append(d)
        max_conf = 0
        max_index = 0
        for i in range(len(lights)):
           if lights[i][1] > max_conf:
               max_conf = lights[i][1]
               max_index = i

        one_green = False
        if len(lights) == 0:
            one_green = True
        else:        
            lights = [lights[max_index]]

            one_green = False
            for light in lights:
                if self.is_green(img,light[2]):
                    one_green = True
                    break
        stop_at_light = not one_green




class Server:

    api_server = Flask(__name__) #API Server

    def send_frame():
        global image
        global frame
        global white
        global yellow
        global feed
        global depth_colormap
        global annotated_boxes

        while feed < 6:
            #if frame == None:
            #    grabbed, frame = self.vs.read()
            if feed == 0:
            	cv2.imwrite('temp.jpg', frame)
            elif feed == 1:
            	cv2.imwrite('temp.jpg', white)
            elif feed == 2:
            	cv2.imwrite('temp.jpg', yellow)
            elif feed == 3:
            	cv2.imwrite('temp.jpg', image)
            elif feed == 4:
            	cv2.imwrite('temp.jpg', depth_colormap)
            elif feed == 5:
                cv2.imwrite('temp.jpg', annotated_boxes)
                
            
            frame2 = open('temp.jpg','rb').read()

            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n')


    @api_server.route('/video_feed')
    def video_feed():
        """Video streaming route. Put this in the src attribute of an img tag."""
        return Response(Server.send_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @api_server.route('/', methods=['GET'])
    def index():
        return render_template('index.html')

    @api_server.route('/api', methods=['POST'])
    def receive_cmd(self):
        global api_cmd
        r = request
        api_cmd = r.data

        print(api_cmd)
        #TODO: Implement parser and integrate with main

        response = {'message': 'command received'}
        response_pickled = jsonpickle.encode(response)
        return Response(response=response_pickled, status=200, mimetype="application/json")

    @api_server.route('/global_cmd', methods=['POST'])
    def global_cmd():
        global run
        global stop_at_light

        start = request.form['run_cmd']

        if start == "Start":
            run = 1
            stop_at_light = False
            print("Starting")
        elif start == "Stop":
            run = 0
            print("Stopping")
        return render_template('index.html')

    @api_server.route('/feed_select', methods=['POST'])
    def feed_select():
        global feed

        feed_sel = request.form['feed_cmd']

        if feed_sel == "Normal":
            feed = 0
            print("Selecting Feed: Normal")
        elif feed_sel == "White":
            feed = 1
            print("Sele3cting Feed: White")
        elif feed_sel == "Yellow":
            feed = 2
            print("Sele3cting Feed: Yellow")
        elif feed_sel == "Transform":
            feed = 3
            print("Selecting Feed: Transform")
        elif feed_sel == "Depth":
            feed = 4
            print("Selecting Feed: Depth")
        elif feed_sel == "Annotated":
            feed = 5
            print("Selecting Feed: Annotated")
        elif feed_sel == "Off":
            feed = 6
        
        return render_template('index.html')

    @api_server.route('/destination', methods=['POST'])
    def destination():
        global dst

        dst_cmd = request.form['dst_cmd']

        if dst_cmd == "Set Destination":
            dst_text = request.form['coordinates'].split(",")
            dst = (float(dst_text[0]),float(dst_text[1]))
            print("Goint to Destination: {}".format(dst))
        elif dst_cmd == "Clear Destination":
            dst = (None,None)
            print("Clearing Destination")
            print(dst)
       
        
        return render_template('index.html')

    @api_server.route('/speed', methods=['POST'])
    def speed():
        global api_speed
        speed_cmd = request.form['speed_cmd']
        if speed_cmd == "Set Speed":
            speed_text = request.form['speed']
            api_speed = float(speed_text)
            print("Setting Speed: {}".format(api_speed))
   
       
        
        return render_template('index.html')


    def start_api_server(self):
        self.api_server.run(host="0.0.0.0", port=5000)


class Self_Drive:

    def __init__(self, speed=0.15, lane_offset=150, wait_period=10):
        self.speed = speed
        self.pid = SteeringPid(lane_offset, kp=0.1, ki=0.006, kd=1.2)
        self.waypoint = Waypoint()
        self.car = Car_Control()
        self.detector = Image_Process()
        self.vs = cv2.VideoCapture("/dev/video2", cv2.CAP_V4L) # TODO: figure out a good way to do this
        self.intersections = Intersections(wait_period=wait_period)
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(self.config)



    def go_to_point(self, point):
        
   
        self.car.steer(0.0)
        self.car.drive(self.speed)
        self.waypoint.set_point(point)

        while not self.waypoint.arrived():
            try:
                grabbed, frame = self.vs.read()
                speed = self.speed

                if not grabbed:
                    print("Couldn't Grab Next Frame")
                    break

                offset, d2c, image = self.detector.offset_detect(frame)
                angle = self.pid.run_pid(offset)

                # TODO: integrate in intersection detection. It will have changed.
                # at_intersection = ((d2c != None) and (d2c < 15))
                # if at_intersection:
                #     speed = speed / 2
                #     at_intersection = ((d2c != None) and (d2c < 10))
                #     if at_intersection:
                #         speed = speed / 2
                #         at_intersection = ((d2c != None) and (d2c < 3))
                #         if at_intersection:
                #             speed = 0
                #             print('at intersection')
                #             self._handle_intersection()

                self._handle_intersection()

                self.car.drive(speed)
                self.car.steer(angle)

            except Exception as e:
                print(repr(e))

        print("Arrived at {}".format(point))
        self.car.stop()


    def self_drive(self):
        global run
        global image
        global white
        global yellow
        global frame
        global dst
        global api_speed
        global depth_colormap
        global run_yolo_bool
        global stop_at_light

        self.car.steer(0.0)
        if run:
            self.car.drive(self.speed)

        yolo_run_count = 0

        while True:
            try:
                # Wait for a coherent pair of frames: depth and color
                frames = self.pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()

                grabbed, frame = self.vs.read()
                speed = self.speed
                if not grabbed or not depth_frame:
                    print("Couldn't Grab Next Frame")
                    break
                
                yolo_run_count += 1
                if yolo_run_count == 30:
                    run_yolo_bool = True
                    yolo_run_count = 0
                
                depth_image = np.asanyarray(depth_frame.get_data())
                #print(depth_image[220][200:400])
                
                num = 0
                for i in range(200,400):
                    if depth_image[220][i] < 600 and depth_image[220][i] > 0:
                        num += 1
                if num >= 20:
                    obstacle = True
                else:
                    obstacle = False

                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_HSV)
    
                offset, image, white, yellow  = self.detector.offset_detect(frame)

                angle = self.pid.run_pid(offset)
                #print(run, obstacle, stop_at_light, dst, api_speed)
                if run and not obstacle and not stop_at_light:
                    if dst != (None,None):
                        self.waypoint.set_point(dst)
                        if self.waypoint.arrived():
                            dst = (0,0)
                            run = 0

                    self._handle_intersection()
                    self.car.drive(api_speed)
                    self.car.steer(angle)
                else:
                    self.car.drive(0)

            except Exception as e:
                print(repr(e))


        self.car.stop()


    def _handle_intersection(self):
        is_intersection, action, coor = self.intersections.get_intersection()

        if not is_intersection:
            return
            raise Exception("Intersections: {} {}: Is not at an intersection!".format(coor, action))

        print("At {} Intersection!".format(action))
        
        which_turn = action

        if action == "FOUR_WAY":
            action = self.waypoint.get_turn()

        print("Driving {}".format(action))

        if action == "STRAIGHT":
            self.car.straight(wait=0+.75,duration=1.5, angle=5, speed=(self.speed + 0.3))
        elif action == "LEFT":
            self.car.steer(0)
            if which_turn != "FOUR_WAY":
                self.car.stop()
                time.sleep(1)
            if which_turn == "FOUR_WAY":
                #self.car.straight(wait=.25, duration=.25, angle=-10, speed=(self.speed + 0.4))
                self.car.left(wait=0+1, duration=1.8, angle=-20, speed=(self.speed+ 0.3))
            else:
                self.car.left(wait=0.1+1, duration=2, angle=-20, speed=(self.speed+ 0.3))
        elif action == "RIGHT":
            self.car.steer(0)
            if which_turn != "FOUR_WAY":
                self.car.stop()
                time.sleep(1)
            if which_turn == "FOUR_WAY":
                print("here")
                #self.car.straight(wait=.2, duration=0.25, angle=20, speed=(self.speed + 0.4))
                self.car.right(wait=0.0+.9, duration=2.5, angle=30, speed=(self.speed + 0.3))
            else:
                self.car.right(wait=0.0+.1, duration=1.6, angle=30, speed=(self.speed + 0.3))

        print("Done Driving {}".format(action))





if __name__ == "__main__":
    global run
    global feed
    global api_speed
    global run_yolo_bool
    global stop_at_light
    stop_at_light = False
    run_yolo_bool = False
    car = Self_Drive()
    interface = Server()
    stop_light = StopLightDetector()
    run = 0
    feed = 0
    api_speed = 0.1
    dst = (0,0)

    t1 = threading.Thread(target=interface.start_api_server, name = "t1")
    t2 = threading.Thread(target=stop_light.run_yolo, name = "t2")
    t3 = threading.Thread(target=car.self_drive, name = "t3")
    
    t3.start()
    time.sleep(5)
    t1.start()
    t2.start()
    


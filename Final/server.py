import cv2
import sys
import numpy as np
from flask import Flask, request, Response, render_template
import threading
import globals


class Server:

    api_server = Flask(__name__) #API Server

    def send_frame():
        
        while True:
            #if frame == None:
            #    grabbed, frame = self.vs.read()
            cv2.imwrite('temp.jpg', globals.image) #can I just yield frame without saving it first?
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
        
        r = request
        globals.api_cmd = r.data

        print(api_cmd)
        #TODO: Implement parser and integrate with main

        response = {'message': 'command received'}
        response_pickled = jsonpickle.encode(response)
        return Response(response=response_pickled, status=200, mimetype="application/json")

    @api_server.route('/', methods=['POST'])
    def form_input():
        

        start = request.form['run']
        

        if start == "Start":
            globals.run = 1
            print("Starting")
        elif start == "Stop":
            run = 0
            print("Stopping")
        return render_template('index.html')


    def start_api_server(self):
        self.api_server.run(host="0.0.0.0", port=5000)


if __name__ == "__main__":
    global frame
    interface = Server()
    vs = cv2.VideoCapture("/dev/video2", cv2.CAP_V4L)

    t1 = threading.Thread(target=interface.start_api_server, name = "t1")

    t1.start()
    
    run = 0

    while True:
        grabbed, image = vs.read()
        #print (run)

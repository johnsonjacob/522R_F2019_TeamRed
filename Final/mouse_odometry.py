#!/usr/bin/python
from multiprocessing import Value, Process
import sys
import usb.core
import usb.util
class mouse_odometry:

    def _pass(self):
        pass

    def __init__(self):
        self.distance_done = Value('b', False)
        self.distance = Value('d', 0)
        self.distance_start = Value('b', False)
        self.current_x = Value('d', 0)
        self.current_y = Value('d',  0)

        #self.line_tilt = Value('d', 0)
        #self.new_line = Value('b', False)
        #self.get_line = Value('b', False)
        process = Process(target = self._run_odometry, args=(self.distance_done, self.distance, self.distance_start, self.current_x, self.current_y))
        process.start()
        #process2 = Process(target = self.find_white_lines, args=(self.angle_offset, self.found_line, self.look_for_line))
        #process2.start()
        process2 = Process(target=run_flask)#,args=[self.line_tilt, self.new_line, self.get_line])
        process2.start()
            
            

    def _run_odometry(self, distance_done, distance, distance_start, current_x, current_y):
        # decimal vendor and product values
        #dev = usb.core.find(idVendor=1118, idProduct=1917)
        # or, uncomment the next line to search instead by the hexidecimal equivalent
        dev = usb.core.find(idVendor=0x0461, idProduct=0x4e22)
        # first endpoint
        interface = 0
        endpoint = dev[0][(0,0)][0]
        # if the OS kernel already claimed the device, which is most likely true
        # thanks to http://stackoverflow.com/questions/8218683/pyusb-cannot-set-configuration
        if dev.is_kernel_driver_active(interface) is True:
            # tell the kernel to detach
            dev.detach_kernel_driver(interface)
            # claim the device
            usb.util.claim_interface(dev, interface)
        
        x_distance = 0
        y_distance = 0
        
        count = 0
        data_str = ''
        while True:#collected < attempts :
            try:
                data = dev.read(endpoint.bEndpointAddress,endpoint.wMaxPacketSize)
                data_str = (str(data[1]).rjust(4) + ' ' + str(data[1]).rjust(4) + ' ' + str(data[2]).rjust(4) + ' ' + str(data[3]).rjust(4))
                
                if distance_start.value:
                    x_distance = 0
                    y_distance = 0
                    distance_start.value = False
                    distance_done.value = False
                    print("\nmouse_run_start")

                if not distance_done.value and y_distance >= distance.value:
                    distance_done.value = True
                    print("\nMouse run end")

                #if data[3] != 0:
                data3str = str(data[3]).rjust(4)
                    #print("\nData 3: " + str(data[3]))
                #if data[0] != 0:
                data0str = str(data[0]).rjust(4)
                    #print("\nData 0:" + str(data[0]))

                x_distance -= data[1] - 256 -10 if data[1] > 127 else data[1] +10
                y_distance -= data[2] - 256 -10 if data[2] > 127 else data[2] +10

                #data_str = str(x_distance).rjust(4), str(y_distance).rjust(4)

                data_str = str(y_distance).rjust(8) + str(count).rjust(8)
                #data_str = data3str + " " + data0str
                count += 1
                
                #print(data_str, end='')
                #print('\b' * len(data_str), end='', flush = True)
            except usb.core.USBError as e:
                data = None
                if e.args == ('Operation timed out',):
                    continue
        # release the device
        usb.util.release_interface(dev, interface)
        # reattach the device to the OS kernel
        dev.attach_kernel_driver(interface)

    def start_run(self, distance):
        self.distance.value = distance
        self.distance_done.value = False
        self.distance_start.value = True

    def in_run(self):
        return not self.distance_done.value


from flask import Flask
from flask import request
import io
from PIL import Image

app = Flask(__name__)

Line_tilt = Value('d',0)
New_line = Value('b', False)
Get_line = Value('b',False)
def run_flask():
    # setup the server
    app.run(host='0.0.0.0',port=5001)

@app.route("/line_tilt/<tilt>", methods = ['GET'])
def web_get(tilt):
    global Get_line
    global Line_tilt
    global New_line
    #generally I think I want to get the tilt closest to the intersection
    if Get_line.value:
        Line_tilt.value = int(tilt)
        New_line.value = True
        Get_line.value = False
    return "accepted"

def phone_tilt_avaliable():
    global Get_line
    global Line_tilt
    global New_line
    if New_line.value:
        New_line.value = False
        return True
    else:
        return False

def phone_get_tilt():
    global Get_line
    global Line_tilt
    global New_line
    return Line_tilt.value

def phone_get_next():
    global Get_line
    global Line_tilt
    global New_line
    New_line.value = False
    Get_line.value = True

if __name__ == "__main__":
    pass
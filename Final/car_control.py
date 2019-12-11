import serial
import numpy
from time import sleep
from mouse_odometry import mouse_odometry
from mouse_odometry import phone_tilt_avaliable
from mouse_odometry import phone_get_next
from mouse_odometry import phone_get_tilt
import time


class Car_Control:

    start_cmd = "!start{cmd}\n"
    init_cmd = "!inits{cmd:.2f}\n" #init vs inits?
    straight_cmd = "!straight{cmd}\n"
    pid_cmd = "!pid{cmd}\n"
    speed_cmd = "!speed{cmd:.1f}\n"
    steering_cmd = "!steering{cmd:.1f}\n"
    kp_cmd = "!kp{cmd:.2f}\n"
    ki_cmd = "!ki{cmd:.2f}\n"
    kd_cmd = "!kd{cmd:.2f}\n"


    def __init__(self, start_value=1595, init_value=0.75, \
                straight_value=1540, pid_on=True, \
                usb_device="/dev/ttyUSB0", baud_rate=115200): #start_value was 1600
        self.ser = serial.Serial(usb_device, baud_rate)
        self.ser.flushInput()

        sleep(2) # do I need this?
        
        self.current_speed = 0.0
        self.current_angle = 0.0
        self._send_command(Car_Control.start_cmd.format(cmd=start_value))
        self._send_command(Car_Control.init_cmd.format(cmd=init_value))
        self._send_command(Car_Control.straight_cmd.format(cmd=straight_value))
        self._send_command(Car_Control.pid_cmd.format(cmd=int(pid_on)))
        self.steer(0.0)
        self.drive(0)
        self.mo = mouse_odometry()

    def __del__(self):
        self.stop()

    def _send_command(self, command):
        self.ser.write(command.encode())

    def _drive_duration(self, wait=0, duration=0, angle=None, speed=None,stop=None):
        if angle is None:
            angle = self.current_angle
        if speed is None:
            speed = self.current_speed
        if stop != None:
            self.steer(0)
            self.stop()
            time.sleep(stop)

        self.steer(0)
        self.drive(speed)
        sleep(wait)
        self.steer(angle)
        self.drive(speed)
        sleep(duration)
        self.steer(0)
        # self.stop() # TODO: check with team about whether or not we want this functionality

    def _mouse_drive_duration(self,wait=0,duration=0, angle=None, speed =None, stop=None):
        phone_get_next()
        tilt = None
        if angle is None:
            angle = self.current_angle
        if speed is None:
            speed = self.current_speed
        if stop != None:
            self.steer(0)
            self.stop()
            start = time.time()
            while time.time() < start + stop:
                if phone_tilt_avaliable():
                    self.mo.start_run(wait)
                    tilt = phone_get_tilt()/3

        if not self.mo.in_run():
            self.mo.start_run(wait)
        
        if tilt != None:
            print("Correcting for tilt " + str(tilt))
            self.steer(-1 * tilt)
            time.sleep(0.1)

        self.steer(0)
        self.drive(speed)
        while self.mo.in_run():
            if stop == None:
                if phone_tilt_avaliable():
                    tilt = phone_get_tilt()/3
                    print("Correcting for tilt " + str(tilt))
                    self.steer(-1 * tilt)
                    time.sleep(0.1)
            sleep(0.001)
        print("wait done")
        self.steer(angle)
        self.mo.start_run(duration)
        while self.mo.in_run():
            sleep(0.001)
        print("turn done")
        #self.steer(0)


    def drive(self, value):
        #return
        if not numpy.isclose(value, 0): # TODO: check with team about whether or not we want this behavior for 0 speeds
            self.current_speed = value
        self._send_command(Car_Control.speed_cmd.format(cmd=value))

    def steer(self, value):
        #return
        if abs(value) > 30:
            value = 30*(numpy.sign(value))
        self.current_angle = value
        self._send_command(Car_Control.steering_cmd.format(cmd=value))

    def stop(self):
        self.drive(0.0)
        self.steer(0.0)

    def left(self, wait=0, duration=2.75, angle=-20, speed=None, stop=None): #1.75
        self._drive_duration(wait=wait, duration=duration, angle=angle, speed=speed, stop=stop)

    def mouse_left(self, wait=9000, duration=14000, angle = -30, speed =None, stop=None):
        self._mouse_drive_duration(wait=wait, duration=duration, angle = angle, speed=speed, stop=stop)
    
    def right(self, wait=0, duration=1, angle=25, speed=None, stop = None): #.75
        self._drive_duration(wait=wait, duration=duration, angle=angle, speed=speed, stop=stop)

    def mouse_right(self, wait=5000, duration=14000, angle = 300, speed =None, stop=None):
        self._mouse_drive_duration(wait=wait, duration=duration, angle = angle, speed=speed, stop = stop)

    def straight(self, wait=0, duration=4, angle=0, speed=None, stop = None):
        self._drive_duration(wait=wait, duration=duration, angle=angle, speed=speed,stop=stop)

    def mouse_straight(self, wait = 0, duration = 32000, angle = 0, speed = None, stop = None):
        self._mouse_drive_duration(wait=wait, duration=duration, angle = angle, speed=speed,stop=stop)

if __name__ == "__main__":
    car = Car_Control()

    car._drive_duration(duration=1, angle=22, speed=0.4)
    car.stop()
    sleep(1)

    car._drive_duration(duration=2.3, angle=-15, speed=0.4)
    car.stop()
    sleep(1)

    car._drive_duration(duration=.5, angle=0, speed=0.4)
    car.stop()
    sleep(1)

    car._drive_duration(duration=5, angle=-30, speed=0.4)
    car.stop()
    sleep(1)
 
    car.drive(0.3)
    car.left()
    car.stop()
    sleep(1)

    car.right()
    car.stop()
    sleep(1)

    car.straight()
    car.stop()
    sleep(1)


    for _ in range(2):
        for i in range(-30, 30):
            car.steer(i)
            sleep(.03)

        for i in range(30, -30, -1):
            car.steer(i)
            sleep(.03)

    car.stop()

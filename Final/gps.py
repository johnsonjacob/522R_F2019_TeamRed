from multiprocessing import Process, Value, Lock
from time import sleep
import requests
import sys

class GPS:

    _URL = "http://192.168.1.8:8080/{color}"
    _COLORS = {}

    @staticmethod
    def _check_color(color):
        if color not in GPS._COLORS:
            url = GPS._URL.format(color=color)
            latitude = Value("d", 0.0)
            longitude = Value("d", 0.0)
            old_latitude = Value("d", 0.0)
            old_longitude = Value("d", 0.0)
            mutex = Lock()
            process = Process(target=GPS._update_gps, args=(url, mutex, latitude, longitude, old_latitude, old_longitude))
            GPS._COLORS[color] = {
                "url" : url,
                "lat" : latitude,
                "long" : longitude,
                "old_lat" : old_latitude,
                "old_long" : old_longitude,
                "process" : process,
                "mutex" : mutex,
            }
            GPS._COLORS[color]["process"].start()
                

    @staticmethod
    def get_gps(color="Red"):

        GPS._check_color(color)
        with GPS._COLORS[color]["mutex"]:
            result = (GPS._COLORS[color]["lat"].value, GPS._COLORS[color]["long"].value)
        
        return result


    @staticmethod
    def get_gps_all(color="Red"):

        GPS._check_color(color)
        with GPS._COLORS[color]["mutex"]:
            result = (GPS._COLORS[color]["lat"].value, GPS._COLORS[color]["long"].value), (GPS._COLORS[color]["old_lat"].value, GPS._COLORS[color]["old_long"].value)
        
        return result
        

    @staticmethod
    def _update_gps(url, mutex, latitude, longitude, old_latitude, old_longitude):
        while True:
            sleep(.05)
            try:
                with mutex:

                    r = requests.get(url = url)
                    coorString = r.text
                    coordinates = coorString.split()

                    lat_tmp = latitude.value
                    long_tmp = longitude.value

                    latitude.value = float(coordinates[0])
                    longitude.value = float(coordinates[1])

                    if latitude.value != lat_tmp:
                        old_latitude.value = lat_tmp
                    
                    if longitude.value != long_tmp:
                        old_longitude.value = long_tmp

                    # # For Testing
                    # latitude.value = latitude.value + 10
                    # longitude.value = longitude.value + 10


            except Exception as e:
                print(repr(e))

if __name__ == '__main__':

    while True:
        for input in sys.argv[1:]:
            gps, old_gps = GPS.get_gps_all(input)
            print("{} : {}, OLD: {}".format(input, gps, old_gps))

            gps = GPS.get_gps(input)
            print("{} : {}".format(input, gps))

        sleep(1)

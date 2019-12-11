from gps import GPS 
from circular_value import Circular_Value
import time

class Waypoint:

    IMAGE_HALF_HEIGHT = 741
    IMAGE_HALF_WIDTH = 474
    #TURN_ORDER = ["LEFT", "RIGHT", "STRAIGHT", "STRAIGHT"]
    TURN_ORDER = ["RIGHT", "LEFT"]

    def __init__(self, box_width=20, box_height=20):
        self.box_width = box_width
        self.box_height = box_height
        self.cv = Circular_Value()
        self.destination = None
        self.destination_quad = None 
        self.index = len(self.TURN_ORDER) - 1


    def arrived(self):
        if self.destination is None:
            raise Exception("Destination Point not set")

        if self._near_destination():
            return True
        
        return False 

    def get_turn(self):
        if self.index >= len(self.TURN_ORDER)-1:
            self.index = 0
        else:
            self.index = self.index + 1
        
        return self.TURN_ORDER[self.index]


    def get_turn_old(self):
        if self.destination is None:
            raise Exception("Destination Point not set") 
        
        current_quad = self._get_quadrant(GPS.get_gps())

        if self.cv.next(current_quad) == self.destination_quad:
            return "LEFT"
        elif self.cv.previous(current_quad) == self.destination_quad:
            return "RIGHT"
        else:
            return "STRAIGHT"

    def _get_quadrant(self, point):
        x = point[0]
        y = point [1]

        if x < Waypoint.IMAGE_HALF_WIDTH:
            if y < Waypoint.IMAGE_HALF_HEIGHT:
                return 0
            else:
                return 3
        else:
            if y < Waypoint.IMAGE_HALF_HEIGHT:
                return 1
            else:
                return 2

    def _near_destination(self):
        current_coor = GPS.get_gps()
        x = current_coor[0]
        y = current_coor[1]

        if x <= 0.0 or y <= 0.0:
            return False

        x_dif = abs(self.destination[0] - x)
        y_dif = abs(self.destination[1] - y)
        if x_dif <  self.box_width and y_dif < self.box_height:
            return True
        else:
            return False

    def set_point(self, point):
        self.destination = point
        self.destination_quad = self._get_quadrant(point)

if __name__ == "__main__":
    import sys 
    from time import sleep
    waypoint = Waypoint() 

    # test_points = [
    #     (0.0, 0.0), 
    #     (840.0, 292.0), 
    #     (844.0, 1520.0), 
    #     (273.0, 941.0), 
    # ]

    # for point in test_points:
    #     waypoint.set_point(point)
    #     current_point = GPS.get_gps()
    #     current_quad = waypoint._get_quadrant(current_point)

    #     print("Dest Point: {} Dest Point Quad: {} Current Point: {} Current Quad: {} Turn: {}, Arrived: {}" \
    #           .format(point, waypoint.destination_quad, \
    #           current_point, current_quad, waypoint.get_turn(), \
    #           waypoint.arrived()))

    if len(sys.argv) < 3:
        raise Exception("Missing Destination Point")

    point = (float(sys.argv[1]), float(sys.argv[2]))

    waypoint.set_point(point)
    while True:
        current_point = GPS.get_gps()
        current_quad = waypoint._get_quadrant(current_point)

        print("Dest Point: {} Dest Point Quad: {} Current Point: {} Current Quad: {} Turn: {}, Arrived: {}" \
              .format(point, waypoint.destination_quad, \
              current_point, current_quad, waypoint.get_turn(), \
              waypoint.arrived()))
        sleep(1)

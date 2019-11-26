from gps import GPS 
import time

class Intersections:

    # INTERSECTIONS = (
    #     ([(360,50),(320,0)], "LEFT"),       # intersection 0 origin coordinates
    #     ([(250, 220),(200,180)], "RIGHT"),  # intersection 1 origin coordinates
    #     ([(193, 500)], "FOUR_WAY"),         # intersection 2 origin coordinates
    #     ([(411, 483)], "FOUR_WAY"),         # intersection 3 origin coordinates
    #     ([(200, 709)], "FOUR_WAY"),         # intersection 4 origin coordinates
    #     ([(413, 701)], "FOUR_WAY"),         # intersection 5 origin coordinates
    #     ([(870, 1370),(820,1320)], "RIGHT"),# intersection 6 origin coordinates
    #     ([(730, 1560),(680,1510)], "LEFT")  # intersection 7 origin coordinates
    # )

    INTERSECTIONS = (
        ((350, 27), "LEFT"),   # intersection 0 origin coordinates
        ((227, 202), "RIGHT"),   # intersection 1 origin coordinates
        ((326, 654), "FOUR_WAY"),  # intersection 2 origin coordinates
        ((570, 597), "FOUR_WAY"),  # intersection 3 origin coordinates
        ((367, 902), "FOUR_WAY"),  # intersection 4 origin coordinates
        ((622, 856), "FOUR_WAY"),  # intersection 5 origin coordinates
        ((829, 1367), "RIGHT"), # intersection 6 origin coordinates
        ((724, 1534), "LEFT")  # intersection 7 origin coordinates
    )


    def __init__(self, color="Red", wait_period=10, intersections=None, box_width=75, box_height=75):
        self.color = color
        self.box_height = box_height
        self.box_width = box_width
        self.time_stamp = time.time() - 3
        self.wait_period = wait_period
        if intersections is None:
            self.intersections = Intersections.INTERSECTIONS
        else:
            self.intersections = intersections

    def _within_box(self, origin, coor):

        x_origin = origin[0]
        y_origin = origin[1]
        x = coor[0]
        y = coor[1]

        x_dif = abs(x_origin-x)
        y_dif = abs(y_origin-y)
        # print("x:{}, y:{}, x_origin:{}, y_origin:{}, width:{}, height:{}".format(x, y, x_origin, y_origin, width, height))
        if x_dif <  self.box_width/2 and y_dif < self.box_height/2:
            return True
        else:
            return False

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

    # def _within_box(self, box_coor, car_coor):
    #     if(car_coor[0] < box_coor[0][0] and \
    #        car_coor[0] > box_coor[1][0] and \
    #        car_coor[1] < box_coor[0][1] and \
    #        car_coor[1] > box_coor[1][1]):
    #         return True
    #     else:
    #         return False

    def get_intersection(self):
        coor = GPS.get_gps(color=self.color)
        for origin, action in self.intersections:
            if self._within_box(origin, coor):
                if time.time() < self.wait_period + self.time_stamp:
                    return False, "Too Soon", coor
                else:
                    self.time_stamp = time.time()
                    return True, action, coor
        return False, "No Intersection", coor

if __name__ == "__main__":
    intersections = Intersections()
    while True:
        is_intersection, action, coor = intersections.get_intersection()
        if is_intersection:
            print("Is Intersection:{action}, {coor}".format(action=action, coor=coor))
        else:
            print("Isn't Intersection: {coor}".format(coor=coor))
        time.sleep(1)

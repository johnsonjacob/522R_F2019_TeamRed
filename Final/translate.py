from sympy.vector import CoordSys3D
import numpy as np
import math
import cv2 as cv

class Translate:

    @staticmethod
    def get_angle(current_pos, old_pos, desired_pos):

        cur_x = current_pos[0]
        cur_y = current_pos[1]
        old_x = old_pos[0]
        old_y = old_pos[1]
        des_x = desired_pos[0]
        des_y = desired_pos[1]

        if cur_x == old_x:
            theta = 0
        else:
            theta = math.atan((cur_y-old_y)/(cur_x-old_x)) 

        print(np.degrees(theta))


        translate_m = np.array([
            [1, 0, -cur_x], 
            [0, 1, -cur_y],
            [0, 0, 1],
        ])

        rotate_m = np.array([
            [math.cos(theta), math.sin(theta), 0],
            [-math.sin(theta), math.cos(theta), 0],
            [0, 0, 1],
        ])


        des_point = np.array([
            [des_x], [des_y], [1]
        ])
        new_des_pos = Translate.translate(translate_m, des_point)


        des_point = np.array([
            [new_des_pos[0]], [new_des_pos[1]], [1]
        ])
        new_des_pos = Translate.rotate(rotate_m, des_point)

        drive_angle = int(np.degrees(math.atan(new_des_pos[0]/new_des_pos[1])))


        drive_angle = abs(drive_angle)

        if drive_angle > 30.0:
            drive_angle = 30.0

        if new_des_pos[0] < 0.0:
            drive_angle = -drive_angle

        return drive_angle, new_des_pos

    @staticmethod
    def translate(M, point):
        result = M.dot(point)
        result = (result[0][0], -result[1][0])
        return result

    def rotate(M, point):
        result = M.dot(point)
        result = (result[0][0], result[1][0])
        return result




if __name__ == "__main__":

    output = "Actual: \t{} Degrees \t{} \nDesired: \t{} Degrees \t{} \n\n"

    tests = [
        {
            "current_pos"   : (4, 4),
            "old_pos"       : (4, 6),
            "desired_pos"   : (2, 2),
            "correct_pos"   : (-2, 2),
            "correct_angle" : -45,
        },

        {
            "current_pos"   : (4, 4),
            "old_pos"       : (4, 6),
            "desired_pos"   : (6, 2),
            "correct_pos"   : (2, 2),
            "correct_angle" : 45,
        },

        {
            "current_pos"   : (4, 4),
            "old_pos"       : (4, 6),
            "desired_pos"   : (2, 6),
            "correct_pos"   : (-2, -2),
            "correct_angle" : -135,
        },

        {
            "current_pos"   : (4, 4),
            "old_pos"       : (4, 6),
            "desired_pos"   : (6, 6),
            "correct_pos"   : (2, -2),
            "correct_angle" : 135,
        },

        {
            "current_pos"   : (4, 4),
            "old_pos"       : (6, 6),
            "desired_pos"   : (2, 2),
            "correct_pos"   : (0, 2.82),
            "correct_angle" : 0,
        },

        {
            "current_pos"   : (4, 4),
            "old_pos"       : (6, 6),
            "desired_pos"   : (6, 2),
            "correct_pos"   : (2.82, 0),
            "correct_angle" : 90,
        },

        {
            "current_pos"   : (4, 4),
            "old_pos"       : (6, 6),
            "desired_pos"   : (2, 6),
            "correct_pos"   : (-2.82, 0),
            "correct_angle" : -90,
        },

        {
            "current_pos"   : (4, 4),
            "old_pos"       : (6, 6),
            "desired_pos"   : (6, 6),
            "correct_pos"   : (0, -2.82),
            "correct_angle" : 180,
        },

        {
            "current_pos"   : (4, 4),
            "old_pos"       : (6, 6),
            "desired_pos"   : (5, 1),
            "correct_pos"   : (2.82, 1.41),
            "correct_angle" : "XX",
        },

        {
            "current_pos"   : (4, 4),
            "old_pos"       : (6, 6),
            "desired_pos"   : (3, 1),
            "correct_pos"   : (1.41, 2.82),
            "correct_angle" : "XX",
        },

        {
            "current_pos"   : (4, 4),
            "old_pos"       : (6, 6),
            "desired_pos"   : (0, 5),
            "correct_pos"   : (-3.53, 2.12),
            "correct_angle" : "XX",
        },

        {
            "current_pos"   : (4, 4),
            "old_pos"       : (6, 6),
            "desired_pos"   : (3, 8),
            "correct_pos"   : (-3.53, -2.12),
            "correct_angle" : "XX",
        },


        {
            "current_pos"   : (4, 4),
            "old_pos"       : (6, 6),
            "desired_pos"   : (9, 3),
            "correct_pos"   : (4.24, -2.82),
            "correct_angle" : "XX",
        },
    ]


    for test in tests:
        angle, pos = Translate.get_angle(test["current_pos"], test["old_pos"], test["desired_pos"])
        print(output.format(angle, pos, test["correct_angle"], test["correct_pos"]))


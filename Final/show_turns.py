from turns import *
import cv2 
	
path = "Global_rotated.jpg"
image = cv2.imread(path) 

radius = 2
thickness = 2

COLORS = {
    "0" : (0, 0, 0), 
    "1" : (0, 0, 255), 
    "2" : (0, 255, 0), 
    "3" : (0, 255, 255), 
    "4" : (255, 0, 0), 
    "5" : (255, 0, 255), 
    "6" : (255, 255, 0), 
    "7" : (255, 255, 255), 
}

for turn in TURNS:
    color = COLORS[turn]
    for direction in TURNS[turn]:
        for coor in TURNS[turn][direction]:
            image = cv2.circle(image, coor, radius, color, thickness) 

# Displaying the image 
cv2.imshow("image", image) 
cv2.waitKey()


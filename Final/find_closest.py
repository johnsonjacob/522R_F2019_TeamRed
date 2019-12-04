import cv2
from turns import *
from scipy.spatial import distance

closest = None
# POINT_SET = RIGHT_TURNS

def find_closest(point, ):
    distance_values = distance.cdist([point], TURNS[turn][direction])
    closest_index = distance_values.argmin()
    print(distance_values[0][closest_index])
    return POINT_SET[closest_index]

def click_and_crop(event, x, y, flags, param):
    global points, closest

    if event == cv2.EVENT_LBUTTONDOWN:
        # cv2.circle(image, (x, y), 2, (0, 255, 0), 2) 

        if closest is not None:
            cv2.circle(image, closest, 2, (255, 0, 0), 2) 

        closest = find_closest((x, y))
        cv2.circle(image, closest, 2, (0, 0, 255), 2) 
        cv2.imshow("image", image) 



# load the image, clone it, and setup the mouse callback function
path = "Global_rotated.jpg"
image = cv2.imread(path) 
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

for coor in POINT_SET:
    image = cv2.circle(image, coor, 2, (255, 0, 0), 2) 

clone = image.copy()

# keep looping until the 'q' key is pressed
while True:
    # display the image and wait for a keypress
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF

    # if the 'r' key is pressed, reset the cropping region
    if key == ord("r"):
        image = clone.copy()

    # if the 'c' key is pressed, break from the loop
    elif key == ord("q"):
        break

cv2.destroyAllWindows()

# import the necessary packages
import argparse
import cv2

points = []

def click_and_crop(event, x, y, flags, param):
    global points

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(image, (x, y), 2, (255, 0, 0), 2) 
        cv2.imshow("image", image) 



# load the image, clone it, and setup the mouse callback function
path = "Global_rotated.jpg"
image = cv2.imread(path) 
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

# keep looping until the 'q' key is pressed
while True:
    # display the image and wait for a keypress
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF

    # if the 'r' key is pressed, reset the cropping region
    if key == ord("r"):
        points = []
        image = clone.copy()

    # if the 'c' key is pressed, break from the loop
    elif key == ord("q"):
        break

cv2.destroyAllWindows()

print("POINTS = [")
for point in points:
    print("    {},".format(point))
print("]")
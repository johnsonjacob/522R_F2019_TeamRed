import cv2
import numpy as np
from PIL import Image



class Image_Process:

    def __init__(self, right_side=False, fit_degree=1, look_ahead_pixels=25, small_img_size=300):
        self.degree = fit_degree
        self.look_ahead = look_ahead_pixels
        self.right_side=right_side
        self.small_img_size=small_img_size
        self.subcross = np.loadtxt('../subcross.txt')
        self.subcross = self.subcross.astype('uint8')


    def remove_horizontal(self, inv_image):
        image = np.ones_like(inv_image) * 255
        image[inv_image > 200] = 0
        thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(image, [c], -1, (255, 255, 255), 2)
        repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,6))
        #result = 255 - cv2.morphologyEx(255 - image, cv2.MORPH_CLOSE, repair_kernel, iterations=1)
        result = image
        inv_result = np.zeros_like(result)
        inv_result[result < 200] = 255
        return inv_result


    def _perspective_warp(self, img,
                          dst_size=(4200, 2800),
                          src=np.float32([(0.3133, 0.5278), (0.75938, 0.55), (-0.183995, 1), (1.32555, 1)]),
                          dst=np.float32([(0, 0), (1, 0), (0, 1), (1, 1)])):
        img_size = np.float32([(img.shape[1], img.shape[0])])
        src = src * img_size
        dst = dst * np.float32(dst_size)
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, dst_size)
        #cv2.imshow("warped", warped)
        #cv2.waitKey()
        return warped


    def _region_of_interest(self, img, vertices):
        """
        Applies an image mask.
        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        # defining a blank mask to start with
        mask = np.zeros_like(img)

        # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vertices, ignore_mask_color)

        # returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image


    def _pipe(self, img):

        RGB_image = img

        HSV_image = cv2.cvtColor(RGB_image, cv2.COLOR_BGR2HSV)
        image_GRAY = cv2.cvtColor(RGB_image, cv2.COLOR_RGB2GRAY)


        lower_yellow = np.array([23, 100, 100], dtype="uint8")
        upper_yellow = np.array([60, 255, 255], dtype="uint8")

        lower_white = np.array([0, 0, 0], dtype="uint8")
        upper_white = np.array([0, 0, 255], dtype="uint8")

        yellow_mask = cv2.inRange(HSV_image, lower_yellow, upper_yellow)
        # white_mask = cv2.inRange(HSV_image, lower_white, upper_white)
        white_mask = cv2.inRange(image_GRAY, 155, 255)

        #white_shape = white_mask.shape
        #lower_left = [0, white_shape[0]]
        #lower_right = [white_shape[1], white_shape[0]]
        #top_left = [0, white_shape[0] / 5]
        #top_right = [white_shape[1], white_shape[0] / 5]
        #vertices = [np.array([lower_left, top_left, top_right, lower_right], dtype=np.int32)]
        #white_mask = self._region_of_interest(white_mask, vertices)

        #yellow_shape = yellow_mask.shape
        #lower_left = [yellow_shape[1] / 9, yellow_shape[0]]
        #lower_right = [yellow_shape[1] - yellow_shape[1] / 9, yellow_shape[0]]
        #top_left = [yellow_shape[1] / 9, yellow_shape[0] / 3]
        #top_right = [yellow_shape[1] - yellow_shape[1] / 9, yellow_shape[0] / 3]
        #vertices = [np.array([lower_left, top_left, top_right, lower_right], dtype=np.int32)]
        #yellow_mask = self._region_of_interest(yellow_mask, vertices)

        #white_mask = self.remove_horizontal(white_mask)
        #white_mask = self.remove_horizontal(white_mask)
        #cv2.imshow("yellow", yellow_mask)
        #cv2.imshow("white", white_mask)
        #cv2.waitKey()

        return yellow_mask, white_mask

    def _find_fit(self, img, x_space):
        line_nonzero = img.nonzero()
        line_nonzeroy = np.array(line_nonzero[0])
        line_nonzerox = np.array(line_nonzero[1])

        if(len(line_nonzeroy) <= 20):
            if not self.right_side:
                return np.zeros((img.shape[0],)).astype(int), False
            else:
                return np.full((img.shape[0],), img.shape[1] -1).astype(int), False

        else:
            line_curve = np.polyfit(line_nonzeroy, line_nonzerox, self.degree)

        line_poly = 0
        for i in range(0,self.degree + 1):
            line_poly += line_curve[i] * x_space ** (self.degree - i)

        for i in range(len(line_poly)):
            if line_poly[i] >= img.shape[1]:
                line_poly[i] = img.shape[1] - 1
                self.right_side = True
            elif line_poly[i] < 0:
                line_poly[i] = 0
                self.right_side = False


        return line_poly.astype(int), True

    def _find_trajectory(self, x_space, y_space):

        line_nonzeroy = x_space
        line_nonzerox = y_space

        line_curve = np.polyfit(line_nonzeroy, line_nonzerox, self.degree)

        line_poly = 0
        for i in range(0, self.degree + 1):
            line_poly += line_curve[i] * x_space ** (self.degree - i)

        return line_poly.astype(int)


    def _white_split(self, img, middle_poly):
        zeros = np.zeros(img.shape)
        img_left = np.zeros(img.shape)
        img_right = np.zeros(img.shape)
        for i in range(0,img.shape[0]):
            img_right[i] = np.concatenate((zeros[i][:middle_poly[i]], img[i][middle_poly[i]:]),axis=None)
            img_left[i] = np.concatenate((img[i][:middle_poly[i]], zeros[i][middle_poly[i]:]),axis=None)

        return img_left, img_right


    def _get_offset(self, y_space, position_offset):
        offset = sum(y_space[self.small_img_size-self.look_ahead:self.small_img_size])/ \
                    len(y_space[self.small_img_size-self.look_ahead:self.small_img_size]) + \
                    position_offset
        return offset


    def _get_hist(self, img):
        hist = np.sum(img[img.shape[0] // 3:, :], axis=0)
        return hist


    def _distance2cross(self, img):
        pixel2inch = 0.1
        y_offset = 30
        x_offset = 75

        # temp = sio.loadmat('/Users/spencerlow/Documents/MATLAB/522R_Cars/subcross.mat')
        # np.savetxt('subcross.txt', temp['subcross'])

        subcross = np.loadtxt('subcross.txt')

        img_pad = np.pad(img, (0, int((max(self.subcross.shape)))), 'constant', constant_values=0)

        res = cv2.matchTemplate(img_pad, self.subcross, cv2.TM_CCORR)

        intersections = res
        if np.amax(intersections) > 60000000:
            intersections[intersections < 60000000] = 0
            intersections_nonzero = intersections.nonzero()
            intersections_nonzero_y = intersections_nonzero[0]
            distance = img.shape[0] - (np.amax(intersections_nonzero_y) + y_offset)
            distance = distance * pixel2inch
        else:
            distance = 1000
        return distance

    # def _distance2cross(self, img):
    #     hist = self._get_hist(img)
    #     # plt.plot(hist)
    #     # plt.show()
    #     pixel2inch = 1
    #     #return None
    #     if np.mean(hist[50:250]) > 500:
    #         # print('Cross')
    #         canny = cv2.Canny(img, 50, 150)
    #         lines = cv2.HoughLinesP(canny[:, 50:230], 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=50)
    #         distance = None
    #         if lines is not None:
    #             horizontal_lines = []
    #             for line in lines:
    #                 x1,y1,x2,y2 = line.reshape(4)
    #                 if x1 == x2:
    #                     continue
    #                 #print("x1 " + str(x1) + " x2 " + str(x2) + " y1 " + str(y1) + " y2 " + str(y2))
    #                 parameters = np.polyfit((x1,x2),(y1,y2),1)
    #                 slope = parameters[0]
    #                 if slope > -0.5 and slope < 0.5:
    #                     cv2.line(img, (line[0, 0], line[0, 1]), (line[0, 2], line[0, 3]), (0, 0, 255), 2)
    #                 horizontal_lines = np.append(horizontal_lines, (2 * img.shape[0] - line[0, 1] - line[0, 3]) / 2)
    #             if horizontal_lines != []:
    #                 closest_line = np.amin(horizontal_lines)
    #                 distance = closest_line * pixel2inch
    #             else:
    #                 distance = None
    #         else:
    #             distance = None
    #     else:
    #         distance = None
    #     return distance

    def offset_detect(self, image):
        dst_size=(self.small_img_size,  self.small_img_size)

        image = self._perspective_warp(image, dst_size)

        yellow, white = self._pipe(image)
        ret, thresh = cv2.threshold(white, 127, 255, 0)
        throwaway, contours, heirarchy = cv2.findContours(thresh,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        example = np.array([[[182, 229]],
        [[182, 230]],
        [[181, 231]],
        [[181, 234]],
        [[180, 235]],
        [[180, 241]],
        [[179, 242]],
        [[179, 245]],
        [[178, 246]],
        [[178, 252]],
        [[177, 253]],
        [[177, 266]],
        [[176, 267]],
        [[176, 271]],
        [[175, 272]],
        [[175, 273]],
        [[177, 275]],
        [[181, 275]],
        [[182, 276]],
        [[192, 276]],
        [[193, 275]],
        [[196, 275]],
        [[196, 274]],
        [[197, 273]],
        [[197, 270]],
        [[198, 269]],
        [[198, 264]],
        [[199, 263]],
        [[199, 256]],
        [[200, 255]],
        [[200, 249]],
        [[201, 248]],
        [[201, 237]],
        [[202, 236]],
        [[201, 235]],
        [[201, 231]],
        [[200, 232]],
        [[199, 231]],
        [[198, 232]],
        [[197, 232]],
        [[196, 231]],
        [[194, 231]],
        [[193, 230]],
        [[192, 230]],
        [[191, 231]],
        [[190, 231]],
        [[189, 230]],
        [[185, 230]],
        [[184, 229]]])

        contour_count = 0

        for i in range(len(contours)):
            cv2.drawContours(image, contours[i], -1, (0,255,0), 3)
            if (cv2.matchShapes(example, contours[i],1,0.0)) < 0.15:
                contour_count += 1

        #print(contour_count)

        x_space = np.linspace(0, white.shape[0]-1, white.shape[0])

        #find the yellow fit
        yellow_poly, yellow_real = self._find_fit(yellow, x_space)

        #split the white into left and right spaces based on where the yellow line is
        white_left, white_right = self._white_split(white, yellow_poly)
        #print(white_right.shape)
        #get the right white fit line
        #lines = cv2.HoughLinesP(white_right, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=50)
        #distance = self._distance2cross(white)
        
        #print(distance)
        
        right_poly, white_right_real = self._find_fit(white_right, x_space)
        #if len(lines) > 2:
            #print("hi") 
        if yellow_real & white_right_real:
            #if self._distance2cross(white) > 14:
            #print("trajectory ", end=" ")
            #get the trajectory polygon
            trajectory = (np.transpose(yellow_poly) + np.transpose(right_poly)) / 2
            trajectory_poly = self._find_trajectory(x_space, trajectory)
            offset = self._get_offset(trajectory_poly, 0)

        if yellow_real:
            #print("yellow line", end=" ")
            offset = self._get_offset(yellow_poly, self.small_img_size/4)

        elif white_right_real: # and self._distance2cross(white) > 14:
            #print("right  line", end=" ")
            offset = self._get_offset(right_poly, -self.small_img_size/4)

        #elif white_right_real and self._distance2cross(white) < 14:
            #offset = self._get_offset(right_poly[0:50], -self.small_img_size/4)

        else:
            #print("left   line", end=" ")
            #get the left white fit line
            left_poly, white_left_real = self._find_fit(white_left, x_space)
            if white_left_real:
                offset = self._get_offset(left_poly, -self.small_img_size/4)
            else:
                #print("goin' blind", end=" ")
                offset = self._get_offset(yellow_poly, self.small_img_size/4)

        #d2c = self._distance2cross(white)    
        
        
        yellow_white_mask = cv2.bitwise_or(yellow, white)
        yellow_white_mask = cv2.cvtColor(yellow_white_mask, cv2.COLOR_GRAY2RGB)
        yellow_white_image = cv2.bitwise_and(image, yellow_white_mask)
        image = yellow_white_image

        return offset, image, white, yellow


if __name__ == "__main__":
    detection = Image_Process()
    # image = # TODO: figure out how to get a test image here
    # print(detection.offset_detect(image))

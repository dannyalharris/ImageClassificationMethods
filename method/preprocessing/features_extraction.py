import cv2
import numpy as np


class ExtractFeatures:
    def __init__(self, image_src, show_image=None):
        self.image_src = image_src
        self.to_show_image = show_image

    @property
    def image(self):
        try:
            return self.to_show_image
        except AttributeError as e:
            return self.image_src

    def preprocess(self):
        ...


class CornersDetection(ExtractFeatures):
    def __init__(self, image_gray, show_image=None):
        ExtractFeatures.__init__(self, image_gray, show_image)
        self.number_of_corners = 0

    def get_corners(self):
        # get corners from good feature to track
        corners = cv2.goodFeaturesToTrack(np.float32(self.image_src), 100, 0.01, 10)
        corners = np.int0(corners)
        self.number_of_corners = len(corners)
        if self.to_show_image is not None:
            for corner in corners:
                x, y = corner.ravel()
                cv2.circle(self.to_show_image, (x, y), 3, (80, 127, 255), 2)
        return corners

    def preprocess(self):
        self.get_corners()


class CornerHarrisDetection(ExtractFeatures):
    def __init__(self, image_grey, show_image=None):
        ExtractFeatures.__init__(self, image_grey, show_image)
        self.number_of_corners = 0

    def get_harris_corners(self):
        # get corners from corner harris
        h_corners = cv2.cornerHarris(np.float32(self.image_src), 2, 3, 0.04)
        h_corners = np.int0(h_corners)
        h_threshold = 0.05
        row = np.where(h_corners > h_corners.max()*h_threshold)[0]
        column = np.where(h_corners > h_corners.max()*h_threshold)[1]
        self.number_of_corners = len(row)

        for y, x in zip(row, column):
            if self.to_show_image is not None:
                cv2.circle(self.to_show_image, (x, y), 1, (255, 100, 0), 1)
        return h_corners

    def preprocess(self):
        self.get_harris_corners()


class ContoursDetection(ExtractFeatures):
    def __init__(self, thresholded_image, show_image=None):
        ExtractFeatures.__init__(self, thresholded_image, show_image)
        self.biggest_contour = None
        self.contours_numbers = None
        self.main_features = dict()
        self.rect_area = 0
        self.hull_area = 0
        self.contour_perimeters = 0
        self.approximation_area = 0
        self.wide = 0
        self.length = 0

    def contour_features(self):
        # make rectangle that boxing the biggest contour
        rect = cv2.minAreaRect(self.biggest_contour)
        self.wide = rect[1][0]
        self.length = rect[1][1]
        self.rect_area = self.wide * self.length
        # make hull that surround the biggest contour
        hull = cv2.convexHull(self.biggest_contour)
        self.hull_area = cv2.contourArea(hull)
        # calculate the perimeters of the biggest contour
        self.contour_perimeters = cv2.arcLength(self.biggest_contour, True)
        # calculate approximation area of the biggest contour
        approx = cv2.approxPolyDP(self.biggest_contour, 0.001 * self.contour_perimeters, True)
        self.approximation_area = cv2.contourArea(approx)

        if self.to_show_image is not None:
            # draw biggest contour in orange!
            cv2.drawContours(self.to_show_image, self.biggest_contour, -1, (0, 100, 255), 2, cv2.LINE_AA)

            # draw the box in red
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(self.to_show_image, [box], 0, (0, 0, 255), 2)

            # draw the hull in blue
            cv2.drawContours(self.to_show_image, [hull], 0, (255, 0, 0), 2)

            # draw the approx perimeters in green
            cv2.drawContours(self.to_show_image, [approx], -1, (0, 255, 0), 3)

        self.main_features = {"rect_area": self.rect_area,
                              "hull_area": self.hull_area,
                              "apprx_area": self.approximation_area,
                              "perimeters_length": self.contour_perimeters}
        return self.main_features

    def get_biggest_contour(self):
        contours, _ = cv2.findContours(self.image_src, cv2.RETR_TREE, cv2.cv2.CHAIN_APPROX_NONE)
        # exclude the contour of the image frame
        im_boundary = (self.image_src.shape[0] - 1) * (self.image_src.shape[1] - 1)
        areas = [cv2.contourArea(ar) for ar in contours]
        cnt = [x for x in areas if x != im_boundary]
        # get the biggest contour
        self.contours_numbers = len(contours)
        self.biggest_contour = contours[areas.index(max(cnt))]
        return self.biggest_contour

    def preprocess(self):
        self.get_biggest_contour()
        self.contour_features()


def thresholding_image(img_gray, threshold):
    _, img_thresh = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    return img_thresh

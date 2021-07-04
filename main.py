from pprint import pprint

import os
import cv2
from csv import DictWriter
from csv import writer
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

image_folder = "resources\\Kamm"
list = os.listdir(image_folder)
number_files = len(list)
print(number_files)
images_list = []
valid_images = [".jpg",".png"]

for f in os.listdir(image_folder):
    images_list.append(os.path.join(image_folder,f))

for image_path in images_list:
    #print(image_path)
    image = cv2.imread(image_path)
    image_copy = image.copy()
    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    dark_bright_threshold = 200
    mean_of_gray_image = np.mean(image_gray)

    #Select threshold for further operation based on image brightness
    if mean_of_gray_image < dark_bright_threshold:
        # Image i dark
        contour_recognition_threshold = 120
        #print("Dark Image")
    else:
        #Image is bright
        contour_recognition_threshold = 200
        #print("Light Image")

    _, image_thresh = cv2.threshold(image_gray, contour_recognition_threshold, 255, cv2.THRESH_BINARY)

    plt.subplot(1,3,1)
    plt.imshow(image)
    plt.subplot(1, 3, 2)
    plt.imshow(image_gray, "gray")
    plt.subplot(1, 3, 3)
    plt.imshow(image_thresh, "gray")

    plt.subplot(1,2,1)
    plt.imshow(image_thresh, "gray")
    kernel = np.ones((3, 3), np.uint8)
    image_thresh = cv2.morphologyEx(image_thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    plt.subplot(1, 2, 2)
    plt.imshow(image_thresh, "gray")

    contours, _ = cv2.findContours(image_thresh, cv2.RETR_TREE, cv2.cv2.CHAIN_APPROX_NONE)

    #i = 0
    #for cnt in contours:
    #    #print(cv2.contourArea(cnt))
    #    im = image_copy.copy()
    #    cv2.drawContours(im, cnt, -1, (0, 255, 0), 2, cv2.LINE_AA)
    #    plt.subplot(1, len(contours), i + 1)
    #    plt.imshow(im, "gray")
    #    i = i + 1

    # get greatest contour by area
    im_boundary = (image_thresh.shape[0] - 1) * (image_thresh.shape[1] - 1)
    areas = [cv2.contourArea(ar) for ar in contours]
    cnt = [x for x in areas if x != im_boundary]
    cnt = contours[areas.index(max(cnt))]
    contour_area = cv2.contourArea(cnt)
    #print("Area", contour_area)

    to_show_contour = image_copy.copy()
    cv2.drawContours(to_show_contour, cnt, -1, (0, 255, 0), 2, cv2.LINE_AA)
    plt.imshow(to_show_contour)

    rect = cv2.minAreaRect(cnt)
    rect_area = rect[1][0] * rect[1][1]
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    #print("Rectangle", rect[1][0], rect[1][1])
    to_show_box = image_copy.copy()
    cv2.drawContours(to_show_box, [box], 0, (0, 0, 255), 2)
    plt.imshow(to_show_box)

    a = rect[1][0] / rect[1][1]
    #print(a)

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    to_show_hull = image_copy.copy()
    cv2.drawContours(to_show_hull, [hull], 0, (255, 0, 0), 2)
    plt.imshow(to_show_hull)

    contour_perimeters = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.001 * contour_perimeters, True)
    approximation_area = cv2.contourArea(approx)
    to_show_approx = image_copy.copy()
    cv2.drawContours(to_show_approx, [approx], -1, (0, 0, 255), 3)
    plt.imshow(to_show_approx)

    # Detect corners from grayscale image
    corners = cv2.goodFeaturesToTrack(np.float32(image_gray), 100, 0.01, 10)
    corners = np.int0(corners)
    to_show_corners = image_copy.copy()
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(to_show_corners, (x, y), 3, (80, 127, 255), 2)
    plt.imshow(to_show_corners)
    #print("good corners", len(corners))

    # Detect corners from grayscale image
    corners = cv2.goodFeaturesToTrack(np.float32(image_gray), 100, 0.01, 10)
    corners = np.int0(corners)
    to_show_corners = image_copy.copy()
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(to_show_corners, (x, y), 3, (80, 127, 255), 2)
    plt.imshow(to_show_corners)
    #print("good corners", len(corners))

    h_corners = cv2.cornerHarris(np.float32(image_gray), 2, 3, 0.04)
    h_corners = np.int0(h_corners)
    to_show_corners_harris = image_copy.copy()
    h_threshold = 0.05
    for i in range(h_corners.shape[0]):
        for j in range(h_corners.shape[1]):
            if h_corners[i, j] > h_corners.max() * h_threshold:
                cv2.circle(to_show_corners_harris, (j, i), 1, (0, 0, 255), 1)
    plt.imshow(to_show_corners_harris)
    amount_h_corners = len(h_corners[h_corners > h_corners.max() * h_threshold])
    #print("harris corners", amount_h_corners)

    index = image_path.lstrip(image_folder)
    index = index.lstrip('image_')
    index = index.rstrip('.png')
    # Store features as dictionary
    data = {
        "contour_points": len(cnt),
        "amount_contours": len(contours),
        "rect_area": rect_area,
        "hull_area": hull_area,
        "approximation_area": approximation_area,
        "contour_perimeters": contour_perimeters,
        "corners": len(corners),
        "harris_corners": amount_h_corners,
        "ratio_wide_length": rect[1][0] / rect[1][1],
        "contour_length_area_ratio": contour_perimeters / contour_area,
        "contour_length_rect_area_ratio": contour_perimeters / rect_area,
        "contour_length_hull_area_ratio": contour_perimeters / hull_area,
        "contour_rect_length_ratio": contour_perimeters / (2 * (rect[1][0] + rect[1][1])),
        "contour_hull_length_ratio": contour_perimeters / cv2.arcLength(hull, True),
        "extent": contour_area / rect_area,
        "solidity": contour_area / hull_area,
        "hull_rectangle_ratio": hull_area / rect_area,
        "Type": 1
    }
    print(data)
    field_names = ["contour_points", "amount_contours", "rect_area", "hull_area", "approximation_area",
        "contour_perimeters", "corners", "harris_corners","ratio_wide_length", "contour_length_area_ratio",
        "contour_length_rect_area_ratio", "contour_length_hull_area_ratio",
        "contour_rect_length_ratio", "contour_hull_length_ratio", "extent", "solidity", "hull_rectangle_ratio", "Type"]

    #data1 = [index,len(cnt),len(contours),(cv2.arcLength(cnt, True) / contour_area),
    #      cv2.arcLength(cnt, True) / rect_area,cv2.arcLength(cnt, True) / hull_area,(cv2.arcLength(cnt, True) / (2 * (rect[1][0] + rect[1][1]))),
    #      (cv2.arcLength(cnt, True) / cv2.arcLength(hull, True)),(contour_area / rect_area),(contour_area / hull_area),
    #      (rect[1][0] / rect[1][1]),(hull_area / rect_area),len(corners),amount_h_corners]

    with open('data.csv', 'a',newline='\n',encoding='utf-8')as f_object:
        # Pass the file object and a list
        # of column names to DictWriter()
        # You will get a object of DictWriter
        dictwriter_object = DictWriter(f_object, fieldnames=field_names)

        # Pass the dictionary as an argument to the Writerow()
        dictwriter_object.writerow(data)

        # Close the file object
        f_object.close()







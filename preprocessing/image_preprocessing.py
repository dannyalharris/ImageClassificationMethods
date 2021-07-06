from preprocessing.features_extraction import CornersDetection, \
	CornerHarrisDetection, ContoursDetection, thresholding_image
from sklearn.preprocessing import LabelEncoder
import csv
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ImageResizer:
	def __init__(self, width, height, inter=cv2.INTER_AREA):
		# store the target image width, height, and interpolation
		# method used when resizing
		self.width = width
		self.height = height
		self.inter = inter

	def preprocess(self, image, **kwargs):
		# resize the image to a fixed size, ignoring the aspect
		# ratio
		return cv2.resize(image, (self.width, self.height), interpolation=self.inter)


class FeatureExtraction:
	def __init__(self):
		self.csv_data = None
		self.list_of_features = []
		self.data_frame = None

	def preprocess(self, image, show=False):
		img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		mean_of_gray_image = np.mean(img_gray)

		if mean_of_gray_image < 180:
			# Image is dark
			# print("dark image")
			contour_recognition_threshold = 120
		else:
			# print("bright image")
			# Image is bright
			contour_recognition_threshold = 200
		img_thresh = thresholding_image(img_gray, contour_recognition_threshold)
		image_to_show = image.copy() if show else None
		a = CornersDetection(img_gray, image_to_show)
		b = CornerHarrisDetection(img_gray, image_to_show)
		c = ContoursDetection(img_thresh, image_to_show)
		features_extraction = [a, b, c]
		for F in features_extraction:
			F.preprocess()
		features_extracted = {
			'n_corner': a.number_of_corners,
			'n_h_corner': b.number_of_corners,
			'n_contour': c.contours_numbers,
			'a_rect': c.rect_area,
			'a_hull': c.hull_area,
			'a_approx': c.approximation_area,
			'l_perimeters': c.contour_perimeters,
			'wide/length': c.wide / c.length,
			'perim/a_rect': c.contour_perimeters / c.rect_area,
			'perim/a_hull': c.contour_perimeters / c.hull_area,
			'perim/a_appx': c.contour_perimeters / c.approximation_area,
			'corner/a_rect': a.number_of_corners / c.rect_area,
			'corner/a_hull': a.number_of_corners / c.hull_area,
			'corner/a_appx': a.number_of_corners / c.approximation_area,
			'corner/l_perim': a.number_of_corners / c.contour_perimeters,
			'h_corner/a_rect': b.number_of_corners / c.rect_area,
			'h_corner/a_hull': b.number_of_corners / c.hull_area,
			'h_corner/a_appx': b.number_of_corners / c.approximation_area,
			'h_corner/l_perim': b.number_of_corners / c.contour_perimeters,
			'extent': c.approximation_area / c.rect_area,
			'solidity': c.approximation_area / c.hull_area
		}
		if show:
			plt.imshow(image_to_show)
		self.list_of_features.append(features_extracted)

	def extract_to_table(self, files, labels, fmt="a"):
		le = LabelEncoder()
		labels = le.fit_transform(labels)
		for val in self.list_of_features:
			val["labels"] = labels[self.list_of_features.index(val)]
		keys = self.list_of_features[0].keys()
		with open(files, fmt, newline='', encoding='utf-8') as out:
			dict_writer = csv.DictWriter(out, keys)
			dict_writer.writeheader()
			for dat in self.list_of_features:
				dict_writer.writerow(dat)

	def extract_to_panda(self, labels):
		le = LabelEncoder()
		labels = le.fit_transform(labels)
		for val in self.list_of_features:
			val["labels"] = labels[self.list_of_features.index(val)]
		self.data_frame = pd.DataFrame.from_dict(self.list_of_features)
		return self.data_frame

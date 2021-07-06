import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import random


class SimpleDatasetLoader:
	def __init__(self, preprocessors=None):
		# store the image preprocessor
		self.preprocessors = preprocessors
		# if the preprocessors are None, initialize them as an empty list
		if self.preprocessors is None:
			self.preprocessors = []

	def load(self, image_paths, verbose=-1, show=-1):
		# initialize the list of features and labels
		data_image = []
		labels = []
		from preprocessing import FeatureExtraction
		cnt = 1
		# loop over the input images
		for (i, imagePath) in enumerate(image_paths):
			image = cv2.imread(imagePath)
			label = imagePath.split(os.path.sep)[-2]

			# check to see if our preprocessors are not None
			if self.preprocessors is not None:
				# loop over the preprocessors and apply each to the image
				x = random.randint(i-5, i+5)

				for p in self.preprocessors:
					if isinstance(p, FeatureExtraction) and show > 0 and x == i and cnt < show:
						row = (show // 5) + 1 if (show % 5) != 0 else (show // 5)
						plt.axis('off')
						plt.subplot(row, 5, cnt)
						image = p.preprocess(image, show=True)
						cnt += 1
					else:
						image = p.preprocess(image)
			# treat our processed image as a "feature vector"
			# by updating the data list followed by the labels
			data_image.append(image)
			labels.append(label)

			# show an update every `verbose` images
			if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
				print("[INFO] processed {}/{}".format(i + 1, len(image_paths)))

		# return a tuple of the data and labels
		return np.array(data_image), np.array(labels)

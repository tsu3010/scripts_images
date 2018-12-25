import os
import humanize
import sys
import operator
import cv2
import bz2
import logging
import numpy as np
import matplotlib as plt
import pickle
from skimage import feature
from skimage.restoration import estimate_sigma
from matplotlib import image as mpimg
from PIL import Image as img
from google.cloud import storage
from io import BytesIO
from skimage.restoration import estimate_sigma


def read_images_from_gcp(bucket_name, count_of_images):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blobs = bucket.list_blobs()
    counter = 1
    img_name_list = []
    img_files = []
    for blob in blobs:
        counter = counter + 1
        if counter > count_of_images:
            break
        img_name_list.append(blob.name)
        image_blob = bucket.get_blob(blob.name)
        #image_file = BytesIO(image_blob.download_as_string())
        nparr = np.fromstring(image_blob.download_as_string(), np.uint8)
        image_file = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        logger.info("Reading Image File %s" % (blob.name))
        img_files.append(image_file)
    return img_name_list, img_files

### Papers ###
# Does iaage quality affect NNs?
# https://arxiv.org/pdf/1604.04004.pdf
# https://stackoverflow.com/questions/2440504/noise-estimation-noise-measurement-in-image

# Things to Estimate Image Quality
# 1) Image Noise
# 2) Image Contrast
# 3) Image Blurriness


def extract_image_attributes(image):
    dims = np.shape(image)
    attributes = {"Pixel Count": dims[0] * dims[1],
                  "Height": dims[0],
                  "Width": dims[1]}
    return images_attributes

# Folliowing function is a robust Wavelet based estimator of the noise standard deviastion
# D. L. Donoho and I. M. Johnstone. “Ideal spatial adaptation by wavelet shrinkage.” Biometrika 81.3 (1994): 425-455. DOI:10.1093/biomet/81.3.425
# In the below function, higher the value, higher is the noisiness of image


def estimate_noise(img):
    return estimate_sigma(img, multichannel=True, average_sigmas=True)

# To measure the image blurrness, I refered to the following paper: "Diatom Autofocusing in Brightfield Microscopy: A Comparative Study".
# In this paper the author Pech-Pacheco et al. has provided variance of the Laplacian Filter which can be used to measure the image blurriness score.
# In this technique, the single channel of an image is convolved with the the laplacian filter. If the specified value is less than a threshold value, then image is blurry otherwise not.
# In the below function, lower the value of image higher is the blurriness


def estimate_blurriness(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(image, cv2.CV_64F).var()
    return fm

# Some images may contain no pixel variation and are entirely uniform. Average Pixel Width is a measure which indicates the amount of edges present in the image. If this number comes out to be very low, then the image is most likely a uniform image and may not represent right content. To compute this measure, we are using skimage's Canny Detection


def estimate_uniformity(image):
    im_array = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges_sigma1 = feature.canny(im_array, sigma=3)
    apw = (float(np.sum(edges_sigma1)) / (image.shape[0] * image.shape[1]))
    return apw * 100


if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    # create a file handler
    handler = logging.FileHandler(filename='file_logs.log')
    handler.setLevel(logging.INFO)
    # add the handlers to the logger
    logger.addHandler(handler)

    # Initialize bucket and count of images to read/analyze
    bucket = "batteryimages_vsi_spotwelding"
    image_count = 1000

    # Read images from GCP #
    image_names, image_files = read_images_from_gcp(bucket, image_count)

    pickle_out = bz2.open("image_files_comp.bz2", "wb")
    pickle.dump(image_files, pickle_out)
    pickle_out.close()

    pickle_names = bz2.open("image_names_comp.bz2", "wb")
    pickle.dump(image_names, pickle_names)
    pickle_names.close()
    logger.info("Pickle Files Complete")

    # Extract and Store image attributes #
    #images_attributes = [extract_image_attributes(f) for f in image_files]

    # Estimate Image Noise
    #images_noise = [estimate_noise(f) for f in image_files]

    # Estimate Image Blurriness
    #images_blurriness = [estimate_blurriness(f) for f in image_files]

    # Extract image Uniformity (Edge Detection using Canny Filters)
    #images_uniformity = [estimate_uniformity(f) for f in image_files]

# Testing
#im = img.open(image_files[0])
# print(im.size)
#im_sample = image_files[0]
## Find length of top row #
#dims = np.shape(im_sample)
#dims[0] * dims[1]
# len(im_sample[1])
#image = mpimg.imread(BytesIO(image_object.get()['Body'].read()),'bmp')

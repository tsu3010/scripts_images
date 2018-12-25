import os
import humanize
import sys
import operator
import cv2
import bz2
import logging
import numpy as np
import pandas as pd
import matplotlib as plt
from skimage import feature
from skimage.restoration import estimate_sigma
from matplotlib import image as mpimg
from PIL import Image as img
from google.cloud import storage
from io import BytesIO
from skimage.restoration import estimate_sigma
import pandas as pd


def extract_image_attributes(image):
    dims = np.shape(image)
    attributes = {"Pixel Count": dims[0] * dims[1],
                  "Height": dims[0],
                  "Width": dims[1]}
    return attributes


def estimate_noise(image):
    return estimate_sigma(image, multichannel=True, average_sigmas=True)


def estimate_blurriness(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(image, cv2.CV_64F).var()
    return fm


def estimate_uniformity(image):
    im_array = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges_sigma1 = feature.canny(im_array, sigma=3)
    apw = (float(np.sum(edges_sigma1)) / (image.shape[0] * image.shape[1]))
    return apw * 100


def read_image_extract_feature(bucket_name, count_of_images):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blobs = bucket.list_blobs()
    counter = 1
    img_name_list = []
    img_attr_dict_list = []
    img_noise_list = []
    img_blur_list = []
    img_uniformity_list = []
    for blob in blobs:
        counter = counter + 1
        if counter > count_of_images:
            break
        image_blob = bucket.get_blob(blob.name)
        nparr = np.fromstring(image_blob.download_as_string(), np.uint8)
        image_file = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        logger.info("Reading Image File %s" % (blob.name))
        image_attr_dict = extract_image_attributes(image_file)
        image_noise = estimate_noise(image_file)
        image_blur = estimate_blurriness(image_file)
        image_uniformity = estimate_uniformity(image_file)
        # Append to List
        img_name_list.append(blob.name)
        img_attr_dict_list.append(image_attr_dict)
        img_blur_list.append(image_blur)
        img_noise_list.append(image_noise)
        img_uniformity_list.append(image_uniformity)
    return img_name_list, img_attr_dict_list, img_blur_list, img_noise_list, img_uniformity_list


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
    image_count = 42000

    image_name, image_dimensions, image_blur, image_noise, image_uniformity = read_image_extract_feature(bucket, image_count)
    df_image_stats = pd.DataFrame({'Name': image_name, "Dimensions": image_dimensions, "Blurness": image_blur, "Noise": image_noise, "Uniformity": image_uniformity})

    logging.info("Writing Attributes to CSV")
    df_image_stats.to_csv("ImageAttributes.csv")
    logging.info("Attribute Generation Complete")

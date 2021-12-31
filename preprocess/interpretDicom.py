
"""
dicom format-->array of shape = （n，28，128，1）
return dataset and original images with shape(600*800)

"""

import numpy as np
import pydicom
import cv2
from tensorflow.keras import preprocessing


def parse_scale(path):
    # 解析像素和cm的转换比例
    file = pydicom.read_file(path)
    try:
        Manufacturer = file["Manufacturer"].value
    except:
        Manufacturer = None
    try:
        SoftwareVersions = file["SoftwareVersions"].value
    except:
        SoftwareVersions = None
    try:
        info = file.SequenceOfUltrasoundRegions
        x_delta = info[0]["PhysicalDeltaX"].value
        y_delta = info[0]["PhysicalDeltaY"].value
        assert x_delta == y_delta, print("scale of x:{} is not equal to y:{}".format(x_delta, y_delta))
    except:
        x_delta = 18/460 #默认18/460
    return x_delta, Manufacturer, SoftwareVersions


def interpretDicom(m,src_path):
    i = 0
    ndarray = np.ones(shape=(120, m, m, 1))  # 一般<120帧
    data = pydicom.dcmread(src_path).pixel_array[:, :, :, 0]
    for i, j in enumerate(data):
        img_cut = cv2.resize(j, (400, 300))
        array = preprocessing.image.img_to_array(img_cut)

        array = array[40:296, 88:344] / 255.0
        array = cv2.resize(array, (m, m))
        ndarray[i] = array.reshape(m, m, 1)
    return ndarray[:i], data/255


def parse_dicom(m,path):
    ndarray = np.ones(shape=(120, m, m, 1))  # 一般<120帧
    data = pydicom.dcmread(path).pixel_array[:, :, :, 0]
    i = 0
    for i, j in enumerate(data):
        img_cut = cv2.resize(j, (400, 300))
        array = preprocessing.image.img_to_array(img_cut)

        array = array[40:296, 88:344] / 255.0
        array = cv2.resize(array, (m, m))
        ndarray[i] = array.reshape(m, m, 1)
    return ndarray[:i], data/255